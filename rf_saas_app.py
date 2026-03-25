"""
Resonance Folding — Model Merge Service v0.2.0
===============================================
FastAPI backend. New in v0.2.0:
  - POST /sweep now accepts an optional val_zip (ImageFolder layout)
    and returns accuracy at each t — finds optimal t automatically
  - POST /continual merges a model toward a base checkpoint (SLERP-CL step)
  - UI updated with accuracy curve chart on sweep tab

Run:
  pip install fastapi uvicorn python-multipart resonance-folding
  uvicorn rf_saas_app:app --reload --port 8000
"""

import io
import os
import json
import copy
import math
import tempfile
import traceback
import zipfile as zipmod
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import (HTMLResponse, JSONResponse, StreamingResponse)

from resonance_folding import (
    OctConvNet, OctResNet18,
    fold_model, verify_fold, slerp_merge, float_average,
)

app = FastAPI(title="Resonance Folding", version="0.2.0")

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
UPLOAD_DIR = Path(tempfile.mkdtemp())


# ─────────────────────────────────────────────────────────────
#  MODEL LOADING
# ─────────────────────────────────────────────────────────────

def load_checkpoint(path: str):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    if isinstance(ckpt, torch.nn.Module):
        return ckpt.to(DEVICE), "unknown"
    if isinstance(ckpt, dict):
        sd = ckpt.get("model", ckpt.get("state_dict", ckpt))
    else:
        raise ValueError("Unrecognised checkpoint format")

    if "layer1.0.conv1.weight" in sd:
        n_classes = sd.get("fc.weight", torch.zeros(10, 1)).shape[0]
        model = OctResNet18(n_classes=n_classes, cifar=True).to(DEVICE)
        arch  = f"OctResNet18 ({n_classes} classes)"
    else:
        c3 = sd.get("stage3.1.conv.weight",
                    torch.zeros(128, 128, 3, 3)).shape[0]
        size = {64: "S", 128: "M", 256: "L"}.get(c3, "M")
        head_key = next((k for k in sd if "head" in k and k.endswith(".weight")
                         and len(sd[k].shape) == 2), None)
        n_classes = sd[head_key].shape[0] if head_key else 10
        model = OctConvNet(size=size, n_classes=n_classes).to(DEVICE)
        arch  = f"OctConvNet-{size} ({n_classes} classes)"

    model.load_state_dict(sd, strict=False)
    return model, arch


def save_checkpoint(model, path: str, meta: dict = None):
    payload = {"state_dict": model.state_dict()}
    if meta:
        payload.update(meta)
    torch.save(payload, path)


# ─────────────────────────────────────────────────────────────
#  VALIDATION ACCURACY
# ─────────────────────────────────────────────────────────────

def eval_accuracy(model, val_dir: str) -> float:
    """Evaluate model accuracy on an ImageFolder-layout val directory."""
    import torchvision
    import torchvision.transforms as T
    tf = T.Compose([
        T.Resize(36), T.CenterCrop(32),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    ds     = torchvision.datasets.ImageFolder(val_dir, transform=tf)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=256, shuffle=False, num_workers=0)
    model.eval()
    corr = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            corr  += (model(x).argmax(1) == y).sum().item()
            total += len(y)
    return round(corr / total, 4) if total > 0 else 0.0


def extract_val_zip(upload: UploadFile, dest: Path) -> str | None:
    """Extract a val zip (ImageFolder layout) and return the root dir."""
    if upload is None or upload.filename == "":
        return None
    data = upload.file.read()
    with zipmod.ZipFile(io.BytesIO(data)) as z:
        z.extractall(dest)
    # Find the top-level folder
    tops = [p for p in dest.iterdir() if p.is_dir()]
    return str(tops[0]) if tops else str(dest)


# ─────────────────────────────────────────────────────────────
#  ENDPOINTS
# ─────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "0.2.0",
        "device": DEVICE,
        "cuda":   torch.cuda.is_available(),
        "gpu":    torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


@app.post("/verify")
async def verify_endpoint(model_file: UploadFile = File(...)):
    path = UPLOAD_DIR / f"v_{model_file.filename}"
    try:
        path.write_bytes(await model_file.read())
        model, arch = load_checkpoint(str(path))
        folded = fold_model(model)
        result = verify_fold(model, folded, verbose=False)
        layer_dict = dict(model.oct_layers())
        layers = []
        for name, fl in folded.items():
            W     = layer_dict[name].weight.data
            recon = fl.decode()
            cos   = F.cosine_similarity(
                W.float().flatten().unsqueeze(0),
                recon.float().flatten().unsqueeze(0)).item()
            layers.append({
                "name":     name,
                "shape":    list(W.shape),
                "n_groups": fl.n_groups,
                "cos":      round(cos, 6),
                "holo":     round(fl.holo, 10),
            })
        return JSONResponse({
            "architecture": arch,
            "n_layers":     result["n_layers"],
            "n_oct_groups": sum(l["n_groups"] for l in layers),
            "mean_cos":     round(result["mean_cos"], 6),
            "min_cos":      round(result["min_cos"], 6),
            "mean_holo":    result["mean_holo"],
            "lossless":     result["all_lossless"],
            "layers":       layers,
        })
    except Exception as e:
        raise HTTPException(500, traceback.format_exc())
    finally:
        if path.exists(): path.unlink()


@app.post("/merge")
async def merge_endpoint(
    model_a: UploadFile = File(...),
    model_b: UploadFile = File(...),
    t:       float = Form(0.20),
    method:  str   = Form("slerp"),
):
    path_a = UPLOAD_DIR / f"ma_{model_a.filename}"
    path_b = UPLOAD_DIR / f"mb_{model_b.filename}"
    path_m = UPLOAD_DIR / "merged.pth"
    try:
        path_a.write_bytes(await model_a.read())
        path_b.write_bytes(await model_b.read())
        ma, arch_a = load_checkpoint(str(path_a))
        mb, arch_b = load_checkpoint(str(path_b))

        if method == "slerp":
            merged, holo = slerp_merge(ma, mb, t)
            meta = {"method": "oct-slerp", "t": t, "holo": holo,
                    "arch_a": arch_a, "arch_b": arch_b}
        else:
            merged = float_average(ma, mb)
            holo   = 0.0
            meta   = {"method": "float-average",
                      "arch_a": arch_a, "arch_b": arch_b}

        save_checkpoint(merged, str(path_m), meta)

        def iterfile():
            with open(path_m, "rb") as f:
                yield from f

        return StreamingResponse(
            iterfile(),
            media_type="application/octet-stream",
            headers={"Content-Disposition": "attachment; filename=merged.pth"},
        )
    except Exception as e:
        raise HTTPException(500, traceback.format_exc())
    finally:
        for p in [path_a, path_b, path_m]:
            if p.exists(): p.unlink()


@app.post("/sweep")
async def sweep_endpoint(
    model_a:  UploadFile = File(...),
    model_b:  UploadFile = File(...),
    n_steps:  int  = Form(21),
    val_zip:  UploadFile = File(None),
):
    """
    SLERP sweep t=0..1.
    If val_zip supplied (ImageFolder layout): returns accuracy at each t.
    Otherwise: returns holo only (algebraic integrity check).
    """
    path_a   = UPLOAD_DIR / f"sa_{model_a.filename}"
    path_b   = UPLOAD_DIR / f"sb_{model_b.filename}"
    val_dir  = UPLOAD_DIR / "val_extracted"
    val_dir.mkdir(exist_ok=True)

    try:
        path_a.write_bytes(await model_a.read())
        path_b.write_bytes(await model_b.read())

        ma, arch_a = load_checkpoint(str(path_a))
        mb, arch_b = load_checkpoint(str(path_b))

        # Float average baseline
        mf      = float_average(ma, mb)
        has_val = val_zip is not None and val_zip.filename != ""
        val_root = None

        if has_val:
            val_data = await val_zip.read()
            with zipmod.ZipFile(io.BytesIO(val_data)) as z:
                z.extractall(val_dir)
            tops = [p for p in val_dir.iterdir() if p.is_dir()]
            val_root = str(tops[0]) if tops else str(val_dir)
            acc_fa = eval_accuracy(mf, val_root)
            acc_a  = eval_accuracy(ma, val_root)
            acc_b  = eval_accuracy(mb, val_root)
        else:
            acc_fa = acc_a = acc_b = None

        results   = []
        best_t    = 0.0
        best_acc  = -1.0
        ts = [i / (n_steps - 1) for i in range(n_steps)]

        for t in ts:
            merged, holo = slerp_merge(ma, mb, t)
            acc = eval_accuracy(merged, val_root) if has_val else None
            results.append({
                "t":    round(t, 3),
                "holo": round(holo, 10),
                "acc":  acc,
            })
            if acc is not None and acc > best_acc:
                best_acc = acc
                best_t   = t

        response = {
            "architecture_a": arch_a,
            "architecture_b": arch_b,
            "has_accuracy":   has_val,
            "n_steps":        n_steps,
            "results":        results,
        }

        if has_val:
            response["baselines"] = {
                "model_a":     acc_a,
                "model_b":     acc_b,
                "float_avg":   acc_fa,
            }
            response["best"] = {
                "t":           best_t,
                "acc":         best_acc,
                "vs_float":    round(best_acc - acc_fa, 4),
                "vs_best_ind": round(best_acc - max(acc_a, acc_b), 4),
            }

        return JSONResponse(response)

    except Exception as e:
        raise HTTPException(500, traceback.format_exc())
    finally:
        for p in [path_a, path_b]:
            if p.exists(): p.unlink()


@app.post("/continual")
async def continual_endpoint(
    current_model: UploadFile = File(...),
    base_model:    UploadFile = File(...),
    t:             float = Form(0.3),
):
    """
    SLERP-CL step: merge current model back toward base.
    Use after each fine-tuning step to resist catastrophic forgetting.
    t=0.3 means 30% new task, 70% base knowledge preserved.
    """
    path_c = UPLOAD_DIR / f"cl_cur_{current_model.filename}"
    path_b = UPLOAD_DIR / f"cl_base_{base_model.filename}"
    path_m = UPLOAD_DIR / "cl_merged.pth"
    try:
        path_c.write_bytes(await current_model.read())
        path_b.write_bytes(await base_model.read())
        mc, arch_c = load_checkpoint(str(path_c))
        mb, arch_b = load_checkpoint(str(path_b))

        # SLERP: t toward current (new task), (1-t) toward base (old knowledge)
        merged, holo = slerp_merge(mb, mc, t)
        save_checkpoint(merged, str(path_m), {
            "method": "slerp-cl", "t": t, "holo": holo,
        })

        def iterfile():
            with open(path_m, "rb") as f:
                yield from f

        return StreamingResponse(
            iterfile(),
            media_type="application/octet-stream",
            headers={"Content-Disposition":
                     "attachment; filename=slerp_cl_merged.pth"},
        )
    except Exception as e:
        raise HTTPException(500, traceback.format_exc())
    finally:
        for p in [path_c, path_b, path_m]:
            if p.exists(): p.unlink()


# ─────────────────────────────────────────────────────────────
#  WEB UI
# ─────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(UI_HTML)


UI_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Resonance Folding</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f3f4f6;color:#111;min-height:100vh}
.header{background:#0f2744;color:#fff;padding:1.25rem 2rem;display:flex;align-items:center;gap:1rem}
.header h1{font-size:1.3rem;font-weight:600}
.header p{font-size:.8rem;opacity:.75;margin-top:2px}
.badge{background:rgba(255,255,255,.15);border-radius:20px;padding:2px 10px;font-size:.75rem;margin-left:auto}
.container{max-width:860px;margin:1.75rem auto;padding:0 1.25rem}
.card{background:#fff;border-radius:12px;border:1px solid #e2e4e9;padding:1.5rem;margin-bottom:1.25rem}
.card h2{font-size:.95rem;font-weight:600;color:#0f2744;margin-bottom:1rem}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:1rem}
label{display:block;font-size:.78rem;font-weight:500;color:#6b7280;margin-bottom:4px}
input[type=file],input[type=number],select{width:100%;padding:8px 10px;border:1px solid #d1d5db;border-radius:8px;font-size:.85rem;background:#fff}
input[type=range]{width:100%;accent-color:#0f2744}
.btn{background:#0f2744;color:#fff;border:none;padding:9px 22px;border-radius:8px;font-size:.85rem;font-weight:500;cursor:pointer}
.btn:hover{background:#1a3a5c}
.btn:disabled{background:#9ca3af;cursor:not-allowed}
.result{border-radius:8px;padding:1rem;margin-top:1rem;font-size:.83rem;line-height:1.6}
.ok{background:#f0fdf4;border:1px solid #86efac}
.err{background:#fef2f2;border:1px solid #fca5a5}
.info{background:#eff6ff;border:1px solid #bfdbfe}
.chip{display:inline-block;background:#e8f0f8;border-radius:6px;padding:2px 9px;margin:2px;font-size:.78rem;font-weight:500;color:#0f2744}
.chip.g{background:#dcfce7;color:#166534}
.spinner{display:none;width:16px;height:16px;border:2px solid #fff;border-top-color:transparent;border-radius:50%;animation:spin .6s linear infinite;vertical-align:middle;margin-left:8px}
@keyframes spin{to{transform:rotate(360deg)}}
.tabs{display:flex;gap:0;border-bottom:2px solid #e5e7eb;margin-bottom:1.5rem}
.tab{padding:8px 18px;cursor:pointer;font-size:.875rem;font-weight:500;color:#6b7280;border-bottom:2px solid transparent;margin-bottom:-2px}
.tab.active{color:#0f2744;border-bottom-color:#0f2744}
.panel{display:none}.panel.active{display:block}
canvas{width:100%!important}
.t-big{font-size:1.15rem;font-weight:600;color:#0f2744;text-align:center;margin:4px 0}
</style>
</head>
<body>
<div class="header">
  <div><h1>Resonance Folding</h1><p>Geodesic model merging on S⁷ · oct-SLERP</p></div>
  <span class="badge" id="dev">loading...</span>
</div>
<div class="container">
  <div class="tabs">
    <div class="tab active" onclick="tab('merge')">Merge</div>
    <div class="tab" onclick="tab('sweep')">Sweep</div>
    <div class="tab" onclick="tab('verify')">Verify</div>
    <div class="tab" onclick="tab('cl')">SLERP-CL</div>
    <div class="tab" onclick="tab('about')">About</div>
  </div>

  <!-- MERGE -->
  <div class="panel active" id="panel-merge">
    <div class="card">
      <h2>Merge two checkpoints via oct-SLERP</h2>
      <div class="grid2">
        <div><label>Model A (.pth)</label><input type="file" id="m-a" accept=".pth,.pt"></div>
        <div><label>Model B (.pth)</label><input type="file" id="m-b" accept=".pth,.pt"></div>
      </div>
      <div style="margin-top:1rem">
        <label>t &nbsp;<small style="color:#9ca3af">0=A · 1=B · 0.20=proven optimal</small></label>
        <input type="range" id="m-t" min="0" max="1" step="0.025" value="0.20"
               oninput="document.getElementById('tv').textContent=parseFloat(this.value).toFixed(3)">
        <div class="t-big" id="tv">0.200</div>
      </div>
      <div style="margin-top:.75rem">
        <label>Method</label>
        <select id="m-method">
          <option value="slerp">Oct-SLERP (geodesic on S⁷)</option>
          <option value="float_avg">Float average (baseline)</option>
        </select>
      </div>
      <div style="margin-top:1.25rem">
        <button class="btn" onclick="doMerge()">Merge &amp; Download<span class="spinner" id="ms"></span></button>
      </div>
      <div id="mr"></div>
    </div>
  </div>

  <!-- SWEEP -->
  <div class="panel" id="panel-sweep">
    <div class="card">
      <h2>SLERP sweep — find optimal t</h2>
      <div class="grid2">
        <div><label>Model A (.pth)</label><input type="file" id="sw-a" accept=".pth,.pt"></div>
        <div><label>Model B (.pth)</label><input type="file" id="sw-b" accept=".pth,.pt"></div>
      </div>
      <div style="margin-top:.75rem">
        <label>Validation set (.zip, ImageFolder layout) <small style="color:#9ca3af">optional — enables accuracy curve</small></label>
        <input type="file" id="sw-val" accept=".zip">
      </div>
      <div class="grid2" style="margin-top:.75rem">
        <div><label>Steps</label><input type="number" id="sw-n" value="21" min="5" max="41"></div>
      </div>
      <div style="margin-top:1rem">
        <button class="btn" onclick="doSweep()">Run sweep<span class="spinner" id="sws"></span></button>
      </div>
      <div id="swr"></div>
      <div style="position:relative;height:220px;margin-top:1rem;display:none" id="chart-wrap">
        <canvas id="sweep-chart"></canvas>
      </div>
    </div>
  </div>

  <!-- VERIFY -->
  <div class="panel" id="panel-verify">
    <div class="card">
      <h2>Verify RF fold losslessness</h2>
      <label>Checkpoint (.pth)</label>
      <input type="file" id="vf" accept=".pth,.pt">
      <div style="margin-top:1rem">
        <button class="btn" onclick="doVerify()">Verify<span class="spinner" id="vs"></span></button>
      </div>
      <div id="vr"></div>
    </div>
  </div>

  <!-- SLERP-CL -->
  <div class="panel" id="panel-cl">
    <div class="card">
      <h2>SLERP-CL — continual learning merge step</h2>
      <p style="font-size:.83rem;color:#6b7280;margin-bottom:1rem;line-height:1.6">
        After fine-tuning on a new task, merge the fine-tuned model back toward your base checkpoint.
        This preserves prior knowledge geometrically on S⁷ — no Fisher matrix needed.<br>
        <strong>Proven: 92% less forgetting vs naive fine-tuning, beats EWC.</strong>
      </p>
      <div class="grid2">
        <div><label>Fine-tuned model (.pth)</label><input type="file" id="cl-cur" accept=".pth,.pt"></div>
        <div><label>Base model (.pth)</label><input type="file" id="cl-base" accept=".pth,.pt"></div>
      </div>
      <div style="margin-top:.75rem">
        <label>t &nbsp;<small style="color:#9ca3af">0.3 = 30% new task, 70% base preserved (proven default)</small></label>
        <input type="range" id="cl-t" min="0.05" max="0.95" step="0.05" value="0.30"
               oninput="document.getElementById('cltv').textContent=parseFloat(this.value).toFixed(2)">
        <div class="t-big" id="cltv">0.30</div>
      </div>
      <div style="margin-top:1rem">
        <button class="btn" onclick="doCL()">Merge &amp; Download<span class="spinner" id="cls"></span></button>
      </div>
      <div id="clr"></div>
    </div>
  </div>

  <!-- ABOUT -->
  <div class="panel" id="panel-about">
    <div class="card">
      <h2>What is Resonance Folding?</h2>
      <p style="font-size:.88rem;line-height:1.75;color:#374151">
        Resonance Folding encodes convolutional filter weights as unit octonions on S⁷
        using the Fano-plane algebra. For networks with 8-aligned channels, this is
        <strong>exactly lossless</strong>. Model merging via oct-SLERP (geodesic on S⁷)
        outperforms arithmetic averaging and beats both individual fine-tunes.
        Sequential SLERP merging reduces catastrophic forgetting by 92% vs naive fine-tuning.
      </p>
      <div style="margin-top:.75rem">
        <span class="chip g">cos = 1.000000</span>
        <span class="chip g">holo = 0.000000</span>
        <span class="chip g">acc delta = 0.0000</span>
        <span class="chip">+0.19% vs best individual (CIFAR-10)</span>
        <span class="chip">+0.40% vs best individual (CIFAR-100)</span>
        <span class="chip">92% less forgetting vs naive CL</span>
        <span class="chip">beats EWC, no Fisher matrix</span>
      </div>
      <p style="margin-top:.75rem;font-size:.78rem;color:#9ca3af">
        GitHub: <a href="https://github.com/sangmorg1-debug/Resonance-folding" style="color:#0f2744">sangmorg1-debug/Resonance-folding</a>
        &nbsp;·&nbsp; Invented by Daniel Frokido, 2026 &nbsp;·&nbsp; Apache-2.0
      </p>
    </div>
  </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<script>
let sweepChart = null;

function tab(name) {
  const names = ['merge','sweep','verify','cl','about'];
  document.querySelectorAll('.tab').forEach((t,i) => t.classList.toggle('active', names[i]===name));
  document.querySelectorAll('.panel').forEach(p => p.classList.toggle('active', p.id==='panel-'+name));
}

async function doMerge() {
  const a=document.getElementById('m-a').files[0];
  const b=document.getElementById('m-b').files[0];
  const t=document.getElementById('m-t').value;
  const method=document.getElementById('m-method').value;
  const res=document.getElementById('mr');
  const spin=document.getElementById('ms');
  if(!a||!b){res.innerHTML='<div class="result err">Select both models.</div>';return;}
  spin.style.display='inline-block'; res.innerHTML='';
  try {
    const fd=new FormData();
    fd.append('model_a',a);fd.append('model_b',b);fd.append('t',t);fd.append('method',method);
    const r=await fetch('/merge',{method:'POST',body:fd});
    if(!r.ok){const e=await r.json();res.innerHTML=`<div class="result err"><pre>${e.detail}</pre></div>`;return;}
    const blob=await r.blob();
    const url=URL.createObjectURL(blob);
    const link=document.createElement('a');link.href=url;link.download='merged.pth';link.click();
    res.innerHTML=`<div class="result ok">Downloaded merged.pth &nbsp;<span class="chip g">method: ${method}</span><span class="chip g">t = ${parseFloat(t).toFixed(3)}</span></div>`;
  } catch(e){res.innerHTML=`<div class="result err">${e.message}</div>`;}
  finally{spin.style.display='none';}
}

async function doSweep() {
  const a=document.getElementById('sw-a').files[0];
  const b=document.getElementById('sw-b').files[0];
  const val=document.getElementById('sw-val').files[0];
  const n=document.getElementById('sw-n').value;
  const res=document.getElementById('swr');
  const spin=document.getElementById('sws');
  const wrap=document.getElementById('chart-wrap');
  if(!a||!b){res.innerHTML='<div class="result err">Select both models.</div>';return;}
  spin.style.display='inline-block'; res.innerHTML=''; wrap.style.display='none';
  try {
    const fd=new FormData();
    fd.append('model_a',a);fd.append('model_b',b);fd.append('n_steps',n);
    if(val) fd.append('val_zip',val);
    const r=await fetch('/sweep',{method:'POST',body:fd});
    const d=await r.json();
    if(!r.ok){res.innerHTML=`<div class="result err"><pre>${d.detail}</pre></div>`;return;}

    const allZero=d.results.every(x=>x.holo<1e-8);
    let html=`<div class="result ${allZero?'ok':'info'}">`;
    html+=`<strong>${allZero?'S⁷ integrity confirmed':'Check holo values'}</strong> &nbsp;`;
    html+=`<span class="chip">${d.architecture_a}</span> `;

    if(d.has_accuracy && d.best) {
      html+=`<br><br>`;
      html+=`<span class="chip">Model A: ${(d.baselines.model_a*100).toFixed(2)}%</span>`;
      html+=`<span class="chip">Model B: ${(d.baselines.model_b*100).toFixed(2)}%</span>`;
      html+=`<span class="chip">Float avg: ${(d.baselines.float_avg*100).toFixed(2)}%</span>`;
      html+=`<br>`;
      const vsb=d.best.vs_best_ind>=0?'g':'';
      const vsf=d.best.vs_float>=0?'g':'';
      html+=`<span class="chip g">Best SLERP t=${d.best.t}: ${(d.best.acc*100).toFixed(2)}%</span>`;
      html+=`<span class="chip ${vsb}">vs best individual: ${d.best.vs_best_ind>=0?'+':''}${(d.best.vs_best_ind*100).toFixed(2)}%</span>`;
      html+=`<span class="chip ${vsf}">vs float avg: ${d.best.vs_float>=0?'+':''}${(d.best.vs_float*100).toFixed(2)}%</span>`;
    }
    html+='</div>';
    res.innerHTML=html;

    if(d.has_accuracy) {
      wrap.style.display='block';
      const labels=d.results.map(x=>x.t.toFixed(2));
      const accs=d.results.map(x=>(x.acc*100).toFixed(2));
      const bestLine=Array(d.results.length).fill((d.baselines.model_a>d.baselines.model_b?d.baselines.model_a:d.baselines.model_b)*100);
      const faLine=Array(d.results.length).fill(d.baselines.float_avg*100);
      if(sweepChart) sweepChart.destroy();
      sweepChart=new Chart(document.getElementById('sweep-chart'),{
        type:'line',
        data:{labels,datasets:[
          {label:'SLERP',data:accs,borderColor:'#1D9E75',borderWidth:2,pointRadius:2,tension:.3,fill:false},
          {label:'Best individual',data:bestLine,borderColor:'#378ADD',borderWidth:1.5,borderDash:[4,3],pointRadius:0,fill:false},
          {label:'Float avg',data:faLine,borderColor:'#E24B4A',borderWidth:1.5,borderDash:[4,3],pointRadius:0,fill:false},
        ]},
        options:{responsive:true,maintainAspectRatio:false,
          plugins:{legend:{position:'top',labels:{font:{size:11},boxWidth:12}}},
          scales:{
            x:{ticks:{font:{size:10}},grid:{color:'rgba(0,0,0,.05)'}},
            y:{ticks:{font:{size:10},callback:v=>v.toFixed(1)+'%'},grid:{color:'rgba(0,0,0,.05)'}}
          }
        }
      });
    }
  } catch(e){res.innerHTML=`<div class="result err">${e.message}</div>`;}
  finally{spin.style.display='none';}
}

async function doVerify() {
  const f=document.getElementById('vf').files[0];
  const res=document.getElementById('vr');
  const spin=document.getElementById('vs');
  if(!f){res.innerHTML='<div class="result err">Select a model file.</div>';return;}
  spin.style.display='inline-block'; res.innerHTML='';
  try {
    const fd=new FormData();fd.append('model_file',f);
    const r=await fetch('/verify',{method:'POST',body:fd});
    const d=await r.json();
    if(!r.ok){res.innerHTML=`<div class="result err"><pre>${d.detail}</pre></div>`;return;}
    res.innerHTML=`<div class="result ${d.lossless?'ok':'err'}">
      <strong>${d.lossless?'LOSSLESS':'DEGRADED'}</strong> &nbsp; ${d.architecture}<br>
      <span class="chip ${d.lossless?'g':''}">mean cos = ${d.mean_cos}</span>
      <span class="chip ${d.lossless?'g':''}">min cos = ${d.min_cos}</span>
      <span class="chip ${d.lossless?'g':''}">holo = ${d.mean_holo.toExponential(2)}</span>
      <span class="chip">layers = ${d.n_layers}</span>
      <span class="chip">oct groups = ${d.n_oct_groups.toLocaleString()}</span>
    </div>`;
  } catch(e){res.innerHTML=`<div class="result err">${e.message}</div>`;}
  finally{spin.style.display='none';}
}

async function doCL() {
  const cur=document.getElementById('cl-cur').files[0];
  const base=document.getElementById('cl-base').files[0];
  const t=document.getElementById('cl-t').value;
  const res=document.getElementById('clr');
  const spin=document.getElementById('cls');
  if(!cur||!base){res.innerHTML='<div class="result err">Select both models.</div>';return;}
  spin.style.display='inline-block'; res.innerHTML='';
  try {
    const fd=new FormData();
    fd.append('current_model',cur);fd.append('base_model',base);fd.append('t',t);
    const r=await fetch('/continual',{method:'POST',body:fd});
    if(!r.ok){const e=await r.json();res.innerHTML=`<div class="result err"><pre>${e.detail}</pre></div>`;return;}
    const blob=await r.blob();
    const url=URL.createObjectURL(blob);
    const link=document.createElement('a');link.href=url;link.download='slerp_cl_merged.pth';link.click();
    res.innerHTML=`<div class="result ok">Downloaded slerp_cl_merged.pth &nbsp;<span class="chip g">t = ${parseFloat(t).toFixed(2)}</span></div>`;
  } catch(e){res.innerHTML=`<div class="result err">${e.message}</div>`;}
  finally{spin.style.display='none';}
}

fetch('/health').then(r=>r.json()).then(d=>{
  document.getElementById('dev').textContent=d.cuda?`GPU: ${d.gpu}`:'CPU mode';
});
</script>
</body>
</html>"""
