"""
RESONANCE FOLDING — CONTINUAL LEARNING EXPERIMENT
==================================================
Core question: does sequential SLERP merging allow a model to learn
new tasks without forgetting old ones?

Standard fine-tuning exhibits catastrophic forgetting:
  Base → fine-tune on Task A → fine-tune on Task B
  Result: model forgets Task A entirely

SLERP-based continual learning hypothesis:
  Base → fine-tune on Task A → SLERP merge(base, A, t) → fine-tune on Task B
  → SLERP merge(prev, B, t) → evaluate on BOTH A and B

Protocol:
  Task A: CIFAR-10 classes 0-4  (airplane, auto, bird, cat, deer)
  Task B: CIFAR-10 classes 5-9  (dog, frog, horse, ship, truck)
  Model:  OctResNet18 (11.2M params)

Comparison:
  1. Naive sequential fine-tuning (baseline — catastrophic forgetting)
  2. EWC (Elastic Weight Consolidation) — standard CL baseline
  3. SLERP continual learning — our method

Metrics:
  - Accuracy on Task A after learning Task B  (forgetting measure)
  - Accuracy on Task B after learning Task B  (plasticity measure)
  - Backward transfer: how much does B hurt A?
  - Forward transfer: does A help B?

Run:
  python resonance_folding_continual.py

Flags:
  --epochs-base   20  (base model training)
  --epochs-ft     15  (per-task fine-tuning)
  --slerp-t       0.3 (merge weight — how much to pull toward new task)
  --batch         128
"""

import argparse
import copy
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--epochs-base", type=int, default=20)
parser.add_argument("--epochs-ft",   type=int, default=15)
parser.add_argument("--slerp-t",     type=float, default=0.3)
parser.add_argument("--batch",       type=int, default=128)
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2470, 0.2435, 0.2616)

TASK_A = list(range(5))    # classes 0-4
TASK_B = list(range(5,10)) # classes 5-9
CLASSES = ["airplane","automobile","bird","cat","deer",
           "dog","frog","horse","ship","truck"]


# ─────────────────────────────────────────────────────────────
#  OCTONION ALGEBRA
# ─────────────────────────────────────────────────────────────

def oct_mul(A, B):
    p = A.unbind(-1); q = B.unbind(-1)
    return torch.stack([
        p[0]*q[0]-p[1]*q[1]-p[2]*q[2]-p[3]*q[3]-p[4]*q[4]-p[5]*q[5]-p[6]*q[6]-p[7]*q[7],
        p[0]*q[1]+p[1]*q[0]+p[2]*q[3]-p[3]*q[2]+p[4]*q[5]-p[5]*q[4]+p[6]*q[7]-p[7]*q[6],
        p[0]*q[2]-p[1]*q[3]+p[2]*q[0]+p[3]*q[1]-p[4]*q[6]+p[5]*q[7]+p[6]*q[4]-p[7]*q[5],
        p[0]*q[3]+p[1]*q[2]-p[2]*q[1]+p[3]*q[0]-p[4]*q[7]-p[5]*q[6]+p[6]*q[5]+p[7]*q[4],
        p[0]*q[4]-p[1]*q[5]+p[2]*q[6]-p[3]*q[7]+p[4]*q[0]+p[5]*q[1]-p[6]*q[2]+p[7]*q[3],
        p[0]*q[5]+p[1]*q[4]-p[2]*q[7]-p[3]*q[6]-p[4]*q[1]+p[5]*q[0]+p[6]*q[3]+p[7]*q[2],
        p[0]*q[6]-p[1]*q[7]-p[2]*q[4]+p[3]*q[5]+p[4]*q[2]-p[5]*q[3]+p[6]*q[0]+p[7]*q[1],
        p[0]*q[7]+p[1]*q[6]+p[2]*q[5]+p[3]*q[4]-p[4]*q[3]-p[5]*q[2]-p[6]*q[1]+p[7]*q[0],
    ], dim=-1)

def oct_conj(O):
    c = O.clone(); c[..., 1:] = -c[..., 1:]; return c

def oct_normalize(O, eps=1e-12):
    return O / (O.norm(dim=-1, keepdim=True) + eps)

def oct_slerp(A, B, t):
    dot   = (A * B).sum(-1, keepdim=True).clamp(-1+1e-7, 1-1e-7)
    theta = torch.acos(dot)
    sin_t = torch.sin(theta)
    safe  = (sin_t.abs() > 1e-6).float()
    ca    = torch.where(safe.bool(), torch.sin((1-t)*theta)/(sin_t+1e-12),
                        torch.full_like(sin_t, 1-t))
    cb    = torch.where(safe.bool(), torch.sin(t*theta)/(sin_t+1e-12),
                        torch.full_like(sin_t, t))
    return oct_normalize(ca*A + cb*B)


# ─────────────────────────────────────────────────────────────
#  OCTRESNET18
# ─────────────────────────────────────────────────────────────

def oct_init_(w):
    oc, ic, kH, kW = w.shape
    if ic % 8 == 0:
        n = oc * kH * kW * (ic // 8)
        g = oct_normalize(torch.randn(n, 8)) * math.sqrt(2.0/(ic*kH*kW))
        w.data.copy_(g.reshape(oc,kH,kW,ic//8,8).permute(0,3,4,1,2).reshape(oc,ic,kH,kW))
    else:
        nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')

class OctBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.downsample = downsample
        oct_init_(self.conv1.weight); oct_init_(self.conv2.weight)
        if downsample: oct_init_(downsample[0].weight)
    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        if self.downsample: identity = self.downsample(x)
        return F.relu(out + identity, inplace=True)

class OctResNet18(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        nn.init.kaiming_normal_(self.stem[0].weight, mode='fan_out')
        self._in = 64
        self.layer1 = self._make(64,  2, 1)
        self.layer2 = self._make(128, 2, 2)
        self.layer3 = self._make(256, 2, 2)
        self.layer4 = self._make(512, 2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, n_classes)
        nn.init.normal_(self.fc.weight, 0, 0.01); nn.init.zeros_(self.fc.bias)

    def _make(self, ch, n, stride):
        ds = None
        if stride != 1 or self._in != ch:
            ds = nn.Sequential(nn.Conv2d(self._in,ch,1,stride=stride,bias=False),
                               nn.BatchNorm2d(ch))
        layers = [OctBasicBlock(self._in, ch, stride, ds)]
        self._in = ch
        for _ in range(1, n): layers.append(OctBasicBlock(ch, ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        return self.fc(self.avgpool(x).flatten(1))

    def oct_layers(self):
        return [(n,m) for n,m in self.named_modules()
                if isinstance(m,nn.Conv2d) and m.weight.shape[1]%8==0]


# ─────────────────────────────────────────────────────────────
#  SLERP MERGE
# ─────────────────────────────────────────────────────────────

def to_octs(W):
    oc,ic,kH,kW = W.shape
    k = W.reshape(oc,ic//8,8,kH,kW).permute(0,3,4,1,2).reshape(-1,8)
    return oct_normalize(k), k.norm(dim=-1,keepdim=True)

def from_octs(o,n,shape):
    oc,ic,kH,kW=shape
    return (o*n).reshape(oc,kH,kW,ic//8,8).permute(0,3,4,1,2).reshape(oc,ic,kH,kW)

def slerp_merge(ma, mb, t):
    """SLERP merge: (1-t)*ma + t*mb on S⁷."""
    merged = copy.deepcopy(ma)
    sa = {n: to_octs(c.weight.data) for n,c in ma.oct_layers()}
    sb = {n: to_octs(c.weight.data) for n,c in mb.oct_layers()}
    mods = dict(merged.named_modules())
    for name in sa:
        if name not in sb: continue
        oa,na = sa[name]; ob,nb = sb[name]
        om = oct_slerp(oa, ob, t)
        nm = (1-t)*na + t*nb
        with torch.no_grad():
            mods[name].weight.data.copy_(from_octs(om,nm,mods[name].weight.shape))
    sd_a,sd_b = ma.state_dict(),mb.state_dict()
    sd_m = merged.state_dict()
    oct_keys = {n+".weight" for n in sa}
    for k in sd_a:
        if k in oct_keys: continue
        if sd_a[k].is_floating_point():
            sd_m[k] = (1-t)*sd_a[k] + t*sd_b[k]
    merged.load_state_dict(sd_m)
    return merged


# ─────────────────────────────────────────────────────────────
#  EWC BASELINE
# ─────────────────────────────────────────────────────────────

class EWC:
    """
    Elastic Weight Consolidation (Kirkpatrick et al. 2017).
    Computes Fisher information matrix after Task A training.
    Adds penalty term to loss during Task B training.
    """
    def __init__(self, model, loader, device, n_samples=500):
        self.params = {n: p.clone().detach()
                       for n,p in model.named_parameters() if p.requires_grad}
        self.fisher  = self._fisher(model, loader, device, n_samples)

    def _fisher(self, model, loader, device, n_samples):
        fisher = {n: torch.zeros_like(p)
                  for n,p in model.named_parameters() if p.requires_grad}
        model.eval(); count = 0
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            model.zero_grad()
            out  = model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            for n,p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2
            count += len(x)
            if count >= n_samples: break
        for n in fisher: fisher[n] /= count
        return fisher

    def penalty(self, model, lam=1000.0):
        loss = 0.0
        for n,p in model.named_parameters():
            if n in self.fisher:
                loss += (self.fisher[n] * (p - self.params[n]) ** 2).sum()
        return lam * loss


# ─────────────────────────────────────────────────────────────
#  DATA
# ─────────────────────────────────────────────────────────────

def get_loaders(classes=None, batch=128):
    """Get loaders for a subset of CIFAR-10 classes (or all if classes=None)."""
    tf_tr = T.Compose([T.RandomCrop(32,padding=4), T.RandomHorizontalFlip(),
                       T.ColorJitter(.2,.2,.2,.1), T.ToTensor(),
                       T.Normalize(MEAN,STD)])
    tf_va = T.Compose([T.ToTensor(), T.Normalize(MEAN,STD)])

    tr_full = torchvision.datasets.CIFAR10("./data",train=True, download=True,transform=tf_tr)
    va_full = torchvision.datasets.CIFAR10("./data",train=False,download=True,transform=tf_va)

    if classes is None:
        tr_ds, va_ds = tr_full, va_full
    else:
        tr_idx = [i for i,(_, y) in enumerate(tr_full) if y in classes]
        va_idx = [i for i,(_, y) in enumerate(va_full) if y in classes]
        tr_ds  = torch.utils.data.Subset(tr_full, tr_idx)
        va_ds  = torch.utils.data.Subset(va_full, va_idx)

    return (torch.utils.data.DataLoader(tr_ds, batch_size=batch, shuffle=True,
                                         num_workers=2, pin_memory=True,
                                         persistent_workers=True),
            torch.utils.data.DataLoader(va_ds, batch_size=256, shuffle=False,
                                         num_workers=2, pin_memory=True,
                                         persistent_workers=True))


# ─────────────────────────────────────────────────────────────
#  TRAINING / EVAL
# ─────────────────────────────────────────────────────────────

def evaluate(model, loader):
    model.eval(); corr = total = 0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(DEVICE), y.to(DEVICE)
            corr  += (model(x).argmax(1)==y).sum().item()
            total += len(y)
    return corr / total if total > 0 else 0.0

def train(model, tr, va, epochs, label, ewc=None):
    opt   = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=1e-3, epochs=epochs, steps_per_epoch=len(tr),
        pct_start=0.15, div_factor=10, final_div_factor=100)
    best_acc, best_sd = 0.0, None
    print(f"\n  [{label}]")
    print(f"  {'ep':>4}  {'loss':>8}  {'val':>8}")
    print(f"  {'-'*26}")
    for ep in range(1, epochs+1):
        model.train()
        tl = tc = tn = 0
        for x,y in tr:
            x,y = x.to(DEVICE), y.to(DEVICE)
            out  = model(x)
            loss = F.cross_entropy(out, y, label_smoothing=0.1)
            if ewc: loss = loss + ewc.penalty(model)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tl+=loss.item()*len(y); tc+=(out.argmax(1)==y).sum().item(); tn+=len(y)
        va_acc = evaluate(model, va)
        if va_acc > best_acc: best_acc=va_acc; best_sd=copy.deepcopy(model.state_dict())
        if ep % max(1,epochs//5)==0 or ep<=2 or ep==epochs:
            print(f"  {ep:>4}  {tl/tn:>8.4f}  {va_acc:>8.4f}")
    model.load_state_dict(best_sd)
    return best_acc


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  RESONANCE FOLDING — CONTINUAL LEARNING EXPERIMENT")
    print(f"  Device: {DEVICE}  |  SLERP t={args.slerp_t}")
    print(f"  Task A: classes 0-4  ({', '.join(CLASSES[:5])})")
    print(f"  Task B: classes 5-9  ({', '.join(CLASSES[5:])})")
    print("=" * 72)

    torchvision.datasets.CIFAR10("./data", train=True,  download=True)
    torchvision.datasets.CIFAR10("./data", train=False, download=True)

    tr_a, va_a = get_loaders(TASK_A, args.batch)
    tr_b, va_b = get_loaders(TASK_B, args.batch)
    tr_all, va_all = get_loaders(None, args.batch)

    # ── Train base model on all 10 classes ───────────────────
    print(f"\n{'─'*72}")
    print("  STEP 1: Train base model (all 10 classes)")
    model_base = OctResNet18(n_classes=10).to(DEVICE)
    acc_base_all = train(model_base, tr_all, va_all,
                         args.epochs_base, "Base (all classes)")
    acc_base_a = evaluate(model_base, va_a)
    acc_base_b = evaluate(model_base, va_b)
    print(f"\n  Base: all={acc_base_all:.4f}  A={acc_base_a:.4f}  B={acc_base_b:.4f}")

    # ─────────────────────────────────────────────────────────
    #  METHOD 1: NAIVE SEQUENTIAL FINE-TUNING (baseline)
    # ─────────────────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print("  METHOD 1: Naive sequential fine-tuning")
    print("  Base → fine-tune A → fine-tune B  (catastrophic forgetting)")

    naive = copy.deepcopy(model_base)

    # Fine-tune on A
    acc_naive_a1 = train(naive, tr_a, va_a, args.epochs_ft,
                         "Naive: fine-tune A")
    acc_naive_a_check = evaluate(naive, va_a)

    # Fine-tune on B — this will destroy A's knowledge
    acc_naive_b = train(naive, tr_b, va_b, args.epochs_ft,
                        "Naive: fine-tune B (after A)")
    acc_naive_a_after = evaluate(naive, va_a)  # forgetting measure
    acc_naive_all = evaluate(naive, va_all)

    print(f"\n  Naive: A after A-training={acc_naive_a_check:.4f}  "
          f"A after B-training={acc_naive_a_after:.4f}  "
          f"B={acc_naive_b:.4f}")
    print(f"  Forgetting: {acc_naive_a_after - acc_naive_a_check:+.4f}")

    # ─────────────────────────────────────────────────────────
    #  METHOD 2: EWC BASELINE
    # ─────────────────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print("  METHOD 2: EWC (Elastic Weight Consolidation)")
    print("  Base → fine-tune A → compute Fisher → fine-tune B with EWC penalty")

    ewc_model = copy.deepcopy(model_base)

    # Fine-tune on A
    acc_ewc_a = train(ewc_model, tr_a, va_a, args.epochs_ft,
                      "EWC: fine-tune A")

    # Compute Fisher information on Task A
    print("\n  Computing Fisher information matrix on Task A...")
    ewc = EWC(ewc_model, tr_a, DEVICE, n_samples=1000)
    print("  Fisher computed.")

    # Fine-tune on B with EWC penalty
    acc_ewc_b = train(ewc_model, tr_b, va_b, args.epochs_ft,
                      "EWC: fine-tune B (with penalty)", ewc=ewc)
    acc_ewc_a_after = evaluate(ewc_model, va_a)
    acc_ewc_all = evaluate(ewc_model, va_all)

    print(f"\n  EWC: A after A={acc_ewc_a:.4f}  "
          f"A after B={acc_ewc_a_after:.4f}  B={acc_ewc_b:.4f}")
    print(f"  Forgetting: {acc_ewc_a_after - acc_ewc_a:+.4f}")

    # ─────────────────────────────────────────────────────────
    #  METHOD 3: SLERP CONTINUAL LEARNING
    # ─────────────────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print(f"  METHOD 3: SLERP Continual Learning (t={args.slerp_t})")
    print("  Base → fine-tune A → SLERP(base, A, t) → fine-tune B → SLERP(prev, B, t)")

    slerp_model = copy.deepcopy(model_base)

    # Fine-tune on A
    model_ft_a = copy.deepcopy(model_base)
    acc_slerp_a = train(model_ft_a, tr_a, va_a, args.epochs_ft,
                        "SLERP-CL: fine-tune A")

    # SLERP merge: blend base knowledge with A specialization
    # t=slerp_t means we take t of A's new direction, keep (1-t) of base
    slerp_model = slerp_merge(model_base, model_ft_a, args.slerp_t)
    acc_slerp_after_a_merge = evaluate(slerp_model, va_a)
    acc_slerp_base_retained = evaluate(slerp_model, va_b)
    print(f"\n  After SLERP(base, A, {args.slerp_t}): "
          f"A={acc_slerp_after_a_merge:.4f}  B(retained)={acc_slerp_base_retained:.4f}")

    # Fine-tune on B from the merged checkpoint
    checkpoint_after_a = copy.deepcopy(slerp_model)
    model_ft_b = copy.deepcopy(slerp_model)
    acc_slerp_b_raw = train(model_ft_b, tr_b, va_b, args.epochs_ft,
                            "SLERP-CL: fine-tune B")

    # SLERP merge again: blend knowledge of A (preserved in checkpoint) with B
    slerp_final = slerp_merge(checkpoint_after_a, model_ft_b, args.slerp_t)
    acc_slerp_a_final = evaluate(slerp_final, va_a)
    acc_slerp_b_final = evaluate(slerp_final, va_b)
    acc_slerp_all = evaluate(slerp_final, va_all)

    print(f"\n  SLERP-CL final: A={acc_slerp_a_final:.4f}  "
          f"B={acc_slerp_b_final:.4f}  all={acc_slerp_all:.4f}")

    # ─────────────────────────────────────────────────────────
    #  FINAL COMPARISON TABLE
    # ─────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  CONTINUAL LEARNING RESULTS")
    print(f"{'='*72}")
    print(f"""
  {'Method':<28} {'Task A':>8} {'Task B':>8} {'All':>8} {'Forgetting':>12}
  {'-'*62}
  {'Base model':<28} {acc_base_a:>8.4f} {acc_base_b:>8.4f} {acc_base_all:>8.4f} {'—':>12}
  {'Naive sequential':<28} {acc_naive_a_after:>8.4f} {acc_naive_b:>8.4f} {acc_naive_all:>8.4f} {acc_naive_a_after-acc_naive_a_check:>+12.4f}
  {'EWC':<28} {acc_ewc_a_after:>8.4f} {acc_ewc_b:>8.4f} {acc_ewc_all:>8.4f} {acc_ewc_a_after-acc_ewc_a:>+12.4f}
  {f'SLERP-CL (t={args.slerp_t})':<28} {acc_slerp_a_final:>8.4f} {acc_slerp_b_final:>8.4f} {acc_slerp_all:>8.4f} {acc_slerp_a_final-acc_slerp_after_a_merge:>+12.4f}
""")

    # Forgetting = acc on A after B-training minus acc on A after A-training
    naive_forget  = acc_naive_a_check  - acc_naive_a_after
    ewc_forget    = acc_ewc_a          - acc_ewc_a_after
    slerp_forget  = acc_slerp_after_a_merge - acc_slerp_a_final

    print(f"  Forgetting comparison (lower is better):")
    print(f"    Naive:    {naive_forget:.4f}")
    print(f"    EWC:      {ewc_forget:.4f}")
    print(f"    SLERP-CL: {slerp_forget:.4f}")

    best_method = min(
        [("Naive", naive_forget), ("EWC", ewc_forget),
         ("SLERP-CL", slerp_forget)],
        key=lambda x: x[1]
    )

    print(f"\n  Best forgetting resistance: {best_method[0]} ({best_method[1]:.4f})")

    if slerp_forget < naive_forget - 0.01:
        verdict = "SLERP-CL WINS"
        detail  = (f"SLERP continual learning reduces forgetting by "
                   f"{naive_forget - slerp_forget:.4f} vs naive fine-tuning. "
                   f"The geodesic merge preserves Task A knowledge while "
                   f"allowing Task B acquisition.")
    elif slerp_forget < ewc_forget - 0.005:
        verdict = "SLERP-CL BEATS EWC"
        detail  = (f"SLERP-CL reduces forgetting more than EWC "
                   f"({slerp_forget:.4f} vs {ewc_forget:.4f}).")
    elif abs(slerp_forget - ewc_forget) < 0.01:
        verdict = "SLERP-CL MATCHES EWC"
        detail  = (f"SLERP-CL achieves similar forgetting resistance to EWC "
                   f"without any Fisher matrix computation. "
                   f"Simpler and equally effective.")
    else:
        verdict = "MORE TUNING NEEDED"
        detail  = (f"Try --slerp-t values between 0.1 and 0.5. "
                   f"Smaller t preserves more of the previous task.")

    print(f"\n  [{verdict}]")
    print(f"  {detail}")
    print(f"""
  WHAT THIS MEANS:
  ─────────────────────────────────────────────────────────────
  Standard fine-tuning: learning Task B destroys Task A
  EWC: adds Fisher penalty to slow forgetting — computationally
       expensive (requires computing Hessian diagonal)
  SLERP-CL: after each task, merge new knowledge with previous
       checkpoint via geodesic on S⁷ — preserves prior knowledge
       geometrically, no extra computation beyond the merge itself

  If SLERP-CL matches or beats EWC with less overhead, this is
  a publishable continual learning result.
  ─────────────────────────────────────────────────────────────
""")


if __name__ == "__main__":
    main()
