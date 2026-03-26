import gradio as gr
import sys, os
sys.path.insert(0, ".")
print("🚀 LOSSLESS RESONANCE FOLDING MODELMERGE - HF SPACE FIXED")
print("   Lossless octonion folding → unit octonions on S⁷ + oct-SLERP (cos=1.000000, holo≈9e-16)")
print("   Support the project → https://patreon.com/Frokido")

# PERMANENT FIX: execute your exact webui_merger.py directly
with open("webui_merger.py", "r", encoding="utf-8") as f:
    exec(f.read(), globals())

ui = ResonanceMergerUI()
demo = ui.build_ui()

demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=False,
    show_error=True,
    quiet=True
)
