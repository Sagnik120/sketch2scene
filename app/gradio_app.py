"""
SketchToScene — Gradio Demo for HuggingFace Spaces
"""

from __future__ import annotations
import os
import torch
import numpy as np
import gradio as gr
from PIL import Image

# ── Load model ────────────────────────────────────────────────────────────────

MODEL = None
DEVICE = "cpu"


def load_model():
    global MODEL, DEVICE
    model_path = os.getenv("MODEL_PATH", "checkpoints/best")
    hf_repo    = os.getenv("HF_MODEL_REPO", "Sagnik120/sketch2scene")

    if torch.backends.mps.is_available():
        DEVICE = "mps"
    elif torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"

    try:
        from src.model.sketch2scene import SketchToScene
        from src.model.transformer import TransformerConfig
        if os.path.exists(model_path):
            MODEL = SketchToScene.load(model_path, TransformerConfig(), device=DEVICE)
        else:
            # Try loading from HuggingFace Hub
            from huggingface_hub import snapshot_download
            local = snapshot_download(repo_id=hf_repo)
            MODEL = SketchToScene.load(local, TransformerConfig(), device=DEVICE)
        print(f"✓ Model loaded on {DEVICE}")
    except Exception as e:
        print(f"⚠ Could not load model: {e}")
        MODEL = None


load_model()


# ── Inference function ────────────────────────────────────────────────────────

def predict(
    sketch_img: np.ndarray,
    temperature: float,
    top_k: int,
    complete_scene: bool,
):
    if MODEL is None:
        return "⚠ Model not loaded. Please check setup.", None

    pil = Image.fromarray(sketch_img).convert("RGB")
    import torchvision.transforms as T
    tf = T.Compose([
        T.Resize((256, 256)),
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        T.Normalize([0.5]*3, [0.5]*3),
    ])
    sketch_tensor = tf(pil).unsqueeze(0).to(DEVICE)

    description = MODEL.generate_description(
        sketch_tensor,
        temperature=temperature,
        top_k=top_k,
    )

    completed_pil = None
    if complete_scene:
        completed = MODEL.complete_image(sketch_tensor, description)
        completed = (completed.squeeze(0) * 0.5 + 0.5).clamp(0, 1)
        completed_pil = Image.fromarray(
            (completed.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        )

    return description, completed_pil


# ── Gradio UI ─────────────────────────────────────────────────────────────────

CSS = """
.gradio-container { font-family: 'Inter', sans-serif; }
#title { text-align: center; margin-bottom: 10px; }
#subtitle { text-align: center; color: #64748b; margin-bottom: 20px; }
"""

with gr.Blocks(css=CSS, theme=gr.themes.Soft(), title="SketchToScene") as demo:
    gr.HTML("<h1 id='title'>🖼️ SketchToScene</h1>")
    gr.HTML(
        "<p id='subtitle'>Draw a sketch → get a scene description and completed realistic image<br>"
        "Powered by a 125M decoder-only multimodal transformer trained from scratch</p>"
    )

    with gr.Row():
        with gr.Column(scale=1):
            sketch_input = gr.Image(
                label="Draw or Upload a Sketch",
                type="numpy",
                tool="sketch",
                image_mode="RGB",
            )
            with gr.Accordion("Generation Settings", open=False):
                temperature = gr.Slider(0.3, 1.5, value=0.7, step=0.05, label="Temperature")
                top_k       = gr.Slider(5, 200, value=50, step=5, label="Top-K")
                complete    = gr.Checkbox(value=True, label="Generate completed scene image")

            generate_btn = gr.Button("✨ Generate Scene", variant="primary")

        with gr.Column(scale=1):
            description_out = gr.Textbox(
                label="Generated Scene Description",
                placeholder="Scene description will appear here...",
                lines=6,
            )
            completed_out = gr.Image(label="Completed Scene Image", type="pil")

    gr.Examples(
        examples=[
            ["examples/mountain_sketch.png", 0.7, 50, True],
            ["examples/city_sketch.png", 0.8, 64, True],
        ],
        inputs=[sketch_input, temperature, top_k, complete],
        label="Example Sketches",
    )

    gr.HTML("""
    <div style='text-align:center; margin-top:20px; color:#94a3b8; font-size:0.85em'>
        Built by <a href='https://github.com/Sagnik120' target='_blank'>Sagnik120</a> |
        <a href='https://huggingface.co/Sagnik120/sketch2scene' target='_blank'>Model on HuggingFace</a> |
        <a href='https://github.com/Sagnik120/sketch2scene' target='_blank'>GitHub</a>
    </div>
    """)

    generate_btn.click(
        fn=predict,
        inputs=[sketch_input, temperature, top_k, complete],
        outputs=[description_out, completed_out],
    )

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
