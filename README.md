# 🖼️ SketchToScene 

> **A Multimodal Decoder-Only Transformer that converts hand-drawn sketches into realistic scene descriptions and completes missing image patches — autoregressively.**

[![HuggingFace](https://img.shields.io/badge/🤗-Model%20on%20HuggingFace-yellow)](https://huggingface.co/Sagnik120/sketch2scene)
[![HuggingFace Spaces](https://img.shields.io/badge/🤗-Demo%20on%20Spaces-blue)](https://huggingface.co/spaces/Sagnik120/sketch2scene)
[![GitHub](https://img.shields.io/badge/GitHub-Sagnik120-black)](https://github.com/Sagnik120/sketch2scene)

--- 

## 🧠 What Makes This Novel

Unlike diffusion models (which denoise) or encoder-decoder models (which have separate pathways), **SketchToScene uses a single decoder-only transformer** that:

1. **Tokenizes** sketch images into patch tokens using a VQ-VAE codebook
2. **Autoregressively generates** a natural language scene description token-by-token
3. **Continues generating** missing/realistic image patch tokens to complete the scene
4. All in **one unified forward pass** — no separate encoder, no cross-attention bottleneck

This is conceptually similar to how GPT handles text, but extended to a **mixed token vocabulary** of image codes + text BPE tokens.

---

## 🗂️ Project Structure

```
sketch2scene/
├── src/
│   ├── model/
│   │   ├── vqvae.py          # VQ-VAE for image tokenization
│   │   ├── transformer.py    # Decoder-only transformer (125M)
│   │   ├── tokenizer.py      # Unified image+text tokenizer
│   │   └── sketch2scene.py   # Full model wrapper
│   ├── data/
│   │   ├── dataset.py        # SketchyCOCO + Sketchy dataset loader
│   │   ├── augmentation.py   # Sketch augmentation pipeline
│   │   └── preprocess.py     # Preprocessing scripts
│   ├── training/
│   │   ├── trainer.py        # Training loop (MPS + CUDA compatible)
│   │   ├── losses.py         # Combined NLL + patch reconstruction loss
│   │   └── scheduler.py      # Cosine LR + warmup
│   └── inference/
│       ├── generate.py       # Autoregressive generation
│       └── visualize.py      # Output visualization
├── api/
│   ├── main.py               # FastAPI app
│   ├── schemas.py            # Pydantic models
│   └── utils.py              # API utilities
├── app/
│   └── gradio_app.py         # HuggingFace Spaces Gradio UI
├── scripts/
│   ├── train.py              # Training entry point
│   ├── evaluate.py           # Evaluation script
│   ├── export_hf.py          # Push model to HuggingFace Hub
│   └── download_data.py      # Dataset download helper
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_vqvae_training.ipynb
│   ├── 03_transformer_training.ipynb  # Kaggle notebook
│   └── 04_inference_demo.ipynb
├── configs/
│   ├── model_config.yaml
│   ├── train_config.yaml
│   └── inference_config.yaml
├── tests/
│   ├── test_model.py
│   ├── test_api.py
│   └── test_data.py
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── .env.example
├── .gitignore
├── requirements.txt
├── requirements-dev.txt
├── setup.py
└── Makefile
```

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/Sagnik120/sketch2scene.git
cd sketch2scene

# Create virtual environment
python -m venv venv
source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -e ".[dev]"

# Copy env file
cp .env.example .env
# Edit .env with your HuggingFace token
```

### 2. Download Data

```bash
python scripts/download_data.py --dataset sketchy_coco --output data/raw/
```

### 3. Train (M5 Mac — dev run)

```bash
python scripts/train.py --config configs/train_config.yaml --device mps --batch_size 4
```

### 4. Train (Kaggle T4 — full run)

Open `notebooks/03_transformer_training.ipynb` on Kaggle with GPU enabled.

### 5. Run API locally

```bash
uvicorn api.main:app --reload --port 8000
```

### 6. Run Gradio demo locally

```bash
python app/gradio_app.py
```

---

## 📦 Docker

```bash
docker-compose up --build
```

API available at `http://localhost:8000`, Gradio at `http://localhost:7860`.

---

## 🤗 HuggingFace

```bash
# Push trained model
python scripts/export_hf.py --repo Sagnik120/sketch2scene
```

---

## 📊 Model Architecture

| Component | Details |
|-----------|---------|
| VQ-VAE codebook | 8192 codes, 256-dim embeddings |
| Patch size | 16×16 pixels |
| Transformer layers | 12 |
| Attention heads | 8 |
| Hidden dim | 768 |
| Total params | ~125M |
| Max seq length | 512 tokens |
| Mixed vocabulary | 8192 image codes + 50257 BPE text tokens |

---

## 📋 Phase-wise Git Commits (follow this order)

```
Phase 1: Project scaffold, configs, README
Phase 2: VQ-VAE implementation + unit tests
Phase 3: Unified tokenizer
Phase 4: Decoder-only transformer
Phase 5: Dataset loaders + augmentation
Phase 6: Training loop + losses
Phase 7: Inference + visualization
Phase 8: FastAPI endpoint
Phase 9: Gradio app
Phase 10: Docker setup
Phase 11: Kaggle notebook
Phase 12: HuggingFace export + model card
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE)

---

*Built by [Sagnik120](https://github.com/Sagnik120) — IIT Jodhpur*
