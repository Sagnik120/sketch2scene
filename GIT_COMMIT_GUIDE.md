# 📋 Phase-wise Git Commit Guide

Follow these phases to maximize meaningful commits on GitHub.
Each phase builds on the last — never commit broken code.

---

## Phase 1 — Project Scaffold
```bash
git init
git remote add origin https://github.com/Sagnik120/sketch2scene.git
git add README.md .gitignore Makefile .env.example setup.py requirements.txt requirements-dev.txt
git commit -m "feat: project scaffold — README, Makefile, deps, gitignore"
git push -u origin main
```

## Phase 2 — VQ-VAE
```bash
git add src/model/vqvae.py src/model/__init__.py configs/
git commit -m "feat(model): VQ-VAE with EMA codebook (8192 codes, 256-dim)"
git push
```

## Phase 3 — Tokenizer
```bash
git add src/model/tokenizer.py
git commit -m "feat(model): unified image+text tokenizer (vocab=66949)"
git push
```

## Phase 4 — Transformer
```bash
git add src/model/transformer.py
git commit -m "feat(model): decoder-only transformer 125M with RoPE, top-k/p sampling"
git push
```

## Phase 5 — Full Model Wrapper
```bash
git add src/model/sketch2scene.py
git commit -m "feat(model): SketchToScene wrapper — VQ-VAE + transformer + save/load"
git push
```

## Phase 6 — Dataset
```bash
git add src/data/ scripts/download_data.py
git commit -m "feat(data): SketchyCOCO + QuickDraw dataset loaders with augmentation"
git push
```

## Phase 7 — Training Loop
```bash
git add src/training/ scripts/train.py
git commit -m "feat(training): trainer with grad accum, cosine LR, MPS+CUDA support"
git push
```

## Phase 8 — Inference & Visualization
```bash
git add src/inference/ configs/inference_config.yaml
git commit -m "feat(inference): autoregressive generation + matplotlib visualization"
git push
```

## Phase 9 — Tests
```bash
git add tests/
git commit -m "test: unit tests for VQ-VAE, tokenizer, transformer"
git push
```

## Phase 10 — FastAPI
```bash
git add api/
git commit -m "feat(api): FastAPI /generate and /describe endpoints with CORS"
git push
```

## Phase 11 — Gradio App
```bash
git add app/
git commit -m "feat(app): Gradio UI for HuggingFace Spaces with sketch drawing tool"
git push
```

## Phase 12 — Docker
```bash
git add docker/
git commit -m "feat(docker): Dockerfile + docker-compose for API + Gradio services"
git push
```

## Phase 13 — Kaggle Notebook
```bash
git add notebooks/
git commit -m "feat(notebooks): Kaggle training notebook — VQ-VAE + transformer on T4 GPU"
git push
```

## Phase 14 — HuggingFace Export
```bash
git add scripts/export_hf.py
git commit -m "feat(deploy): HuggingFace Hub export script with model card"
git push
```

## Phase 15 — Polish
```bash
# After getting results, add sample outputs, update README with metrics
git add examples/ results/
git commit -m "docs: add demo outputs, training curves, sample generations"
git push
```

---

## Quick tag for HuggingFace Spaces
```bash
git tag v1.0.0
git push origin v1.0.0
```

---

*Each commit = one clean unit of work. Future employers will see the full build arc.*
