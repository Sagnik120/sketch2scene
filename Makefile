.PHONY: install setup data train-dev train api gradio docker test push-hf git-init

# ── Setup ─────────────────────────────────────────────────────────────────────
install:
	python -m venv venv && source venv/bin/activate && pip install -e ".[dev]"

setup: install
	cp .env.example .env
	@echo "✓ Edit .env with your HF_TOKEN"

# ── Data ──────────────────────────────────────────────────────────────────────
data:
	python scripts/download_data.py --dataset quickdraw --output data/raw/ --max_per_class 5000

# ── Training ──────────────────────────────────────────────────────────────────
train-dev:
	python scripts/train.py --config configs/train_config.yaml --device mps --batch_size 4 --debug

train:
	python scripts/train.py --config configs/train_config.yaml --device mps --batch_size 8

train-kaggle:
	python scripts/train.py --config configs/train_config.yaml --device cuda --batch_size 32

# ── API ───────────────────────────────────────────────────────────────────────
api:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# ── Gradio ────────────────────────────────────────────────────────────────────
gradio:
	python app/gradio_app.py

# ── Docker ────────────────────────────────────────────────────────────────────
docker:
	cd docker && docker-compose up --build

# ── Tests ─────────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v

# ── HuggingFace export ────────────────────────────────────────────────────────
push-hf:
	python scripts/export_hf.py --repo Sagnik120/sketch2scene

# ── Git init (run once) ───────────────────────────────────────────────────────
git-init:
	git init
	git remote add origin https://github.com/Sagnik120/sketch2scene.git
	git add .
	git commit -m "Phase 1: Project scaffold, configs, README"
	@echo "✓ Now push with: git push -u origin main"
