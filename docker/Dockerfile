FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .
RUN pip install -e . --no-deps

# Expose ports
EXPOSE 8000 7860

ENV MODEL_PATH=checkpoints/best
ENV DEVICE=cpu

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
