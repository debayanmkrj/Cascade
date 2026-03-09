#!/usr/bin/env bash
# =============================================================================
# Cascade — WSL Setup Script
# Run from the lowkcdr_jspy directory:  bash setup.sh
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "=========================================="
echo "  Cascade Setup"
echo "=========================================="

# --- GPU detection -----------------------------------------------------------
HAS_GPU=false
if command -v nvidia-smi &>/dev/null && nvidia-smi --query-gpu=name --format=csv,noheader &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    echo "  GPU detected: $GPU_NAME"
    HAS_GPU=true
else
    echo "  No GPU detected — installing CPU-only packages"
fi

# --- System dependencies -----------------------------------------------------
echo ""
echo "[1/5] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y python3 python3-venv python3-pip nodejs npm build-essential \
    libsndfile1 libgl1 libglib2.0-0 > /dev/null

echo "  node $(node --version), npm $(npm --version)"

# --- Python venv -------------------------------------------------------------
echo ""
echo "[2/5] Creating Python venv..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel -q

# --- Python packages ---------------------------------------------------------
echo ""
echo "[3/5] Installing Python packages..."

# Core
pip install -q \
    flask==3.0.0 \
    flask-cors==4.0.0 \
    werkzeug==3.1.4 \
    Jinja2==3.1.6 \
    python-dotenv \
    pydantic \
    pyyaml \
    rich \
    requests \
    httpx \
    tqdm \
    psutil

# LLM / AI routing
pip install -q \
    litellm \
    openai

# HuggingFace stack
pip install -q \
    transformers \
    accelerate \
    tokenizers \
    sentencepiece \
    huggingface_hub \
    safetensors \
    datasets \
    peft \
    tiktoken

# Vision / embeddings
pip install -q \
    sentence-transformers \
    Pillow \
    opencv-python-headless

# Numerics
pip install -q \
    numpy \
    scipy \
    scikit-learn \
    networkx \
    faiss-cpu

# PyTorch — GPU or CPU
if [ "$HAS_GPU" = true ]; then
    echo "  Installing PyTorch with CUDA support..."
    pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install -q bitsandbytes xformers
else
    echo "  Installing PyTorch CPU-only..."
    pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

# --- Node modules ------------------------------------------------------------
echo ""
echo "[4/5] Installing Node modules..."
npm install --silent

# --- Knowledge graph check ---------------------------------------------------
echo ""
echo "[5/5] Checking RAG knowledge graph..."
GRAPH_FILE="rag_cache/knowledge_graph.pkl"

if [ -f "$GRAPH_FILE" ]; then
    SIZE=$(du -sh "$GRAPH_FILE" | cut -f1)
    echo "  rag_cache/knowledge_graph.pkl found ($SIZE) — RAG ready"
else
    echo "  ERROR: rag_cache/knowledge_graph.pkl not found."
    echo "  Make sure the rag_cache/ directory is present inside lowkcdr_jspy/."
fi

# --- Done --------------------------------------------------------------------
echo ""
echo "=========================================="
echo "  Setup complete"
echo ""
echo "  CLIP (openai/clip-vit-base-patch32) and"
echo "  SBERT (all-MiniLM-L6-v2) will download"
echo "  automatically on first run (~400MB total)"
echo ""
echo "  To start (cloud mode):"
echo "    source venv/bin/activate"
echo "    export USE_CLOUD_LLM=1"
echo "    python app_web.py"
echo ""
echo "  Open: http://localhost:5000"
echo "=========================================="
