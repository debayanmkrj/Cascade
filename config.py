"""Configuration for Low-k-cdr"""
import os
from pathlib import Path

# Base paths
PROJECT_DIR = Path(__file__).parent  # lowkcdr_jspy root
BASE_DIR = Path(__file__).parent.parent  # Dataset root (for shared assets)
GRAPH_FILE = PROJECT_DIR / "rag_cache" / "knowledge_graph.pkl"
CHUNKS_FILE = BASE_DIR / "annotated_chunks" / "all_chunks.json"  # unused at runtime

# LLM Settings
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:latest"  # General purpose model for overall reasoning
MODEL_NAME_REASONING = "llama3.2:latest"  # For semantic reasoning / node planning
MODEL_NAME_CODING = "qwen2.5-coder:7b"  # For Mason code generation (GLSL/JS)
MODEL_NAME_FALLBACK = "llama3.2:latest"  # Fallback when other models fail
MODEL_NAME_REVIEW = "llama3.2:latest"  # Aider supervisory review model

# Cloud LLM Settings — set USE_CLOUD_LLM=1 in env to activate; set USE_CLOUD_LLM=0 to roll back
USE_CLOUD_LLM = os.environ.get("USE_CLOUD_LLM", "0") == "1"
CLOUD_API_KEY = "" #add here
CLOUD_API_BASE = os.environ.get("CLOUD_API_BASE", "https://api.ollama.com/v1")
CLOUD_MODEL_CODING = "qwen3-coder:480b-cloud"      # Replaces MODEL_NAME_CODING in cloud mode
CLOUD_MODEL_REASONING = "deepseek-v3.2:cloud"       # Replaces MODEL_NAME_REASONING in cloud mode

# Effective models — auto-select cloud or local based on USE_CLOUD_LLM flag
EFFECTIVE_MODEL_CODING = CLOUD_MODEL_CODING if USE_CLOUD_LLM else MODEL_NAME_CODING
EFFECTIVE_MODEL_FALLBACK = MODEL_NAME_FALLBACK   # Always local — cloud fallback makes no sense if primary already failed
EFFECTIVE_MODEL_REVIEW = CLOUD_MODEL_REASONING if USE_CLOUD_LLM else MODEL_NAME_REVIEW

# Embedding Models
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CLIP_MODEL = "openai/clip-vit-base-patch32"

# RAG Settings - No limits, retrieve all relevant results
TOP_K_UI_NODES = 100  # No artificial limit
TOP_K_NODE_ARCHETYPES = 100  # No artificial limit

# Creative Level Weights (from Phase 1 spec - section 3.2)
DEFAULT_LEVEL_WEIGHTS = {
    'surface': 0.4,
    'flow': 0.3,
    'narrative': 0.3
}

# Divergence Scaling (from Phase 1 spec - section 3.3)
# Mode multipliers: functional/aesthetic/flow
MODE_FACTORS = {
    "functional": {"surface": 0.5, "flow": 0.4, "narrative": 0.3},
    "aesthetic": {"surface": 1.0, "flow": 0.6, "narrative": 1.0},
    "flow": {"surface": 0.7, "flow": 0.9, "narrative": 0.7}
}

# UI Generation (from Phase 1 spec - section 5)
MIN_UI_CONTROLS = 5
MAX_UI_CONTROLS = 12
UI_CANDIDATE_K_MIN = 8
UI_CANDIDATE_K_MAX = 15
ENTROPY_SCALING_LAMBDA = 1.5  # Guided entropy scaling parameter

# Node Archetypes (from Phase 1 spec - section 6)
# Widened range to allow more dynamic node counts based on workflow complexity
MIN_NODE_COUNT = 3
MAX_NODE_COUNT = 25
BASE_NODES = {
    "surface": 2,
    "flow": 2,
    "narrative": 1
}
EXTRA_NODES_SCALE = {
    "surface": 4,
    "flow": 3,
    "narrative": 3
}

# Temperature ranges for sampling
TEMPERATURE_MIN = 0.3
TEMPERATURE_MAX = 1.5

# Brand/Emotion Labels (for CLIP zero-shot)
EMOTION_LABELS = [
    "excitement", "calmness", "joy", "melancholy", "energy", "serenity",
    "playfulness", "seriousness", "warmth", "coldness", "chaos", "order",
    "luxury", "simplicity", "organic", "geometric", "bold", "subtle",
    "trust", "tension", "surprise", "nostalgia", "futurism", "intensity"
]

BRAND_ATTRIBUTE_LABELS = [
    "trustworthy", "innovative", "professional", "creative", "friendly",
    "sophisticated", "accessible", "premium", "minimal", "expressive",
    "technical", "artistic", "modern", "timeless", "playful", "serious",
    "calm", "intense"
]

# UI Control Types (from TouchDesigner/Max MSP) - NORMALIZED
UI_CONTROL_TYPES = [
    "slider", "toggle", "dropdown", "xy_pad", "color",
    "text_field", "number_input", "button", "radio_group", "envelope",
    "knob", "fader"
]

# Node Roles (from spec section 6.1)
NODE_ROLES = ["input", "process", "output", "control", "utility"]

# Image Search API
PEXELS_API_KEY = "w6A5SuJlWpjeBWE2RF22RLRYMMAuNx9PcuJx1nERzT5qHzSoqcZDsWHU"

# CLIP Labels for clustering
VISUAL_MOOD_LABELS = [
    "minimalist", "maximalist", "energetic", "calm", "serious",
    "playful", "melancholic", "optimistic", "professional", "creative",
    "organic", "geometric", "warm", "cold", "luxurious", "raw",
    "glitchy", "smooth", "chaotic", "ordered"
]

# Output
OUTPUT_DIR = PROJECT_DIR / "outputs"
SESSIONS_DIR = PROJECT_DIR / "sessions"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SESSIONS_DIR, exist_ok=True)

# Control parameter normalization by type
CONTROL_SCHEMAS = {
    "slider": {"min": "min_val", "max": "max_val", "default": "default_val", "step": "step"},
    "knob": {"min": "min_val", "max": "max_val", "default": "default_val"},
    "toggle": {"default": "default_state"},
    "dropdown": {"options": "options", "default": "default_option"},
    "xy_pad": {"x_range": "x_range", "y_range": "y_range", "default_x": "default_x", "default_y": "default_y"},
    "color": {"default": "default_color"},
    "envelope": {"points": "points", "default": "default_envelope"},
    "button": {"action": "action"},
    "fader": {"min": "min_val", "max": "max_val", "default": "default_val"}
}
