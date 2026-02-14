"""Utility functions for Low-k-cdr"""
from typing import Dict, List, Optional
from config import CONTROL_SCHEMAS

# Global reference to RAG library for dynamic category validation
_rag_library_ref: Optional['UINodeLibrary'] = None


def set_rag_library(rag_library):
    """Set global reference to RAG library for dynamic category validation"""
    global _rag_library_ref
    _rag_library_ref = rag_library


def validate_and_coerce_category(category: str) -> str:
    """
    Validate category, using dynamic categories from knowledge graph if available.
    Falls back to basic validation if no RAG library is set.
    Fixes: "generator|effect" -> "generator", invalid -> kept as-is or "utility"
    """
    if not category:
        return "utility"

    # Handle pipe-separated categories (LLM sometimes returns multiple)
    if "|" in category:
        parts = category.split("|")
        for part in parts:
            clean = part.strip().lower()
            # Use dynamic categories from knowledge graph if available
            if _rag_library_ref:
                available_categories = _rag_library_ref.discovered_categories
                if clean in available_categories:
                    return clean
        # Return first part if none match
        return parts[0].strip().lower()

    # Clean category
    clean = category.strip().lower()

    # Use dynamic categories from knowledge graph if available
    if _rag_library_ref and _rag_library_ref.discovered_categories:
        if clean in _rag_library_ref.discovered_categories:
            return clean
        # If not found but categories exist, keep the LLM's category
        # (it might be a new valid category from the context)
        return clean

    # Fallback fuzzy matching for common abbreviations
    fuzzy_map = {
        "gen": "generator",
        "generate": "generator",
        "fx": "effect",
        "effects": "effect",
        "process": "modifier",
        "out": "output",
        "ctrl": "control",
        "util": "utility",
        "comp": "composite",
        "route": "router"
    }

    if clean in fuzzy_map:
        return fuzzy_map[clean]

    # Accept the category as-is (trust the LLM's context-driven category)
    return clean


def _safe_float(value, default=0.0):
    """Safely convert value to float, handling empty strings and None."""
    if value is None or value == '':
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def normalize_control_parameters(control_type: str, parameters: Dict) -> Dict:
    """
    Normalize control parameters to canonical schema.
    Outputs canonical keys: min/max/default (NOT min_val/max_val/default_val).
    Fixes inconsistent LLM outputs (e.g., toggle with min/max).
    """
    if control_type not in CONTROL_SCHEMAS:
        # Unknown type, return as-is
        return parameters

    normalized = {}

    if control_type == "slider" or control_type == "fader":
        # CANONICAL: min/max/default/step (not *_val)
        normalized["min"] = _safe_float(parameters.get("min", parameters.get("min_val", 0.0)), 0.0)
        normalized["max"] = _safe_float(parameters.get("max", parameters.get("max_val", 1.0)), 1.0)
        normalized["default"] = _safe_float(parameters.get("default", parameters.get("default_val", 0.5)), 0.5)
        normalized["step"] = _safe_float(parameters.get("step", 0.01), 0.01)

    elif control_type == "knob":
        # CANONICAL: min/max/default (not *_val)
        normalized["min"] = _safe_float(parameters.get("min", parameters.get("min_val", 0.0)), 0.0)
        normalized["max"] = _safe_float(parameters.get("max", parameters.get("max_val", 1.0)), 1.0)
        normalized["default"] = _safe_float(parameters.get("default", parameters.get("default_val", 0.5)), 0.5)

    elif control_type == "toggle":
        # CANONICAL: default (boolean, not default_state)
        normalized["default"] = bool(parameters.get("default", parameters.get("default_state", False)))

    elif control_type == "dropdown":
        normalized["options"] = parameters.get("options", ["option1", "option2", "option3"])
        normalized["default"] = parameters.get("default", parameters.get("default_option", normalized["options"][0]))

    elif control_type == "xy_pad":
        normalized["x_range"] = parameters.get("x_range", [0.0, 1.0])
        normalized["y_range"] = parameters.get("y_range", [0.0, 1.0])
        normalized["default_x"] = float(parameters.get("default_x", 0.5))
        normalized["default_y"] = float(parameters.get("default_y", 0.5))

    elif control_type == "color":
        # CANONICAL: default (hex string)
        normalized["default"] = parameters.get("default", parameters.get("default_color", "#ffffff"))

    elif control_type == "envelope":
        # CANONICAL: points is a list of [x,y] pairs, move default_envelope -> points if needed
        if "points" in parameters:
            normalized["points"] = parameters["points"]
        elif "default_envelope" in parameters:
            normalized["points"] = parameters["default_envelope"]
        elif "default" in parameters:
            normalized["points"] = parameters["default"]
        else:
            normalized["points"] = [[0, 0], [0.5, 1], [1, 0]]

    elif control_type == "button":
        normalized["action"] = parameters.get("action", "trigger")

    else:
        # Unknown type, keep original
        normalized = parameters.copy()

    return normalized


def infer_role_from_concept(concept_label: str) -> str:
    """
    Infer node role from concept label.
    From spec section 6.1: input/process/output/control/utility
    """
    concept_lower = concept_label.lower()

    # Input indicators
    if any(kw in concept_lower for kw in ["input", "in", "source", "audio_in", "video_in", "midi", "osc", "capture"]):
        return "input"

    # Output indicators
    if any(kw in concept_lower for kw in ["output", "out", "render", "export", "display", "speaker", "write"]):
        return "output"

    # Control indicators
    if any(kw in concept_lower for kw in ["control", "lfo", "envelope", "sequencer", "timer", "trigger", "switch"]):
        return "control"

    # Utility indicators
    if any(kw in concept_lower for kw in ["math", "logic", "convert", "map", "scale", "utility", "select", "route"]):
        return "utility"

    # Default to process
    return "process"
