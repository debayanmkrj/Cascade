"""
Uniform Validator - Dynamic extraction and validation of shader uniforms vs parameters

This module ensures consistency between:
1. What uniforms the generated code uses (u_* in GLSL, params.* in JS)
2. What parameters are declared in the node spec

NO hardcoded schemas - fully dynamic extraction and reconciliation.
"""

import re
from typing import Dict, List, Set, Tuple, Any, Optional


# Standard uniforms that are always available (don't need to be in params)
STANDARD_UNIFORMS = {
    # Time/resolution (provided by runtime)
    "u_time", "u_resolution",
    # Input textures (provided by runtime based on connections)
    "u_input0", "u_input1", "u_input2", "u_input3",
    "u_input4", "u_input5", "u_input6", "u_input7",
}

# Sensible defaults for common parameter names.
# Rule: if the param is used as mix(base, effect, u_param), default should be 0.5 (moderate).
# If used as a multiplier where 1.0=identity, keep 1.0. If used as speed/scale keep natural.
DEFAULT_VALUES = {
    # Motion/animation
    "speed": 0.5,
    "velocity": 0.5,
    "rate": 1.0,
    "rotation": 0.5,
    "ease": 0.05,      # easing factor — small value = smooth

    # Frequency/scale
    "frequency": 6.0,
    "freq": 6.0,
    "scale": 1.0,
    "zoom": 1.0,
    "tile": 2.0,       # tiling count — 1.0 = single tile, 2+ = repeating

    # Intensity/strength — use 0.5 as "moderate effect", not 1.0 (which was identity for some uses)
    "amplitude": 0.5,
    "intensity": 0.5,
    "strength": 0.5,
    "amount": 0.5,
    "power": 1.0,

    # Visual effects — use mix-based defaults (0.5 = half effect, visible)
    "glow": 0.5,
    "blur": 0.4,
    "grade": 0.5,       # as mix weight: 0=no grade, 1=full grade
    "tone": 0.5,        # as mix weight: 0=no tone, 1=full tone
    "glitch": 0.0,      # off by default — dramatic when turned on
    "distortion": 0.3,
    "warp": 0.3,

    # Color adjustments
    "brightness": 0.0,  # as additive offset: 0=no change, +0.3=brighter
    "contrast": 1.2,    # multiplicative: 1.0=no change, >1=more contrast
    "saturation": 1.0,  # multiplier: 1.0=unchanged
    "hue_shift": 0.0,
    "hue": 0.0,
    "gamma": 1.0,
    "exposure": 0.0,    # EV stops: 0=no change, 1=+1 stop brighter
    "transparency": 1.0,
    "opacity": 1.0,
    "alpha": 1.0,

    # Thresholds/limits
    "threshold": 0.5,
    "min": 0.0,
    "max": 1.0,
    "cutoff": 0.5,

    # Size/radius
    "size": 1.0,
    "radius": 0.5,
    "width": 1.0,
    "height": 1.0,

    # Counts
    "count": 100,
    "segments": 6,
    "octaves": 4,
    "iterations": 4,

    # Blending
    "mix": 0.5,
    "blend": 0.5,

    # Offsets
    "offset": 0.0,
    "offset_x": 0.0,
    "offset_y": 0.0,
    "phase": 0.0,

    # Decay/feedback
    "decay": 0.95,
    "feedback": 0.9,
    "damping": 0.98,

    # Noise specific
    "lacunarity": 2.0,
    "persistence": 0.5,
    "gain": 0.5,
    "perlin": 0.5,
    "noise": 0.5,

    # Composite/blend modes
    "field": 0.3,       # field density — small default avoids overdraw
    "bezier": 0.5,
    "mirror": 0.0,      # off by default — binary toggle (0=off, 1=on)
    "transform": 1.0,
    "texture": 1.0,
}


class UniformValidator:
    """
    Validates and reconciles uniforms between code and parameters.

    Usage:
        validator = UniformValidator()
        result = validator.validate_and_reconcile(code, params, engine)

        if result["needs_fix"]:
            # Use result["fixed_params"] or result["fixed_code"]
    """

    def __init__(self):
        self.standard_uniforms = STANDARD_UNIFORMS
        self.default_values = DEFAULT_VALUES

    def extract_uniforms_from_glsl(self, code: str) -> Set[str]:
        """
        Extract all u_* uniform references from GLSL code.
        Returns set of uniform names (without u_ prefix).
        """
        if not code:
            return set()

        # Match u_<name> but not inside comments
        # Remove single-line comments first
        code_no_comments = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
        # Remove multi-line comments
        code_no_comments = re.sub(r'/\*.*?\*/', '', code_no_comments, flags=re.DOTALL)

        # Find all u_<identifier> patterns
        matches = re.findall(r'\bu_([a-zA-Z_][a-zA-Z0-9_]*)\b', code_no_comments)

        # Filter out standard uniforms and node references (u_node_*)
        uniforms = set()
        for m in matches:
            full_name = f"u_{m}"
            if full_name not in self.standard_uniforms and not m.startswith("node_"):
                uniforms.add(m)

        return uniforms

    def extract_params_from_js(self, code: str) -> Set[str]:
        """
        Extract all params.* references from JavaScript code.
        Returns set of parameter names.
        """
        if not code:
            return set()

        # Remove comments
        code_no_comments = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
        code_no_comments = re.sub(r'/\*.*?\*/', '', code_no_comments, flags=re.DOTALL)

        # Find params.name or params["name"] or params['name']
        dot_matches = re.findall(r'\bparams\.([a-zA-Z_][a-zA-Z0-9_]*)\b', code_no_comments)
        bracket_matches = re.findall(r'\bparams\[["\']([a-zA-Z_][a-zA-Z0-9_]*)["\']\]', code_no_comments)

        return set(dot_matches) | set(bracket_matches)

    def validate_and_reconcile(
        self,
        code: str,
        parameters: Dict[str, Any],
        engine: str
    ) -> Dict[str, Any]:
        """
        Validate code against parameters and reconcile mismatches.

        Returns:
            {
                "valid": bool,
                "needs_fix": bool,
                "missing_params": list,  # uniforms in code but not in params
                "unused_params": list,   # params not used in code
                "fixed_params": dict,    # reconciled parameters
                "fixed_code": str,       # original code (unchanged)
                "analysis": str          # human-readable summary
            }
        """
        # Extract uniforms based on engine
        if engine in ("glsl", "regl"):
            code_uniforms = self.extract_uniforms_from_glsl(code)
        elif engine in ("three_js", "p5", "p5js", "canvas2d", "events", "js_module"):
            code_uniforms = self.extract_params_from_js(code)
        else:
            # Unknown engine, skip validation
            return {
                "valid": True,
                "needs_fix": False,
                "missing_params": [],
                "unused_params": [],
                "fixed_params": parameters,
                "fixed_code": code,
                "analysis": f"Skipped validation for engine: {engine}"
            }

        # Get declared parameter names
        declared_params = set(parameters.keys())

        # Find mismatches
        missing_params = code_uniforms - declared_params
        unused_params = declared_params - code_uniforms

        # Reconcile: add missing params with defaults
        fixed_params = dict(parameters)
        for missing in missing_params:
            default = self._get_default_value(missing)
            fixed_params[missing] = default

        # Build analysis
        analysis_parts = []
        if missing_params:
            analysis_parts.append(f"Missing params added: {list(missing_params)}")
        if unused_params:
            analysis_parts.append(f"Unused params (kept): {list(unused_params)}")
        if not missing_params and not unused_params:
            analysis_parts.append("All uniforms match declared parameters")

        return {
            "valid": len(missing_params) == 0,
            "needs_fix": len(missing_params) > 0,
            "missing_params": list(missing_params),
            "unused_params": list(unused_params),
            "fixed_params": fixed_params,
            "fixed_code": code,  # Code unchanged, we fix params instead
            "analysis": "; ".join(analysis_parts)
        }

    def _get_default_value(self, param_name: str) -> float:
        """Get sensible default value for a parameter name."""
        # Direct match
        if param_name in self.default_values:
            return self.default_values[param_name]

        # Partial match (e.g., "glow_intensity" matches "intensity")
        param_lower = param_name.lower()
        for key, value in self.default_values.items():
            if key in param_lower:
                return value

        # Default fallback
        return 1.0

    def batch_validate(
        self,
        nodes: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Validate multiple nodes and return fixed versions.

        Args:
            nodes: List of node dicts with 'code_snippet', 'parameters', 'engine'

        Returns:
            List of nodes with reconciled parameters
        """
        fixed_nodes = []

        for node in nodes:
            code = node.get("code_snippet", "")
            params = node.get("parameters", {})
            engine = node.get("engine", "glsl")
            node_id = node.get("id", "unknown")

            result = self.validate_and_reconcile(code, params, engine)

            if result["needs_fix"]:
                print(f"  [UniformValidator] {node_id}: {result['analysis']}")

            # Create fixed node
            fixed_node = dict(node)
            fixed_node["parameters"] = result["fixed_params"]
            fixed_nodes.append(fixed_node)

        return fixed_nodes


# Singleton instance
_validator = None


def get_uniform_validator() -> UniformValidator:
    """Get or create singleton UniformValidator instance."""
    global _validator
    if _validator is None:
        _validator = UniformValidator()
    return _validator


def validate_node(node: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to validate and fix a single node.

    Returns the node with reconciled parameters.
    """
    validator = get_uniform_validator()

    code = node.get("code_snippet", "")
    params = node.get("parameters", {})
    engine = node.get("engine", "glsl")

    result = validator.validate_and_reconcile(code, params, engine)

    fixed_node = dict(node)
    fixed_node["parameters"] = result["fixed_params"]

    return fixed_node
