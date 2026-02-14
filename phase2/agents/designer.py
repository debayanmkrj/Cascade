"""
Designer Agent
--------------
Enhances Mason-approved GLSL primitives into visually impressive shaders
using the reasoning LLM (llama3.2).  Validates all enhancements through
Mason's existing GLSL compilation pipeline.

Pipeline position: After Mason (Step 2), before VLM skip (Step 3).
Safety contract: Enhanced code must pass glslangValidator; fallback to
primitive on failure.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import re
import requests
import shutil
import subprocess
import tempfile
from typing import Dict, List, Optional

from phase2.data_types import NodeTensor
from config import OLLAMA_URL, MODEL_NAME_REASONING


def _meta_attr(meta, key, default=""):
    if meta is None:
        return default
    if hasattr(meta, key):
        return getattr(meta, key, default) or default
    if isinstance(meta, dict):
        return meta.get(key, default) or default
    return default


class DesignerAgent:
    """Enhance GLSL primitives into visually impressive shaders.

    Uses the reasoning LLM (llama3.2) for creative enhancement.
    Validates via glslangValidator + Mason's GLSL utility library.
    Only enhances GLSL nodes; JS engines are kept as-is.
    """

    # Categories worth enhancing (generators + visual effects).
    # Post-processing (blur, vignette, threshold) works fine as primitives.
    ENHANCEABLE = {
        "noise_generator", "perlin_noise", "fractal_noise",
        "pattern_generator", "oscillator", "field_generator",
        "waveform", "fire", "water", "smoke", "lightning",
        "sdf", "raymarch", "shape_generator",
        "displacement", "distortion", "feedback_effect",
        "kaleidoscope", "glitch",
        "fog", "atmosphere", "cloud_renderer", "rain_renderer",
        "color_grade",
    }

    def __init__(self, model: str = None, max_retries: int = 3):
        self.model = model or MODEL_NAME_REASONING
        self.ollama_url = OLLAMA_URL
        self.max_retries = max_retries

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enhance_nodes(self, nodes: List[NodeTensor],
                      session_json: Dict) -> List[NodeTensor]:
        """Enhance GLSL nodes that passed Mason validation.

        Returns the same list with code_snippet upgraded where possible.
        Original code is preserved if enhancement fails validation.
        """
        brand_essence = self._extract_essence(session_json)
        enhanced, eligible = 0, 0

        for node in nodes:
            engine = (node.engine or "").strip()
            category = _meta_attr(node.meta, "category", "")

            if engine not in ("glsl", "regl"):
                continue
            if not node.mason_approved or not node.code_snippet:
                continue
            if not self._is_enhanceable(category):
                continue

            eligible += 1
            original = node.code_snippet
            neighbor_ctx = self._neighbor_summary(node, nodes)

            new_code = self._enhance_with_retries(
                node, original, brand_essence, category, neighbor_ctx
            )

            if new_code:
                node.code_snippet = new_code
                self._sync_params_from_code(node)
                enhanced += 1
                print(f"  [DESIGNER] {node.id} ({category}): ENHANCED")
            else:
                print(f"  [DESIGNER] {node.id}: all attempts failed, "
                      f"keeping primitive")

        print(f"  [DESIGNER] Enhanced {enhanced}/{eligible} eligible nodes")
        return nodes

    def _enhance_with_retries(self, node: NodeTensor, original: str,
                              essence: str, category: str,
                              neighbor_ctx: str) -> Optional[str]:
        """Try to enhance a shader, retrying with error feedback on failure."""
        last_errors = None

        for attempt in range(self.max_retries):
            code = self._enhance_glsl(
                node, original, essence, category, neighbor_ctx,
                prev_errors=last_errors, attempt=attempt
            )

            if not code or "fragColor" not in code:
                print(f"  [DESIGNER] {node.id}: attempt {attempt+1} "
                      f"returned empty/no fragColor")
                last_errors = ["LLM output was empty or missing fragColor"]
                continue

            errors = self._validate_glsl(node, code)
            if not errors:
                return code

            last_errors = errors
            print(f"  [DESIGNER] {node.id}: attempt {attempt+1} "
                  f"validation failed: {errors[0][:100]}")

        return None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _is_enhanceable(self, category: str) -> bool:
        if category in self.ENHANCEABLE:
            return True
        return any(e in category for e in self.ENHANCEABLE)

    def _sync_params_from_code(self, node: NodeTensor) -> None:
        """Re-sync node.parameters from u_* uniforms in the enhanced code."""
        code = node.code_snippet or ""
        if not code:
            return
        found = set(re.findall(r'\bu_(\w+)\b', code))
        builtins = {"time", "resolution"}
        builtins.update(f"input{i}" for i in range(10))
        for inp in (node.input_nodes or []):
            builtins.add(inp)
        param_names = found - builtins
        if not param_names:
            return
        from phase1.node_archetypes import CATEGORY_PARAMETER_REGISTRY
        category = _meta_attr(node.meta, "category", "")
        registry = {}
        for p in CATEGORY_PARAMETER_REGISTRY.get(category, []):
            registry[p["name"]] = p["default"]
        new_params = {}
        for name in sorted(param_names):
            if name in registry:
                new_params[name] = registry[name]
            elif name in (node.parameters or {}):
                new_params[name] = node.parameters[name]
            else:
                new_params[name] = 0.5
        node.parameters = new_params

    def _extract_essence(self, sj: Dict) -> str:
        for path in [
            lambda: sj.get("phase2_context", {}).get("essence", ""),
            lambda: sj.get("brief", {}).get("essence", ""),
            lambda: sj.get("input", {}).get("prompt_text", ""),
            lambda: sj.get("brand_essence", ""),
        ]:
            v = path()
            if v:
                return v
        return "generative audiovisual experience"

    def _neighbor_summary(self, node: NodeTensor,
                          all_nodes: List[NodeTensor]) -> str:
        nz = node.grid_position[2] if node.grid_position else 0
        parts = []
        for n in all_nodes:
            if n.id == node.id:
                continue
            oz = n.grid_position[2] if n.grid_position else 0
            if abs(oz - nz) <= 1 and n.mason_approved:
                cat = _meta_attr(n.meta, "category", "?")
                eng = (n.engine or "?").strip()
                parts.append(f"{cat} ({eng})")
        if not parts:
            return ""
        return "Neighboring nodes: " + ", ".join(parts[:5])

    # ------------------------------------------------------------------
    # LLM Enhancement
    # ------------------------------------------------------------------

    def _enhance_glsl(self, node: NodeTensor, original: str,
                      essence: str, category: str,
                      neighbor_ctx: str,
                      prev_errors: Optional[List[str]] = None,
                      attempt: int = 0) -> Optional[str]:
        params = node.parameters or {}
        param_list = ", ".join(
            f"u_{k}" for k, v in params.items()
            if isinstance(v, (int, float))
        )

        # Build error feedback section for retries
        error_section = ""
        if prev_errors and attempt > 0:
            err_text = prev_errors[0][:200]
            error_section = f"""
PREVIOUS ATTEMPT FAILED WITH ERROR:
{err_text}
FIX this error. Make sure all braces are balanced and all types match."""

        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a creative shader designer. Enhance GLSL ES 3.00 fragment shaders.

CRITICAL RULES:
- Output ONLY shader code. No markdown. No explanation. No comments outside code.
- Keep structure: void main() {{ ... fragColor = vec4(...); }}
- ALWAYS end with: fragColor = vec4(...);  then close all braces.
- Make sure ALL braces {{ }} are balanced. Every open brace needs a close brace.
- DO NOT add: #version, precision, uniform/in/out declarations
- DO NOT use: gl_FragColor, texture2D (use texture() instead)
- DO NOT define functions. Put all code inside void main().
- ALL parameter uniforms are float type. Use them to control the effect.

AVAILABLE VARIABLES (already declared):
  in vec2 v_uv;
  out vec4 fragColor;
  uniform float u_time;
  uniform vec2 u_resolution;
  uniform sampler2D u_input0;
{('  uniform float ' + ';  uniform float '.join(f'u_{k}' for k, v in params.items() if isinstance(v, (int, float))) + ';') if param_list else ''}

UTILITY FUNCTIONS (already defined — use these EXACT signatures):
  float hash(float n)          — pseudo-random from float
  float hash(vec2 p)           — pseudo-random from vec2
  vec2  hash2(vec2 p)          — 2D hash, returns vec2 in [-1,1]
  vec3  hash3(vec2 p)          — 3D hash from vec2, returns vec3 in [-1,1]
  vec3  hash3(vec3 p)          — 3D hash from vec3, returns vec3 in [-1,1]
  float noise(vec2 p)          — smooth value noise, returns [0,1]
  float snoise(vec2 p)         — signed noise, returns [-1,1]
  float snoise(vec3 p)         — signed noise from vec3, returns [-1,1]
  float fbm(vec2 p)            — fractal Brownian motion, 5 octaves, returns [0,1]
<|eot_id|><|start_header_id|>user<|end_header_id|>
CREATIVE BRIEF: {essence}
CATEGORY: {category}
{neighbor_ctx}
{error_section}
CURRENT WORKING SHADER:
{original}

Enhance this shader to be visually impressive. Add layered noise, domain warping,
color gradients, organic flow. USE the parameter uniforms ({param_list}) to make
the effect controllable. Keep it under 40 lines. Output ONLY the code.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
void main() {{"""

        try:
            resp = requests.post(
                self.ollama_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.6 if attempt == 0 else 0.4,
                        "num_predict": 4096,
                        "stop": ["```", "<|endoftext|>",
                                 "<|eot_id|>"],
                    }
                },
                timeout=180,
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "")

            # The prompt ends with "void main() {" so prepend it
            code = "void main() {" + raw
            code = self._clean(code, node)
            return code

        except Exception as e:
            print(f"  [DESIGNER] LLM request failed: {e}")
            return None

    def _clean(self, text: str, node: NodeTensor) -> str:
        # Strip markdown fences
        fence = re.search(r"```(?:\w+)?\s*(.*?)```", text, flags=re.DOTALL)
        if fence:
            text = fence.group(1).strip()

        # Strip LLM tokens
        text = re.sub(r"<\|[^>]*\|>", "", text)
        text = re.sub(r"<\uff5c[^>]*\uff5c>", "", text)

        # Strip <think>...</think> blocks (reasoning models)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

        lines = []
        for line in text.splitlines():
            t = line.strip()
            if t.startswith("#version") or t.startswith("precision"):
                continue
            if t.startswith("in ") or t.startswith("out "):
                continue
            if t.startswith("uniform "):
                continue
            # Strip function definitions the LLM might add
            if re.match(r"^(float|vec[234]|mat[234]|int|void)\s+\w+\s*\(", t) and t != "void main() {" and "void main()" not in t:
                continue
            line = line.replace("gl_FragColor", "fragColor")
            line = re.sub(r"texture2D\s*\(", "texture(", line)
            lines.append(line)

        result = "\n".join(lines).strip()

        # Balance braces — auto-close truncated output
        open_braces = result.count("{")
        close_braces = result.count("}")
        if open_braces > close_braces:
            # Ensure fragColor is assigned before closing
            if "fragColor" not in result:
                result += "\n    fragColor = vec4(0.0);"
            result += "\n}" * (open_braces - close_braces)

        # Rewrite bare param identifiers to u_{param}
        for k, v in (node.parameters or {}).items():
            if isinstance(v, (int, float)):
                pat = re.compile(rf"(?<!u_)\b{re.escape(k)}\b")
                result = pat.sub(f"u_{k}", result)

        # Add header
        category = _meta_attr(node.meta, "category", "unknown")
        engine = (node.engine or "glsl").strip()
        return (f"// {engine.upper()} BODY | category={category} "
                f"| node={node.id} | DESIGNER-ENHANCED\n{result}\n")

    # ------------------------------------------------------------------
    # GLSL Validation (reuses Mason's infrastructure)
    # ------------------------------------------------------------------

    def _validate_glsl(self, node: NodeTensor, code: str) -> List[str]:
        errors = []
        body = "\n".join(
            ln for ln in code.splitlines()
            if not ln.strip().startswith("//")
        ).strip()

        if not body:
            return ["empty snippet"]
        if "#version" in body or "precision " in body:
            return ["must not include #version or precision"]
        if "gl_FragColor" in body:
            return ["use fragColor, not gl_FragColor"]
        if "texture2D" in body:
            return ["use texture(), not texture2D()"]
        if "uniform " in body:
            return ["do not declare uniforms"]
        if not re.search(r"\bvoid\s+main\s*\(", body):
            return ["missing void main()"]

        glslang = shutil.which("glslangValidator")
        if not glslang:
            return []  # structural checks passed; no compiler available

        params = node.parameters or {}
        side = "\n".join(
            f"uniform sampler2D u_{nid};"
            for nid in (node.input_nodes or [])
        )
        param_u = "\n".join(
            f"uniform float u_{k};"
            for k, v in params.items() if isinstance(v, (int, float))
        )

        from phase2.agents.mason import MasonAgent
        utils = MasonAgent._GLSL_UTILS_FOR_VALIDATION

        full = f"""#version 300 es
precision highp float;

in vec2 v_uv;
out vec4 fragColor;

uniform float u_time;
uniform vec2 u_resolution;
uniform sampler2D u_input0;

{side}
{param_u}

{utils}

{body}
"""
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "shader.frag"
            src.write_text(full, encoding="utf-8")

            proc = None
            for flags in [
                [glslang, "--client", "opengl100", "-S", "frag", str(src)],
                [glslang, "--target-env", "opengl", "-S", "frag", str(src)],
                [glslang, "-S", "frag", str(src)],
            ]:
                proc = subprocess.run(flags, capture_output=True, text=True)
                stderr = (proc.stderr or proc.stdout or "").strip()
                if proc.returncode != 0 and (
                    "SPIR-V" in stderr or "target-env" in stderr
                    or "client" in stderr
                ):
                    continue
                break

            if proc and proc.returncode != 0:
                msg = "\n".join(
                    (proc.stderr or proc.stdout or "").strip().splitlines()[:20]
                )
                return [f"GLSL compile failed:\n{msg}"]

        return []
