"""Semantic Reasoning Layer - RAG + LLM driven node stack selection.

Flow:
1. Query knowledge graph (RAG) for techniques relevant to the brief
2. Visual CLIP embedding context if images are provided
3. Brand context for emotional/aesthetic grounding
4. LLM generates SEMANTIC node descriptions with keywords
5. Mason uses keywords to find best code match via tag similarity

Output format:
[
    {"id": "particles_fluid_sim", "keywords": ["fluid", "organic", "smoke", "flow"]},
    {"id": "neon_color_grade", "keywords": ["neon", "glow", "vibrant", "synthwave"]},
]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import math
import re
import requests
from typing import List, Dict, Optional

import litellm
litellm.suppress_debug_info = True
litellm.set_verbose = False

from config import (
    OLLAMA_URL, MODEL_NAME_REASONING, MODEL_NAME_FALLBACK,
    CLOUD_API_KEY, CLOUD_API_BASE, DEFAULT_LEVEL_WEIGHTS,
)


class SemanticReasoner:
    """RAG + LLM driven node stack selection."""

    # GLSL/JS keywords that are NOT valid shader pipeline step names
    GARBAGE_IDS = {
        "float", "int", "bool", "void", "vec2", "vec3", "vec4",
        "mat2", "mat3", "mat4", "sampler2d", "sampler2D",
        "uniform", "varying", "attribute", "const", "in", "out",
        "highp", "mediump", "lowp", "precision",
        "true", "false", "null", "undefined", "var", "let", "function",
        "return", "if", "else", "for", "while", "break", "continue",
        "process", "input", "output", "control", "generic", "unknown",
        "node", "step", "layer", "type", "data", "value", "string",
    }

    def __init__(self, rag_library=None):
        self.ollama_url = OLLAMA_URL
        self.cloud_mode = False  # Semantic reasoner always uses local Ollama
        self.model = MODEL_NAME_REASONING
        self.fallback_model = MODEL_NAME_FALLBACK
        self.max_retries = 3
        self.rag = rag_library
        self._ollama_healthy = None  # cached health state

    def _check_ollama_health(self) -> bool:
        """Quick health check — verify Ollama is responsive before expensive calls.
        In cloud mode, always returns True (health checked at call time)."""
        if self.cloud_mode:
            return True
        try:
            # Use the tags endpoint (lightweight, no model loading)
            base_url = self.ollama_url.rsplit('/api/', 1)[0]
            resp = requests.get(f"{base_url}/api/tags", timeout=10)
            if resp.status_code != 200:
                print("SemanticReasoner - Ollama health check failed: not responding")
                return False
            # Check if our model is available
            models = resp.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            if not any(self.model in n for n in model_names):
                print(f"SemanticReasoner - Model '{self.model}' not found in Ollama. Available: {model_names}")
                # Model might still work (could be pulling), don't block
            print("SemanticReasoner - Ollama health check passed")
            return True
        except requests.exceptions.ConnectionError:
            print("SemanticReasoner - Ollama is not running (connection refused)")
            return False
        except Exception as e:
            print(f"SemanticReasoner - Health check error: {e}")
            return False

    def _call_cloud_llm(self, model_name: str, prompt: str, temperature: float,
                        top_p: float, presence_penalty: float) -> str:
        """Call cloud LLM via litellm OpenAI-compatible API."""
        response = litellm.completion(
            model=f"openai/{model_name}",
            messages=[{"role": "user", "content": prompt}],
            api_base=CLOUD_API_BASE,
            api_key=CLOUD_API_KEY,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            max_tokens=2000,
            stream=False,
            timeout=120,
        )
        return response.choices[0].message.content or ""

    @staticmethod
    def _softmax(logits: list) -> list:
        """Standard softmax over a list of floats."""
        max_l = max(logits)
        exps = [math.exp(x - max_l) for x in logits]
        total = sum(exps)
        return [e / total for e in exps]

    def _compute_divergence_params(self, level_weights: dict,
                                   divergence_values: dict = None) -> dict:
        """Level 1 — compute LLM sampling params via softmax over D values.

        Algorithm:
          1. D_agg = weighted mean of D_surface/D_flow/D_narrative
             using level_weights as importance (falls back to simple mean).
          2. Softmax over 3 regime centers [0.0, 0.5, 1.0] with sharpness β=6
             gives weights [w_conv, w_mid, w_div].
          3. Each param is the dot product of regime weights and regime anchors:
               T    : [0.2,  0.70, 1.30]  — greedy → balanced → experimental
               top_p: [0.70, 0.88, 0.98]  — restricted → moderate → open
               pp   : [0.0,  0.50, 1.20]  — none → moderate → strong diversity push
        """
        # --- Step 1: aggregate D ---
        if divergence_values:
            D_s = float(divergence_values.get('D_surface',  divergence_values.get('surface',  0.5)))
            D_f = float(divergence_values.get('D_flow',     divergence_values.get('flow',     0.5)))
            D_n = float(divergence_values.get('D_narrative',divergence_values.get('narrative',0.5)))
            w_s = float(level_weights.get('surface',   0.333))
            w_f = float(level_weights.get('flow',      0.333))
            w_n = float(level_weights.get('narrative', 0.333))
            w_total = w_s + w_f + w_n or 1.0
            D_agg = (D_s * w_s + D_f * w_f + D_n * w_n) / w_total
        else:
            # Fallback: estimate D_agg from level_weights directly
            # High narrative weight → more converged; high flow → more divergent
            N = float(level_weights.get('narrative', 0.333))
            F = float(level_weights.get('flow',      0.333))
            D_agg = 0.3 + 0.5 * F + 0.2 * (1.0 - N)

        D_agg = max(0.0, min(1.0, D_agg))

        # --- Step 2: softmax over regime centers ---
        CENTERS = [0.0, 0.5, 1.0]
        BETA = 6.0  # sharpness — higher = harder transitions between regimes
        logits = [-BETA * (D_agg - c) ** 2 for c in CENTERS]
        w_conv, w_mid, w_div = self._softmax(logits)

        # --- Step 3: weighted sum of regime anchors ---
        T    = w_conv * 0.20 + w_mid * 0.70 + w_div * 1.30
        p    = w_conv * 0.70 + w_mid * 0.88 + w_div * 0.98
        pp   = w_conv * 0.00 + w_mid * 0.50 + w_div * 1.20

        return {
            "temperature":       round(T,  3),
            "top_p":             round(p,  3),
            "presence_penalty":  round(pp, 3),
            "D_agg":             round(D_agg, 3),  # exposed for logging
        }

    def _build_divergence_injection(self, level_weights: dict,
                                    divergence_values: dict = None) -> str:
        """Level 2 — prompt text injected based on D_agg regime."""
        # Reuse _compute_divergence_params to get D_agg (avoids duplicating logic)
        params = self._compute_divergence_params(level_weights, divergence_values)
        D_agg = params["D_agg"]

        if D_agg < 0.35:
            return (
                "CONVERGENCE MODE: Generate a tight, functional node graph. "
                "Prefer well-established rendering patterns and high-impact standard nodes. "
                "Minimize node count — every node must serve a clear purpose."
            )
        elif D_agg > 0.65:
            return (
                "DIVERGENCE MODE: Maximize creative exploration. "
                "Propose unconventional, avant-garde node topologies. "
                "Combine unexpected techniques — do not default to standard linear pipelines."
            )
        else:
            return (
                "BALANCED MODE: Mix aesthetic ambition with technical reliability. "
                "Use some standard nodes as anchors, but include at least one experimental technique."
            )

    def _stream_ollama_response(self, payload: dict, timeout_per_chunk: float = 60.0) -> str:
        """Stream response from Ollama, accumulating tokens. Aborts if no token arrives within timeout_per_chunk seconds."""
        payload["stream"] = True
        accumulated = []
        resp = requests.post(
            self.ollama_url,
            json=payload,
            stream=True,
            timeout=(15, timeout_per_chunk),  # (connect_timeout, read_timeout per chunk)
        )
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}")

        try:
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                token = chunk.get("response", "")
                if token:
                    accumulated.append(token)
                if chunk.get("done", False):
                    break
        except requests.exceptions.ReadTimeout:
            # Stream stalled — use whatever we accumulated so far
            if accumulated:
                partial = "".join(accumulated).strip()
                print(f"SemanticReasoner - Stream stalled after {len(partial)} chars, using partial response")
                return partial
            raise

        return "".join(accumulated).strip()

    def _get_rag_context(self, prompt: str) -> str:
        """Query RAG knowledge graph for relevant technique chunks.
        Strips out session JSON / timestamps that confuse the fine-tuned model."""
        if not self.rag:
            return ""
        try:
            chunks = self.rag.retrieve_chunks(
                f"{prompt} node pipeline", top_k=25
            )
            if not chunks:
                return ""
            lines = []
            for c in chunks:
                text = c.get("text", "")[:300].strip()
                if not text:
                    continue
                # Skip chunks that look like raw session JSON / training data
                if any(marker in text for marker in [
                    '"session_id"', '"timestamp"', '"platform"',
                    '"D_global"', '"user_mode"', 'SESSION CONTEXT',
                    '"brief":', '"seed":'
                ]):
                    continue
                lines.append(f"- {text}")
            return "\n".join(lines)
        except Exception as e:
            print(f"SemanticReasoner - RAG query failed: {e}")
            return ""

    def _format_brand_context(self, brand_context: Dict) -> str:
        """Format brand/emotion context for the prompt."""
        if not brand_context:
            return ""
        lines = []
        if brand_context.get("emotions"):
            top = sorted(brand_context["emotions"].items(), key=lambda x: x[1], reverse=True)[:5]
            lines.append(f"Emotions: {', '.join(f'{e} ({v:.2f})' for e, v in top)}")
        if brand_context.get("attributes"):
            top = sorted(brand_context["attributes"].items(), key=lambda x: x[1], reverse=True)[:5]
            lines.append(f"Attributes: {', '.join(f'{a} ({v:.2f})' for a, v in top)}")
        if brand_context.get("visual_mood"):
            lines.append(f"Visual mood: {brand_context['visual_mood']}")
        if brand_context.get("palette"):
            lines.append(f"Palette: {brand_context['palette']}")
        return "\n".join(lines)

    def _build_system_prompt(self, prompt, num_nodes, design_principles, brand_section, tech_context, toolset, retry_note="", divergence_note=""):
        """Build the LLM prompt for semantic node extraction."""
        return f"""You are a Senior Technical Artist designing a visual node pipeline. You must focus on {design_principles} to create a semantically focused pipeline.
{brand_section}
{tech_context}
AVAILABLE TOOLSET:
{toolset}

ENGINE SELECTION — choose the best engine for each node:
- "p5": Best for shapes, curves, drawing, text, HUD, complex geometry (flowers, trees, mandalas, icons, UI, compositing). Uses p5.js (ellipse, bezier, fill, stroke, push/pop). USE THIS for ALL drawing and shape nodes.
- "glsl": Best for pixel-level effects, noise, blur, glow, feedback, color grading, SDFs, post-processing. The LLM writes GLSL fragment shaders.
- "three_js": Best for 3D geometry, meshes, particles, point clouds.
- "webaudio": Best for audio input, analysis, synthesis, beat detection.

RULE: Use p5 for any node that draws shapes, structures, composites, or animations. Use glsl ONLY for noise generation, blur, glow, color grading, and pixel-level post-effects. NEVER use "canvas2d" — always use "p5" instead.

FORMAT (this is ONLY a structural example — do NOT copy these node names, invent your own based on the brief):
[
  {{"id": "base_structure", "keywords": ["shape", "form", "foundation"], "engine": "p5", "description": "Description of what this node does for the ACTUAL brief"}},
  {{"id": "motion_layer", "keywords": ["animate", "transform", "ease"], "engine": "p5", "description": "Description of the motion this node adds"}},
  {{"id": "post_effect", "keywords": ["post", "glow", "blur"], "engine": "glsl", "description": "Description of the post-processing effect"}},
  {{"id": "final_grade", "keywords": ["color", "grade", "tone"], "engine": "glsl", "description": "Description of the final color treatment"}}
]
IMPORTANT: The example above shows the JSON FORMAT only. You MUST create node IDs and descriptions specific to the brief below. Do NOT reuse "base_structure", "motion_layer", "post_effect", or "final_grade".

{divergence_note}
{retry_note}
TASK: Design a {num_nodes}-step visual pipeline for: "{prompt}"

RULES:
- EVERY node must directly relate to the brief "{prompt}". If the brief says "flower", include flower/petal shapes. If it says "particle", include particle nodes. Do NOT generate generic nodes that ignore the brief.
- Do NOT add webcam, audio, video, FFT, 3D, particle, or tracking nodes UNLESS the brief specifically asks for them. A "flower blooming" brief needs shapes, animation, glow — NOT a webcam or audio analyser.
- If the brief implies specific aesthetics (neon, glow, pastel), include nodes that achieve that.
- You MUST generate exactly {num_nodes} nodes. Do not simplify.
- EVERY node id MUST be UNIQUE. NEVER repeat the same node id.
- id: snake_case step name (e.g. fbm_noise, tone_map, bloom_pass). NEVER use GLSL keywords (float, int, vec2, void).
- keywords: 3-5 technical terms describing the node.
- engine: REQUIRED. One of "p5", "glsl", "three_js", "webaudio", "html_video". Use p5 for shapes/drawing/compositing, glsl for effects/noise/color. NEVER use "canvas2d".
- description: 1 sentence explaining what this node does FOR THIS SPECIFIC BRIEF. NOT generic — must reference the actual visual intent.
- At least 5 nodes. End with output/composite node.
- Output ONLY a JSON array, nothing else:
[{{"id": "example_node", "keywords": ["tag1", "tag2", "tag3"], "engine": "p5", "description": "What this node does for the brief"}}]"""

    def _try_model(self, model_name, system_prompt,
                   max_attempts, timeout, best_nodes, last_error, min_nodes=4,
                   temperature=0.4, top_p=0.95, presence_penalty=0.0):
        """Try a specific model with retries. Returns nodes list or None."""
        for attempt in range(1, max_attempts + 1):
            retry_note = f"\nPREVIOUS ATTEMPT FAILED: {last_error}. Fix the output and try again.\n" if (attempt > 1 and last_error) else ""
            if retry_note:
                current_prompt = system_prompt + retry_note
            else:
                current_prompt = system_prompt

            # On retry: bump temperature for diversity (model may be stuck)
            attempt_temp = temperature if attempt == 1 else min(temperature + 0.3, 1.5)

            try:
                if self.cloud_mode:
                    print(f"SemanticReasoner - [{model_name}] Attempt {attempt}/{max_attempts} (cloud, T={attempt_temp:.2f}, top_p={top_p:.2f}, pp={presence_penalty:.2f})...")
                    llm_output = self._call_cloud_llm(
                        model_name, current_prompt,
                        temperature=attempt_temp, top_p=top_p,
                        presence_penalty=presence_penalty,
                    )
                else:
                    print(f"SemanticReasoner - [{model_name}] Attempt {attempt}/{max_attempts} (streaming, T={attempt_temp:.2f})...")
                    llm_output = self._stream_ollama_response(
                        {
                            "model": model_name,
                            "prompt": current_prompt,
                            "options": {
                                "temperature": attempt_temp,
                                "num_predict": 2000,
                            }
                        },
                        timeout_per_chunk=timeout,
                    )

                if not llm_output.startswith('[') and not llm_output.startswith('{'):
                    llm_output = '[' + llm_output

                print(f"SemanticReasoner - [{model_name}] Raw: {llm_output[:300]}{'...' if len(llm_output) > 300 else ''}")

                semantic_nodes = self._parse_semantic_nodes(llm_output)

                if len(semantic_nodes) >= min_nodes:
                    print(f"SemanticReasoner - [{model_name}] Semantic Nodes ({len(semantic_nodes)}):")
                    for node in semantic_nodes:
                        print(f"  - {node['id']}: {node['keywords']}")
                    return semantic_nodes

                last_error = f"Only {len(semantic_nodes)} unique nodes (need {min_nodes}+)"
                print(f"SemanticReasoner - [{model_name}] {last_error}")
                if len(semantic_nodes) > len(best_nodes):
                    best_nodes = semantic_nodes

            except Exception as e:
                last_error = str(e)
                print(f"SemanticReasoner - [{model_name}] Error (attempt {attempt}): {e}")
                if "ConnectionError" in type(e).__name__ or "ConnectionRefused" in str(e):
                    print(f"SemanticReasoner - [{model_name}] Server unreachable, skipping")
                    return None  # Don't try more models if server is down
                if "ReadTimeout" in type(e).__name__ or "timed out" in str(e).lower():
                    print(f"SemanticReasoner - [{model_name}] Timed out, trying next option")
                    continue

        return best_nodes if len(best_nodes) >= min_nodes else None

    def extract_semantic_nodes(self, prompt: str, num_nodes: int, brand_context: Dict = None,
                               level_weights: Dict = None,
                               divergence_values: Dict = None) -> List[Dict]:
        """
        Generate SEMANTIC node descriptions with keywords for the visual effect.

        Returns a list of semantic node objects:
        [
            {"id": "particles_fluid_sim", "keywords": ["fluid", "organic", "smoke", "flow"]},
            {"id": "neon_color_grade", "keywords": ["neon", "glow", "vibrant"]},
        ]
        """
        print(f"\nSemanticReasoner - Building semantic node stack for: '{prompt}'")

        # Health check — skip retries if Ollama is unreachable
        if not self._check_ollama_health():
            print("SemanticReasoner - Ollama unreachable, skipping LLM and using keyword fallback")
            return self._generate_fallback_nodes(prompt, num_nodes)

        # Gather all available context
        rag_context = self._get_rag_context(prompt)
        brand_str = self._format_brand_context(brand_context)

        # Build context sections
        tech_context = ""
        if rag_context:
            tech_context = f"\nTECHNICAL CONTEXT (Best Practices):\n{rag_context}\n"

        brand_section = ""
        if brand_str:
            brand_section = f"\nBRAND CONTEXT:\n{brand_str}\n"

        last_error = ""
        best_nodes = []

        design_principles = """
        Core Rule:
        First parse the visual intent and structure from the brief, then pick words which can directly relate to a node type for example - circle -> sdf_circle, particle system -> particle_system.
        Always reason from visual intent and structure first.

        1. STRUCTURE FIRST — Begin with intentional form (shape, layout, or pattern). Never start from pure texture or randomness.

        2. HIERARCHY — Establish a clear focal point and visual priority before adding detail.

        3. DEPTH — Plan spatial relationships (layering, scale, occlusion, parallax) before selecting techniques.

        4. MOTION — Define what moves and why. Motion must support the concept; static-only designs are discouraged unless requested.

        5. RHYTHM — Use repetition and timing intentionally to create flow and visual cadence.

        6. CONTRAST — Use value, color, scale, or motion contrast to improve clarity and impact.

        7. COMPOSITION — Think in layers. Combine elements intentionally (masking, blending, transitions).

        8. TRANSFORMATION — Build complexity through distortion, modulation, or transformation of structure — not random decoration.

        9. COHESION — Ensure all elements feel part of the same visual system (consistent color, motion logic, and form language).

        10. SEPARATION OF CONCERNS — Keep structure, motion, composition, and polish as distinct stages.

        11. POLISH — Apply finishing effects (lighting, glow, glitch, grading, texture) deliberately and as separate refinement steps.

        12. CLARITY OVER COMPLEXITY — Prefer simple, readable graphs that clearly reflect design intent.
        """

        # 2. DEFINITIVE TOOLSET (The Menu)
        # Engines + core categories. Specialized sections (3D, audio, video)
        # are only shown contextually to avoid the LLM adding irrelevant nodes.
        toolset = """

        [ENGINES - choose one per node]
        - "p5"       : For drawing shapes, curves, geometry, text, UI, compositing. Uses p5.js (ellipse, bezier, fill, stroke). USE THIS for all drawing.
        - "glsl"     : For pixel shaders, noise, blur, glow, feedback, color grading, SDFs, post-processing.
        - "three_js" : For 3D geometry, meshes, particles. ONLY if brief mentions 3D.
        - "webaudio" : For audio analysis. ONLY if brief mentions audio/music.
        - "html_video" : For video/webcam. ONLY if brief mentions video/camera.

        [DRAWING & SHAPES] (engine: p5)
        - Shapes: ellipse, rect, triangle, quad, arc, bezier, curve
        - Organic: flowers, petals, leaves, trees, spirals, mandalas
        - Motion: animated drawing, easing, scale/rotate transforms
        - Compositing: layering, blending, alpha, output compositing

        [GENERATORS] (engine: glsl)
        - Noise: noise_perlin, noise_simplex, noise_worley, noise_fbm
        - Patterns: gradient_linear, gradient_radial, checker_pattern
        - SDF: sdf_circle, sdf_box, sdf_triangle

        [POST-EFFECTS] (engine: glsl)
        - Blur/Glow: blur_gaussian, blur_radial, glow, bloom, vignette
        - Color: color_grade, hue_shift, saturation, chromatic_aberration
        - Glitch: glitch_rgb, glitch_scanline, displacement_map
        - Stylize: posterize, quantize, dither, halftone

        [TRANSFORMS] (engine: glsl)
        - Space: kaleidoscope, mirror, tile, zoom, distort_wave, feedback_loop
        - Blend: blend_normal, blend_add, blend_multiply
        """

        # --- Divergence math (Level 1 + 2) ---
        weights = level_weights if level_weights else DEFAULT_LEVEL_WEIGHTS
        div_params = self._compute_divergence_params(weights, divergence_values)
        divergence_note = self._build_divergence_injection(weights, divergence_values)
        print(f"SemanticReasoner - Divergence params: T={div_params['temperature']}, "
              f"top_p={div_params['top_p']}, pp={div_params['presence_penalty']}, "
              f"D_agg={div_params['D_agg']}")
        if divergence_note:
            print(f"SemanticReasoner - Divergence injection: {divergence_note[:80]}...")
        # Sampling params only — D_agg is for logging, not passed to the model call
        sampling_params = {k: v for k, v in div_params.items() if k != 'D_agg'}

        system_prompt = self._build_system_prompt(
            prompt, num_nodes, design_principles, brand_section, tech_context, toolset,
            divergence_note=divergence_note,
        )

        # Minimum unique nodes required — don't accept a handful of generic repeats
        min_nodes = max(num_nodes // 2, 4)

        # Try models in order: primary → fallback → fallback again with lower threshold
        models_to_try = [
            (self.model, 2, 90.0),           # primary: 2 attempts, 90s timeout
            (self.fallback_model, 2, 90.0),   # fallback: 2 attempts, 90s timeout
        ]

        for model_name, max_attempts, timeout in models_to_try:
            result = self._try_model(
                model_name, system_prompt,
                max_attempts, timeout, best_nodes, last_error,
                min_nodes=min_nodes,
                **sampling_params,
            )
            if result is not None:
                if isinstance(result, list) and len(result) >= min_nodes:
                    return result
                if isinstance(result, list) and len(result) > len(best_nodes):
                    best_nodes = result

        # Last resort LLM attempt: retry fallback with relaxed threshold (accept 3+)
        if len(best_nodes) < 3:
            print(f"SemanticReasoner - Final retry with {self.fallback_model} (relaxed threshold)")
            result = self._try_model(
                self.fallback_model, system_prompt,
                1, 120.0, best_nodes, last_error,
                min_nodes=3,
                **sampling_params,
            )
            if result is not None and isinstance(result, list) and len(result) >= 3:
                return result
            if isinstance(result, list) and len(result) > len(best_nodes):
                best_nodes = result

        if best_nodes:
            print(f"SemanticReasoner - Returning best result ({len(best_nodes)} nodes):")
            for node in best_nodes:
                print(f"  - {node['id']}: {node['keywords']}")
            return best_nodes

        print("SemanticReasoner - All LLMs failed, generating fallback nodes from prompt keywords")
        return self._generate_fallback_nodes(prompt, num_nodes)

    def _generate_fallback_nodes(self, prompt: str, num_nodes: int) -> List[Dict]:
        """Generate fallback semantic nodes purely from the prompt — no hardcoded mappings.

        Extracts meaningful words from the brief and turns each into a node.
        'flower blooming' → flower_shape, blooming_anim, color_pass, composite_output
        """
        stop_words = {
            'with', 'from', 'that', 'this', 'have', 'create', 'make', 'like',
            'very', 'some', 'want', 'need', 'should', 'would', 'could', 'about',
            'just', 'into', 'over', 'under', 'through', 'between', 'using',
            'based', 'will', 'also', 'each', 'then', 'when', 'more', 'than',
            'both', 'after', 'before', 'been', 'being', 'does', 'done',
        }
        words = prompt.lower().replace(',', ' ').replace('.', ' ').replace('-', ' ').split()
        keywords = [w for w in words if len(w) > 2 and w not in stop_words]

        nodes = []
        seen = set()

        # Each meaningful word from the brief becomes a node
        brief_short = prompt[:60]
        for kw in keywords:
            if len(nodes) >= num_nodes - 2:
                break
            node_id = f"{kw}_shape"
            if node_id not in seen:
                nodes.append({
                    'id': node_id,
                    'keywords': [kw, 'visual', 'shader'],
                    'description': f"Generate {kw} visual element for: {brief_short}",
                })
                seen.add(node_id)

        # Add animation + output
        if len(nodes) < num_nodes - 1:
            nodes.append({
                'id': 'motion_anim',
                'keywords': ['motion', 'animate', 'time'],
                'description': f"Animate the visual elements with motion for: {brief_short}",
            })
        nodes.append({
            'id': 'composite_output',
            'keywords': ['composite', 'output', 'blend'],
            'description': f"Composite all layers into final output for: {brief_short}",
        })

        print(f"SemanticReasoner - Fallback: {len(nodes)} nodes from prompt words: {keywords[:6]}")
        return nodes[:num_nodes]

    def _parse_semantic_nodes(self, llm_output: str) -> List[Dict]:
        """Parse semantic node objects from LLM output."""
        cleaned = self._clean_text(llm_output)

        parsed = None
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        if isinstance(parsed, dict):
            # Try unwrapping {"nodes": [...]} style wrappers first
            unwrapped = False
            for key in ["nodes", "node_archetypes", "archetypes", "pipeline"]:
                if key in parsed and isinstance(parsed[key], list):
                    parsed = parsed[key]
                    unwrapped = True
                    break
            # If it's a single node dict (has "id" or "name"), wrap it as a list
            if not unwrapped and ("id" in parsed or "name" in parsed):
                parsed = [parsed]

        # Handle [{"nodes": [...]}] — list wrapping a single dict with a nodes key
        if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict):
            inner = parsed[0]
            for key in ["nodes", "node_archetypes", "archetypes", "pipeline"]:
                if key in inner and isinstance(inner[key], list):
                    parsed = inner[key]
                    break

        if not isinstance(parsed, list):
            array_text = self._extract_array_json(cleaned)
            if array_text:
                try:
                    parsed = json.loads(array_text)
                except json.JSONDecodeError:
                    parsed = self._repair_semantic_json(array_text)

        if not parsed or not isinstance(parsed, list):
            parsed = self._repair_semantic_json(cleaned)
            if not parsed:
                return []

        nodes = []
        for item in parsed:
            if not isinstance(item, dict):
                continue

            node_id = (
                item.get("id", "") or
                item.get("name", "") or
                item.get("type", "") or
                item.get("category", "")
            )
            if isinstance(node_id, str):
                node_id = node_id.strip()
            else:
                node_id = ""

            if node_id and re.match(r'^n\d+$', node_id):
                node_id = item.get("type", "") or item.get("category", "") or node_id
                if isinstance(node_id, str):
                    node_id = node_id.strip()

            if not node_id:
                continue

            # Filter out GLSL/JS keywords that are not valid pipeline step names
            if node_id.lower() in self.GARBAGE_IDS:
                print(f"SemanticReasoner - Filtered garbage ID: '{node_id}'")
                continue

            keywords = item.get("keywords", [])

            if not keywords and isinstance(item.get("meta"), dict):
                keywords = item["meta"].get("keywords", [])

            if not keywords:
                kw_sources = [
                    item.get("type", ""),
                    item.get("category", ""),
                    item.get("role", ""),
                ]
                for src in kw_sources:
                    if src:
                        keywords.extend(src.replace("_", " ").split())

            if isinstance(keywords, str):
                keywords = [k.strip() for k in keywords.split(",")]
            elif isinstance(keywords, list):
                keywords = [str(k).strip().lower() for k in keywords if k]

            if not keywords:
                keywords = [k.lower() for k in node_id.split("_") if len(k) > 2]

            node_entry = {
                "id": node_id,
                "keywords": keywords[:8],
            }
            # Preserve description and engine from LLM output
            desc = item.get("description", "")
            if desc and isinstance(desc, str):
                node_entry["description"] = desc.strip()
            engine = item.get("engine", "")
            if engine and isinstance(engine, str):
                node_entry["engine"] = engine.strip().lower()
            nodes.append(node_entry)

        # Deduplicate: remove nodes with duplicate IDs (keep first occurrence)
        seen_ids = set()
        deduped = []
        for node in nodes:
            nid = node["id"]
            if nid not in seen_ids:
                seen_ids.add(nid)
                deduped.append(node)
            else:
                print(f"SemanticReasoner - Removed duplicate node: '{nid}'")
        if len(deduped) < len(nodes):
            print(f"SemanticReasoner - Dedup: {len(nodes)} -> {len(deduped)} nodes")
        return deduped

    def _repair_semantic_json(self, s: str) -> Optional[List[Dict]]:
        """Repair truncated semantic node JSON."""
        nodes = []

        pattern1 = r'\{\s*"id"\s*:\s*"([^"]+)"\s*,\s*"keywords"\s*:\s*\[([^\]]*)\]'
        for m in re.finditer(pattern1, s):
            node_id = m.group(1).strip()
            keywords_str = m.group(2)
            keywords = []
            for km in re.finditer(r'"([^"]+)"', keywords_str):
                keywords.append(km.group(1).strip().lower())
            if node_id and keywords:
                if not re.match(r'^n\d+$', node_id) and node_id.lower() not in self.GARBAGE_IDS:
                    nodes.append({"id": node_id, "keywords": keywords})

        if not nodes:
            pattern2 = r'"type"\s*:\s*"([^"]+)"'
            types_found = re.findall(pattern2, s)
            for t in types_found:
                t = t.strip()
                if t and t.lower() not in self.GARBAGE_IDS:
                    keywords = [k.lower() for k in t.split("_") if len(k) > 2]
                    if keywords:
                        nodes.append({"id": t, "keywords": keywords})

        if not nodes:
            pattern3 = r'"category"\s*:\s*"([^"]+)"'
            cats_found = re.findall(pattern3, s)
            for c in cats_found:
                c = c.strip()
                if c and c.lower() not in self.GARBAGE_IDS:
                    keywords = [k.lower() for k in c.split("_") if len(k) > 2]
                    if keywords:
                        nodes.append({"id": c, "keywords": keywords})

        seen = set()
        unique_nodes = []
        for n in nodes:
            if n["id"] not in seen:
                seen.add(n["id"])
                unique_nodes.append(n)

        return unique_nodes if unique_nodes else None

    def extract_semantic_categories(self, prompt: str, num_categories: int, brand_context: Dict = None) -> List[str]:
        """LEGACY METHOD - Returns flat list of category strings for backward compatibility."""
        semantic_nodes = self.extract_semantic_nodes(prompt, num_categories, brand_context)
        return [node["id"] for node in semantic_nodes]

    def _clean_text(self, s: str) -> str:
        """Clean LLM output - remove thinking tags and markdown fences."""
        if not s:
            return ""
        s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL).strip()
        if "```" in s:
            m = re.search(r"```json\s*([\s\S]*?)```", s, re.IGNORECASE)
            if m:
                s = m.group(1).strip()
            else:
                m = re.search(r"```\s*([\s\S]*?)```", s)
                if m:
                    s = m.group(1).strip()
        return s.strip()

    def _extract_array_json(self, s: str) -> str:
        """Extract JSON array [...] from text."""
        if not s:
            return ""
        start = s.find("[")
        end = s.rfind("]")
        if start == -1 or end == -1 or end <= start:
            return ""
        return s[start:end + 1].strip()
