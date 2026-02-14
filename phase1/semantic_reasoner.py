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
import re
import requests
from typing import List, Dict, Optional

from config import OLLAMA_URL, MODEL_NAME_REASONING, MODEL_NAME_FALLBACK


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
        # mason-architect:latest — fine-tuned specifically for node planning
        self.model = MODEL_NAME_REASONING
        self.fallback_model = MODEL_NAME_FALLBACK
        self.max_retries = 3
        self.rag = rag_library
        self._ollama_healthy = None  # cached health state

    def _check_ollama_health(self) -> bool:
        """Quick health check — verify Ollama is responsive before expensive calls."""
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

    def extract_semantic_nodes(self, prompt: str, num_nodes: int, brand_context: Dict = None) -> List[Dict]:
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
        # We inject your exact training categories so the planner knows what exists.
        toolset = """

        [AVAILABLE ENGINES - YOU MUST CHOOSE ONE FOR EACH NODE]
        - "canvas2d" : Use for algorithmic drawing, loops, curves, UI (flowers, trees, mandalas, hud).
        - "three_js" : Use for 3D geometry and emitters (particles, meshes, point_clouds).
        - "glsl"     : Use for pixel shaders, textures, and post-processing (noise, SDFs, bloom, glitch, color).
        - "webaudio" : Use for audio input, analysis, and synthesis.
        - "html_video" : Use for video input and processing.
        - "events"   : Use for control logic, routing, and event handling.

        [GENERATORS & SDFs]
        - Noise: noise_perlin, noise_simplex, noise_worley, noise_fbm
        - Patterns: gradient_linear, gradient_radial, checker_pattern
        - Shapes (2D SDF): sdf_circle, sdf_box, sdf_triangle

        [PROCESSORS & FX]
        - Glitch: glitch_rgb, glitch_scanline, glitch_noise, displacement_map
        - Stylize: posterize, quantize, dither, halftone, edge_detect, emboss
        - Color: color_grade, hue_shift, saturation, exposure, chromatic_aberration
        - Blur/Glow: blur_gaussian, blur_radial, glow, bloom, vignette

        [TRANSFORMS]
        - Space: kaleidoscope, mirror, tile, zoom, distort_wave, distort_polar
        - Logic: blend_normal, blend_add, blend_multiply, feedback_loop

        [3D / GEOMETRY] (Engine: three_js)
        - Meshes: mesh_sphere, mesh_cube, mesh_plane
        - Particles: particle_system, point_cloud
        - Materials: phong_material, standard_material

        [AUDIO & EVENTS] (Engine: webaudio / events)
        - Sources: oscillator_sine, oscillator_saw
        - Analysis: fft_analyser, beat_detector, amplitude_follower
        - Control: lfo_sine, lfo_square, envelope_adsr, smooth_follower
        """

        for attempt in range(1, self.max_retries + 1):
            retry_note = f"\nPREVIOUS ATTEMPT FAILED: {last_error}. Fix the output and try again.\n" if (attempt > 1 and last_error) else ""

            # Prompt structure: CONTEXT first, then INSTRUCTION last
            # Fine-tuned models latch onto the last thing they see
            system_prompt = f"""You are a Senior Technical Artist designing a visual node pipeline. You must focus on {design_principles} to create a semantically focused pipeline. 
{brand_section}
{tech_context}
AVAILABLE TOOLSET:
{toolset}

VALID NODE TYPES (Examples):
- Shapes (2D): circle_sdf, rect_shape, polygon_draw
- Shapes (3D): sphere_sdf, box_sdf, raymarch_scene
- Patterns: voronoi_noise, grid_pattern, fbm_noise
- Effects: glow_post, chromatic_aberration, dither

FORMAT:
[
  {{"id": "blooming_flower", "keywords": ["petals", "loops", "draw"], "engine": "canvas2d"}},
  {{"id": "glow_bloom", "keywords": ["post", "light", "blur"], "engine": "glsl"}}
]

{retry_note}
TASK: Design a {num_nodes}-step shader pipeline for: "{prompt}"

RULES:
- If the brief implies a specific shape like particle system or a shape like circle then You are strictly BANNED from using other types of nodes that do not directly relate to the shape or structure.
- If the brief implies specific aesthetics like neon color, then include nodes that 
 achieve that.
- If the brief implies an input type (e.g. audio-reactive), include relevant input and analysis nodes.
- You MUST generate exactly {num_nodes} nodes. Do not simplify.
- id: snake_case step name (e.g. fbm_noise, tone_map, bloom_pass). NEVER use GLSL keywords (float, int, vec2, void).
- keywords: 3-5 technical terms describing the node.
- At least 5 nodes. End with output/composite node.
- Output ONLY a JSON array, nothing else:
[{{"id": "example_node", "keywords": ["tag1", "tag2", "tag3"]}}]"""

            try:
                print(f"SemanticReasoner - Attempt {attempt}/{self.max_retries} (streaming)...")
                llm_output = self._stream_ollama_response(
                    {
                        "model": self.model,
                        "prompt": system_prompt,
                        # No "format": "json" — mason-architect is fine-tuned to
                        # output JSON arrays natively. Forcing JSON mode constrains
                        # it to a single object instead of the expected [...] array.
                        "options": {
                            "temperature": 0.3 if attempt == 1 else 0.2,
                            "num_predict": 800,
                        }
                    },
                    timeout_per_chunk=180.0,
                )

                # Ensure output starts with '[' (JSON mode should handle this,
                # but the model may wrap in {"nodes": [...]})
                if not llm_output.startswith('[') and not llm_output.startswith('{'):
                    llm_output = '[' + llm_output

                print(f"SemanticReasoner - Raw output: {llm_output[:300]}{'...' if len(llm_output) > 300 else ''}")

                semantic_nodes = self._parse_semantic_nodes(llm_output)

                if len(semantic_nodes) > len(best_nodes):
                    best_nodes = semantic_nodes

                if len(semantic_nodes) >= 2:
                    print(f"SemanticReasoner - Semantic Nodes ({len(semantic_nodes)}):")
                    for node in semantic_nodes:
                        print(f"  - {node['id']}: {node['keywords']}")
                    return semantic_nodes

                last_error = f"Only {len(semantic_nodes)} nodes extracted (need 2+)"
                print(f"SemanticReasoner - Error: {last_error}")

            except Exception as e:
                last_error = str(e)
                print(f"SemanticReasoner - Error (attempt {attempt}): {e}")
                # If connection refused or server down, don't waste time on more retries
                if "ConnectionError" in type(e).__name__ or "ConnectionRefused" in str(e):
                    print("SemanticReasoner - Server unreachable, skipping remaining retries")
                    break

        if best_nodes:
            print(f"SemanticReasoner - Returning best result ({len(best_nodes)} nodes):")
            for node in best_nodes:
                print(f"  - {node['id']}: {node['keywords']}")
            return best_nodes

        print("SemanticReasoner - Failed after all retries, generating fallback nodes from prompt")
        return self._generate_fallback_nodes(prompt, num_nodes)

    def _generate_fallback_nodes(self, prompt: str, num_nodes: int) -> List[Dict]:
        """Generate fallback semantic nodes from prompt keywords when LLM fails."""
        words = prompt.lower().replace(',', ' ').replace('.', ' ').split()
        keywords = [w for w in words if len(w) > 3 and w not in ('with', 'from', 'that', 'this', 'have', 'create', 'make')]

        nodes = []

        keyword_to_node = {
            'particle': ('particle_system', ['particle', 'emitter', 'flow']),
            'particles': ('particle_system', ['particle', 'emitter', 'flow']),
            'noise': ('noise_generator', ['noise', 'perlin', 'pattern']),
            'perlin': ('noise_generator', ['noise', 'perlin', 'organic']),
            'flow': ('flow_field', ['flow', 'field', 'motion']),
            'fluid': ('fluid_simulation', ['fluid', 'flow', 'organic']),
            'organic': ('organic_motion', ['organic', 'flow', 'natural']),
            'neon': ('neon_glow', ['neon', 'glow', 'bright']),
            'glow': ('bloom_effect', ['glow', 'bloom', 'bright']),
            'color': ('color_grade', ['color', 'grade', 'hue']),
            'colors': ('color_grade', ['color', 'grade', 'saturation']),
            'blur': ('blur_effect', ['blur', 'smooth', 'soft']),
            'distort': ('distortion', ['distort', 'warp', 'wave']),
            'glitch': ('glitch_effect', ['glitch', 'digital', 'noise']),
            'retro': ('retro_effect', ['retro', 'vintage', 'crt']),
            'grid': ('grid_pattern', ['grid', 'lines', 'pattern']),
            'wave': ('wave_distort', ['wave', 'distort', 'motion']),
            'gradient': ('gradient_generator', ['gradient', 'color', 'ramp']),
        }

        seen = set()
        for kw in keywords:
            if kw in keyword_to_node and kw not in seen:
                node_id, node_kws = keyword_to_node[kw]
                nodes.append({'id': node_id, 'keywords': node_kws})
                seen.add(kw)
                if len(nodes) >= num_nodes - 1:
                    break

        nodes.append({'id': 'composite_output', 'keywords': ['composite', 'output', 'blend']})

        generic_nodes = [
            {'id': 'sdf_shape', 'keywords': ['geometry', 'draw', 'shape']}, # <--- Priority 1
            {'id': 'noise_generator', 'keywords': ['noise', 'texture']},
            {'id': 'color_grade', 'keywords': ['color', 'grade']},
        ]
        for gn in generic_nodes:
            if len(nodes) >= num_nodes:
                break
            if gn['id'] not in [n['id'] for n in nodes]:
                nodes.insert(-1, gn)

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

            nodes.append({
                "id": node_id,
                "keywords": keywords[:8]
            })

        return nodes

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
