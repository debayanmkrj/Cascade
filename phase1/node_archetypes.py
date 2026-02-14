"""Node Archetype Generator - Phase 2 (no hard node-count limiter)

Changes:
- Removes MIN_NODE_COUNT / MAX_NODE_COUNT clamping.
- Uses creative_levels to *estimate* desired node count, but does not force/truncate
  the semantic reasoner's output to an exact number.
- Keeps backward compatible NodeTensor output structure.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List, Dict
import random
import math

from phase2.data_types import NodeTensor
from config import MIN_NODE_COUNT, MAX_NODE_COUNT  # kept for compatibility (unused)
from phase1.semantic_reasoner import SemanticReasoner


# ---------------------------------------------------------------------------
# Note: Parameters are extracted from generated code by Mason
# ---------------------------------------------------------------------------
# Mason generates code with specific uniforms/parameters. The parameters
# are extracted from the generated code rather than using a hardcoded registry.
# This allows semantic nodes with custom IDs to get appropriate parameters.
# ---------------------------------------------------------------------------


class NodeArchetypeGenerator:
    """Generate node archetypes based on semantic categories + creative levels."""

    def __init__(self, rag_library=None):
        self.semantic_reasoner = SemanticReasoner(rag_library=rag_library)

    def _safe_float(self, v, default=0.0) -> float:
        try:
            f = float(v)
            if math.isnan(f) or math.isinf(f):
                return default
            return f
        except Exception:
            return default

    def _compute_node_count(self, D_narrative: float, creative_levels: Dict) -> int:
        """
        Estimate an appropriate node count.

        Capped at 12 to stay within LLM quality limits.
        Too many nodes = LLM produces garbage IDs.
        The filler mechanism can pad if needed.
        """
        surface = self._safe_float(creative_levels.get("surface", creative_levels.get("visual", 0.4)), 0.4)
        flow = self._safe_float(creative_levels.get("flow", creative_levels.get("motion", 0.4)), 0.4)
        narrative = self._safe_float(creative_levels.get("narrative", creative_levels.get("visceral", 0.4)), 0.4)
        D = self._safe_float(D_narrative, 0.5)

        # 2. Aggressive Multipliers
        # Previous Max potential: 17
        # New Max potential: 5 + 6 + 6 + 5 + 5 = 27 (Clamped to 25)
        
        base = 5.0
        base += 6.0 * surface    # Was 3.0 (Focus on Visuals)
        base += 6.0 * flow       # Was 4.0 (Focus on Motion)
        base += 5.0 * narrative  # Was 3.0
        base += 5.0 * D          # Was 2.0

        # Soft jitter (Randomness)
        base += random.uniform(-1.0, 2.0)

        # 3. Hard Clamp
        # We cap at 30 to prevent the LLM from timing out
        return max(5, min(30, int(round(base))))

    def _determine_creative_level(self, creative_levels: Dict) -> str:
        """Choose a descriptive creative level name based on weights."""
        if not creative_levels:
            return "medium"
        # pick highest key
        best = max(creative_levels.items(), key=lambda x: x[1])[0]
        return str(best)

    def _keyword_fallback_categories(self, essence: str, target_count: int) -> List[str]:
        """Generate categories from keywords in the essence when LLM fails."""
        essence_lower = essence.lower()
        categories = []

        # Keyword -> category mappings
        KEYWORD_MAP = {
            # Noise/Pattern
            "noise": ["noise_generator", "perlin_noise"],
            "perlin": ["perlin_noise"],
            "fractal": ["fractal_noise"],
            "pattern": ["pattern_generator"],
            "grid": ["prim_pattern_grid", "pattern_generator"],
            "stripe": ["prim_pattern_stripes"],
            "checker": ["prim_pattern_checker"],

            # Geometry
            "circle": ["prim_sdf_circle", "shape_generator"],
            "tunnel": ["prim_transform_polar", "displacement"],
            "spiral": ["prim_transform_polar", "oscillator"],
            "wave": ["oscillator", "prim_pattern_wave"],
            "polar": ["prim_transform_polar"],

            # Effects
            "blur": ["blur"],
            "glow": ["glow", "bloom"],
            "bloom": ["bloom"],
            "neon": ["glow", "bloom", "color_grade"],
            "glitch": ["glitch", "chromatic_aberration"],
            "scanline": ["glitch", "prim_pattern_stripes"],
            "distort": ["distortion", "displacement"],
            "chromatic": ["chromatic_aberration"],
            "feedback": ["feedback_effect"],
            "kaleidoscope": ["kaleidoscope"],

            # Motion
            "particle": ["particle_system", "particle_emitter"],
            "flow": ["prim_flow_field", "field_generator"],
            "moving": ["oscillator", "displacement"],
            "animate": ["oscillator"],

            # Color
            "color": ["color_grade"],
            "gradient": ["prim_gradient", "color_grade"],
            "hue": ["color_grade"],
            "saturation": ["color_grade"],
            "retro": ["color_grade", "vignette"],
            "synthwave": ["glow", "color_grade", "prim_pattern_grid"],
            "80s": ["glow", "color_grade", "scanline"],
            "cyberpunk": ["glow", "glitch", "chromatic_aberration"],

            # Natural
            "sun": ["prim_sdf_circle", "glow", "bloom"],
            "fog": ["fog"],
            "fire": ["fire", "particle_system"],
            "water": ["water", "displacement"],
            "cloud": ["perlin_noise", "fog"],
            "smoke": ["particle_system", "blur"],

            # Input/Output
            "video": ["video_input"],
            "webcam": ["webcam_input"],
            "audio": ["audio_input", "fft_analyzer"],
            "music": ["audio_input", "beat_detector"],
        }

        # Find matching categories
        for keyword, cats in KEYWORD_MAP.items():
            if keyword in essence_lower:
                for cat in cats:
                    if cat not in categories:
                        categories.append(cat)

        # Ensure minimum diversity
        if len(categories) < 3:
            categories.extend(["noise_generator", "color_grade", "sdf_shape", "particle_system"])

        # Always end with output
        if "composite_output" not in categories:
            categories.append("composite_output")

        # Limit to target count but ensure at least 6
        max_cats = max(target_count, 6)
        return categories[:max_cats]

    def generate_node_archetypes(self,
                                brand_essence: str,
                                brand_values: Dict,
                                divergence,
                                creative_levels: Dict,
                                visual_palette=None) -> List[NodeTensor]:
        """
        Generate node archetypes (NodeTensor objects).

        Unlike the earlier version, we do NOT truncate semantic categories
        to an exact node count. We ask for an approximate count, ensure a minimal
        scaffold, and then produce one NodeTensor per category.

        Args:
            visual_palette: Image-derived visual features (shapes, motion_words, etc.)
                           Has equal weightage as text description for node selection.
        """
        # Extract D_narrative from divergence object
        D_narrative = getattr(divergence, 'D_narrative', 0.5)
        target_count = self._compute_node_count(D_narrative, creative_levels)

        # Combine brand values with visual palette for comprehensive context
        enhanced_context = dict(brand_values) if brand_values else {}
        if visual_palette:
            # Add visual features from images (equal weightage as text)
            enhanced_context['image_shapes'] = getattr(visual_palette, 'shapes', [])
            enhanced_context['image_motion'] = getattr(visual_palette, 'motion_words', [])
            enhanced_context['image_colors'] = getattr(visual_palette, 'primary_colors', [])

        # Use extract_semantic_nodes to get BOTH category IDs AND keywords
        semantic_nodes = self.semantic_reasoner.extract_semantic_nodes(
            brand_essence,
            target_count,
            enhanced_context  # Pass brand context + visual features
        )

        # If semantic reasoner completely failed, use keyword-based fallback
        if len(semantic_nodes) == 0:
            print(f"  [NODE ARCH] SemanticReasoner failed, using keyword fallback for {target_count} nodes")
            fallback_cats = self._keyword_fallback_categories(brand_essence, target_count)
            # Convert to semantic node format
            semantic_nodes = [{"id": cat, "keywords": cat.split("_")} for cat in fallback_cats]

        # Ensure a minimal scaffold (no truncation)
        # Only add essential nodes if LLM returned too few
        existing_ids = set(n["id"] for n in semantic_nodes)
        if len(semantic_nodes) < 4:
            if "composite_output" not in existing_ids and "final_composite" not in existing_ids:
                semantic_nodes.append({"id": "composite_output", "keywords": ["composite", "output", "blend"]})
            if len(semantic_nodes) < 4 and "color_grade" not in existing_ids:
                semantic_nodes.append({"id": "color_grade", "keywords": ["color", "grade", "saturation"]})
            if len(semantic_nodes) < 4 and "noise_generator" not in existing_ids:
                semantic_nodes.append({"id": "noise_generator", "keywords": ["noise", "perlin", "texture"]})

        # If still fewer than target, pad with generic fillers (capped at target_count)
        min_nodes = min(target_count, max(6, len(semantic_nodes) + 2))
        filler_pool = [
            {"id": "sdf_shape", "keywords": ["geometry", "shape", "sdf"]},       # <--- Added
            {"id": "grid_pattern", "keywords": ["grid", "structure", "lines"]},   # <--- Added
            {"id": "kaleidoscope", "keywords": ["mirror", "symmetry", "fractal"]},# <--- Added
            {"id": "noise_generator", "keywords": ["noise", "texture"]},
            {"id": "color_grade", "keywords": ["color", "grade"]},
        ]
        existing_ids = set(n["id"] for n in semantic_nodes)
        for filler in filler_pool:
            if filler["id"] not in existing_ids and len(semantic_nodes) < min_nodes:
                semantic_nodes.append(filler)

        creative_level = self._determine_creative_level(creative_levels)

        nodes: List[NodeTensor] = []
        used_ids = set()

        for idx, sem_node in enumerate(semantic_nodes):
            # Use short IDs to avoid GLSL uniform name length limits
            stable_id = f"n{idx}"
            used_ids.add(stable_id)

            # Extract category and keywords from semantic node
            category = sem_node.get("id", "unknown")
            keywords = sem_node.get("keywords", [])
            assigned_engine = sem_node.get("engine", "glsl").lower()

            # Parameters will be populated by Mason from generated code
            params = {}

            node = NodeTensor(
                id=stable_id,
                meta={
                    "category": category,
                    "label": category.replace("_", " ").title(),
                    "description": f"{category.replace('_', ' ')} node",
                    "creative_level": creative_level,
                    "engine_hint": assigned_engine,
                    "keywords": keywords,  # Store keywords in meta too
                },
                engine=assigned_engine,
                code_snippet="",
                parameters=params,
                input_nodes=[],
                output_texture={"width": 512, "height": 512, "format": "rgba8"},
                grid_position=[idx % 4, idx // 4, 0],  # placeholder; architect assigns z
                grid_size=[1, 1],
                keywords=keywords,  # Pass keywords to NodeTensor
            )
            nodes.append(node)

        return nodes
