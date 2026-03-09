"""Architect Agent - Graph Planner + Filter 1 (Logic Validation)

Takes Phase 1 session JSON and produces a VolumetricGrid with NodeTensors.
Uses LLM-driven creative topology design (via CreativeTopologyAgent) validated
against Phase 1 brand data to place ALL archetypes in a 3D volumetric layout.
LLM reasons about narrative flow, creative intent, and aesthetic purpose.
Implements Filter 1: validates graph connectivity, no cycles in data flow,
all inputs satisfied, and Z-layer ordering is correct.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import re
from typing import Dict, List, Tuple, Optional, Any

from config import MODEL_NAME_REASONING
from phase2.aider_llm import get_aider_llm
from phase2.data_types import (
    NodeTensor, NodeMeta, TextureHandle, Connection,
    VolumetricGrid, ArchitectPlan
)
try:
    from phase2.agents.creative_topology import CreativeTopologyAgent
except ImportError:
    from phase2.agents._deprecated.creative_topology import CreativeTopologyAgent
from phase2.agents.dag_layout import DAGLayout

# Engine selection heuristics based on node category
CATEGORY_ENGINE_MAP = {
    # --- Noise / procedural (GLSL is ideal) ---
    "noise_generator": "glsl",
    "perlin_noise": "glsl",
    "fractal_noise": "glsl",
    "oscillator": "glsl",
    "sdf_shape": "glsl",
    "sdf_circle": "glsl",
    "sdf_box": "glsl",
    "raymarch_scene": "glsl",

    # --- Shapes & drawing → canvas2d (ctx.arc, ctx.fillRect, ctx.bezierCurveTo) ---
    "pattern_generator": "canvas2d",
    "field_generator": "canvas2d",
    "circle": "canvas2d",
    "rectangle": "canvas2d",
    "triangle": "canvas2d",
    "polygon": "canvas2d",
    "shape": "canvas2d",
    "shape_generator": "canvas2d",
    "draw": "canvas2d",
    "line": "canvas2d",
    "curve": "canvas2d",
    "path": "canvas2d",
    "star": "canvas2d",
    "spiral": "canvas2d",
    "grid_pattern": "canvas2d",
    "text_renderer": "canvas2d",
    "typography": "canvas2d",

    # --- Gradients → canvas2d (createLinearGradient / createRadialGradient) ---
    "gradient_radial": "canvas2d",
    "gradient_linear": "canvas2d",
    "gradient": "canvas2d",

    # --- Particles & flow → canvas2d (array-based state, easy for LLMs) ---
    "particle_emitter": "canvas2d",
    "particle_system": "canvas2d",
    "particle_renderer": "canvas2d",
    "flow_field": "canvas2d",
    "fluid": "canvas2d",
    "organic_motion": "canvas2d",
    "flocking": "canvas2d",
    "boids": "canvas2d",
    "swarm": "canvas2d",
    "text_overlay": "canvas2d",
    "hud_grid": "canvas2d",

    # --- 3D Geometry → three_js ---
    "sphere": "three_js",
    "cube": "three_js",
    "torus": "three_js",
    "plane_3d": "three_js",
    "wireframe": "three_js",
    "point_cloud": "three_js",
    "mesh_renderer": "three_js",
    "physics": "three_js",
    "collision_detector": "three_js",
    "mesh_cube": "three_js",    # Meshes are 3D!
    "mesh_sphere": "three_js",
    "particle_system": "three_js",

    # --- Forces (GLSL for per-pixel, canvas2d for particle-based) ---
    "gravity": "glsl",
    "gravity_field": "glsl",
    "wind": "glsl",
    "turbulence": "glsl",
    "attraction": "glsl",
    "repulsion": "glsl",

    # --- Post-processing effects → GLSL (what mason:latest is good at) ---
    "blur": "glsl",
    "feedback_effect": "glsl",
    "displacement": "glsl",
    "chromatic_aberration": "glsl",
    "bloom": "glsl",
    "glow": "glsl",
    "distortion": "glsl",
    "glitch": "glsl",
    "kaleidoscope": "glsl",
    "color_ramp": "glsl",
    "motion_detector": "glsl",
    "optical_flow": "glsl",
    "frame_buffer": "glsl",
    "sdf": "glsl",
    "raymarch": "glsl",
    "cloud_renderer": "glsl",
    "rain_renderer": "glsl",
    "fog": "glsl",
    "atmosphere": "glsl",
    "water": "glsl",
    "fire": "glsl",
    "smoke": "glsl",
    "lightning": "glsl",
    "phong_shader": "glsl",

    # --- Audio → webaudio ---
    "audio_input": "webaudio",
    "fft_analyzer": "webaudio",
    "beat_detector": "webaudio",
    "amplitude_follower": "webaudio",
    "spectrum_analyzer": "webaudio",
    "audio_reactive": "webaudio",
    "waveform": "webaudio",

    # --- Video/media → html_video ---
    "webcam_input": "html_video",
    "video_input": "html_video",
    "image_input": "html_video",

    # --- Tracking / detection → canvas2d (uses ml5.js via dynamic script load) ---
    "tracking": "canvas2d",
    "face_detection": "canvas2d",
    "hand_tracking": "canvas2d",
    "p5_tracking": "canvas2d",
    "face_tracking": "canvas2d",
    "body_tracking": "canvas2d",
    "pose_tracking": "canvas2d",

    # --- Generic roles ---
    "generator": "canvas2d",
    "effect": "glsl",
    "modifier": "glsl",
    "output": "glsl",
    "control": "events",
    "math": "glsl",
    "utility": "events",
    "composite": "glsl",
    "feedback": "glsl",
    "router": "events",
}

def infer_engine_from_category(category: str, keywords: List[str] = None) -> str:
    """
    Dynamically routes nodes to the correct engine based on semantic intent,
    eliminating the need to hardcode every possible node name.
    """
    cat = (category or "").strip().lower()
    kws = " ".join(keywords or []).lower()
    
    # Combine category and keywords into one search string
    intent_string = f"{cat} {kws}"

    if not intent_string.strip():
        return "glsl"

    # 1. THREE.JS ROUTE (3D Geometry, Particles, Instancing)
    threejs_triggers = [
        "mesh", "particle", "point_cloud", "3d", "three", "volume", 
        "scatter", "scene", "geo_", "instancing", "sphere", "cube", 
        "torus", "wireframe", "physics", "collision"
    ]
    if any(trigger in intent_string for trigger in threejs_triggers):
        return "three_js"

    # 2. P5.JS ROUTE (Generative Art, Shapes, Lines, Vectors, Flow, Compositing)
    p5_triggers = [
        "draw", "shape", "canvas", "flower", "tree", "mandala",
        "fractal", "2d", "line", "path", "vector", "petals",
        "organic_growth", "hud", "text", "polygon", "sketch",
        "ui_", "typography", "circle", "rect", "triangle",
        "gradient", "star", "spiral", "grid_pattern", "flow",
        "fluid", "organic", "flock", "boid", "swarm", "pattern",
        "tracking", "face", "body", "pose", "hand",
        "composite", "output", "layer", "blend", "combine",
        "animation", "motion", "tween", "ease", "scale",
        "petal", "leaf", "stem", "seed", "bloom"
    ]
    if any(trigger in intent_string for trigger in p5_triggers):
        return "p5"

    # 3. WEBAUDIO ROUTE
    audio_triggers = [
        "audio", "fft", "beat", "amplitude", "spectrum", "waveform"
    ]
    if any(trigger in intent_string for trigger in audio_triggers):
        return "webaudio"

    # 4. HTML VIDEO ROUTE
    video_triggers = [
        "video", "webcam", "camera", "image", "footage"
    ]
    if any(trigger in intent_string for trigger in video_triggers):
        return "html_video"

    # 5. EVENTS ROUTE (Logic & Routing) — narrow triggers to avoid false positives
    event_triggers = [
        "event_router", "logic_gate", "lfo_oscillator", "param_router"
    ]
    if any(trigger in intent_string for trigger in event_triggers):
        return "events"

    # 6. GLSL ROUTE (Pixels, Shaders, Textures, Noise, Post-Processing)
    # Default fallback for noise, bloom, blur, color grading, SDFs, etc.
    return "glsl"

# Modality from category
CATEGORY_MODALITY_MAP = {
    "particle_emitter": "particles",
    "particle_system": "particles",
    "particle_renderer": "particles",
    "physics": "particles",
    "shape_generator": "mesh",
    "mesh_renderer": "mesh",
    "audio_input": "audio",
    "fft_analyzer": "audio",
    "beat_detector": "audio",
    "webaudio": "audio",
    "control": "event",
    "events": "event",
    "router": "event",
}

# Domain from category
CATEGORY_DOMAIN_MAP = {
    "audio_input": "audio",
    "fft_analyzer": "audio",
    "beat_detector": "audio",
    "audio_reactive": "audio",
    "physics": "physics",
    "gravity": "physics",
    "gravity_field": "physics",
    "particle_system": "physics",
    "attraction": "physics",
    "repulsion": "physics",
    "control": "logic",
    "router": "logic",
    "utility": "logic",
}

# Role → Z-layer mapping (deterministic)
ROLE_Z_MAP = {
    "input": 0,
    "control": 1,
    "utility": 1,
    "process": 2,   # will spread across Z=2..N-2
    "output": -1,    # sentinel: assigned to max Z
}

# Category → role inference
CATEGORY_ROLE_MAP = {
    "webcam_input": "input",
    "video_input": "input",
    "image_input": "input",
    "audio_input": "input",
    "noise_generator": "input",
    "perlin_noise": "input",
    "fractal_noise": "input",
    "oscillator": "input",
    "pattern_generator": "input",
    "field_generator": "input",
    "control": "control",
    "router": "control",
    "utility": "utility",
    "math": "utility",
    "output": "output",
    "composite": "output",
}


# Engine compatibility: which engines can feed into which.
# GLSL nodes consume textures via sampler2D — only other GLSL/regl nodes
# (and html_video via TextureHub) produce textures.
# Three.js render-to-texture is a special case handled by the runtime, so
# three_js → glsl is allowed (runtime does RTT).
# webaudio/events produce data, not textures — they can feed into any JS
# engine or into GLSL only if the runtime encodes data into a DataTexture.
ENGINE_CAN_FEED: Dict[str, set] = {
    "glsl":       {"glsl", "regl", "three_js", "html_video", "canvas2d"},
    "regl":       {"glsl", "regl", "three_js", "html_video", "canvas2d"},
    "three_js":   {"three_js", "glsl", "regl", "events", "webaudio"},
    "webaudio":   {"webaudio", "events"},
    "events":     {"events", "webaudio", "glsl", "three_js"},
    "canvas2d":   {"canvas2d", "glsl", "regl", "html_video"},
    "html_video": {"glsl", "regl", "canvas2d"},
}


def _engines_compatible(source_engine: str, target_engine: str) -> bool:
    """Check if source engine's output can feed into target engine's input."""
    allowed_sources = ENGINE_CAN_FEED.get(target_engine, set())
    return source_engine in allowed_sources


class ArchitectAgent:
    """Deterministic graph planner with topology validation (Filter 1).

    Guarantees ALL Phase 1 archetypes appear in the grid.
    Uses role-based Z-layer assignment, then asks LLM for connection refinement.
    Engine-aware wiring ensures output nodes connect to compatible predecessors.
    """

    def __init__(self):
        self.model = MODEL_NAME_REASONING
        self.max_retries = 3
        self.creative_topology = CreativeTopologyAgent()

    def plan(self, session_json: Dict) -> ArchitectPlan:
        """Take Phase 1 session JSON → produce ArchitectPlan with VolumetricGrid.

        ALL archetypes from Phase 1 are guaranteed to appear in the grid.
        Uses LLM-driven creative topology validated against Phase 1 brand data.
        """
        phase2_ctx = session_json.get('phase2_context', {})
        archetypes = phase2_ctx.get('node_archetypes', [])
        essence = phase2_ctx.get('essence', '')
        brief = (
            session_json.get('input', {}).get('prompt_text', '')
            or session_json.get('brief', {}).get('essence', '')
            or session_json.get('phase2_context', {}).get('essence', '')
            or session_json.get('user_input', {}).get('brief', '')
            or ''
        )
        brand_values = session_json.get('brand_values', {})
        visual_palette = session_json.get('visual_palette', {})

        if not archetypes:
            return ArchitectPlan(
                grid=VolumetricGrid(dimensions=(8, 8, 4)),
                reasoning="No archetypes from Phase 1",
                topology_valid=False,
                validation_log=["ERROR: No node archetypes provided"]
            )

        n = len(archetypes)
        print(f"  [ARCHITECT] Planning grid for ALL {n} archetypes...")

        # Step 1: LLM designs creative topology validated by Phase 1 data
        topology = self.creative_topology.design_topology(
            nodes=archetypes,
            brief=brief,
            essence=essence,
            brand_values=brand_values,
            visual_palette=visual_palette
        )

        # Step 2: Set grid_size from max_z (CreativeTopology returns max_z, not grid_size)
        max_z = topology.get('max_z', 4)
        topology['grid_size'] = [8, 8, max_z + 1]  # +1 because Z is 0-indexed

        # Step 3: Build VolumetricGrid
        grid = self._build_grid(topology, archetypes)

        # Step 4: Filter 1 — validate
        valid, log = self._validate_topology(grid)

        retries = 0
        while not valid and retries < self.max_retries:
            retries += 1
            print(f"  [ARCHITECT] Repair attempt {retries}/{self.max_retries}...")
            grid = self._repair_topology(grid, log)
            valid, log = self._validate_topology(grid)

        print(f"  [ARCHITECT] Final grid: {len(grid.nodes)} nodes, "
              f"{len(grid.connections)} connections, {grid.get_z_layers()} Z-layers")

        return ArchitectPlan(
            grid=grid,
            reasoning=topology.get('reasoning', f'Placed all {n} archetypes'),
            topology_valid=valid,
            validation_log=log
        )

    # =========================================================================
    # Deterministic Topology — guarantees ALL archetypes
    # =========================================================================

    def _build_deterministic_topology(self, archetypes: List[Dict]) -> Dict:
        """Build topology using DAG-first approach: classify roles, build DAG, layout.

        Uses DAGLayout for deterministic position assignment from the DAG.
        """
        n = len(archetypes)

        # Classify each archetype by role
        sources, processes, outputs = [], [], []
        for i, arch in enumerate(archetypes):
            category = arch.get('category', 'effect')
            role = arch.get('role', CATEGORY_ROLE_MAP.get(category, 'process'))
            if role == 'input':
                sources.append(i)
            elif role == 'output':
                outputs.append(i)
            else:
                processes.append(i)

        if not outputs and processes:
            outputs.append(processes.pop())
        if not sources and processes:
            sources.insert(0, processes.pop(0))

        # Build DAG: sources -> process chain -> outputs
        dag = []
        if processes:
            for src in sources:
                dag.append({"from": src, "to": processes[0]})
            for i in range(len(processes) - 1):
                dag.append({"from": processes[i], "to": processes[i + 1]})
            for out in outputs:
                dag.append({"from": processes[-1], "to": out})
        else:
            for src in sources:
                for out in outputs:
                    dag.append({"from": src, "to": out})

        # Fill orphans
        node_roles = {}
        for i in sources:
            node_roles[i] = "source"
        for i in processes:
            node_roles[i] = "process"
        for i in outputs:
            node_roles[i] = "output"
        dag = DAGLayout.ensure_dag_complete(dag, n, node_roles)

        # Convert DAG -> positions
        layout = DAGLayout()
        topology = layout.layout(dag, n)

        # Add grid_size to each node
        for tn in topology.get("nodes", []):
            if "grid_size" not in tn:
                tn["grid_size"] = [1, 1]

        topology['reasoning'] = (f'Deterministic DAG: {len(sources)} sources, '
                                 f'{len(processes)} processes, {len(outputs)} outputs')
        return topology

    # =========================================================================
    # LLM Connection Refinement (optional, does NOT drop nodes)
    # =========================================================================

    def _llm_refine_connections(self, archetypes: List[Dict], essence: str,
                                 topology: Dict) -> Optional[List[Dict]]:
        """Ask LLM to suggest better connections between nodes.
        Returns list of {from: idx, to: idx} or None on failure."""
        summary = "\n".join([
            f"  [{i}] {a['name']} (role={a.get('role','process')}, category={a.get('category','effect')})"
            for i, a in enumerate(archetypes)
        ])

        prompt = f"""You are refining connections for a visual node graph.

Essence: {essence}

Nodes (ALL must stay, do NOT remove any):
{summary}

Current topology has {len(topology['nodes'])} nodes across {topology['grid_size'][2]} Z-layers.

Suggest connections as a JSON array of {{\"from\": index, \"to\": index}}.
Rules:
- from must have lower Z than to (no backwards connections)
- Input nodes (role=input) should feed into process nodes
- Process nodes chain into each other
- Output nodes receive from process nodes
- Return 10-30 connections for {len(archetypes)} nodes

Respond ONLY with a JSON array, no markdown."""

        try:
            aider = get_aider_llm()
            text = aider.call(prompt, self.model)
            if text:
                match = re.search(r'\[[\s\S]*\]', text)
                if match:
                    conns = json.loads(match.group())
                    if isinstance(conns, list) and len(conns) > 0:
                        return conns
        except Exception as e:
            print(f"  [ARCHITECT] LLM connection refinement error: {e}")

        return None

    def _merge_llm_connections(self, topology: Dict, llm_conns: List[Dict],
                               n: int, archetypes: List[Dict] = None) -> Dict:
        """Merge LLM-suggested connections into the deterministic topology.
        Engine-aware: rejects connections between incompatible engines."""
        for node in topology['nodes']:
            node['input_from'] = []  # reset connections

        for conn in llm_conns:
            try:
                from_idx = int(conn.get('from', -1))
                to_idx = int(conn.get('to', -1))
            except (ValueError, TypeError):
                continue
            if 0 <= from_idx < n and 0 <= to_idx < n and from_idx != to_idx:
                # Engine compatibility check
                if archetypes:
                    src_cat = archetypes[from_idx].get('category', 'effect')
                    tgt_cat = archetypes[to_idx].get('category', 'effect')
                    src_eng = infer_engine_from_category(src_cat)
                    tgt_eng = infer_engine_from_category(tgt_cat)
                    if not _engines_compatible(src_eng, tgt_eng):
                        continue  # skip incompatible connection

                # Support both old (archetype_index) and new (node_index) formats
                target_node = next(
                    (nd for nd in topology['nodes']
                     if nd.get('node_index', nd.get('archetype_index')) == to_idx),
                    None
                )
                if target_node and from_idx not in target_node['input_from']:
                    target_node['input_from'].append(from_idx)

        # Ensure at least chain connectivity for orphans
        idx_to_node = {nd.get('node_index', nd.get('archetype_index')): nd
                       for nd in topology['nodes']}
        for nd in topology['nodes']:
            if not nd['input_from'] and nd['grid_position'][2] > 0:
                # Find any node at lower Z
                my_z = nd['grid_position'][2]
                candidates = [
                    other.get('node_index', other.get('archetype_index'))
                    for other in topology['nodes']
                    if other['grid_position'][2] < my_z
                ]
                if candidates:
                    nd['input_from'] = candidates[-4:]  # up to 4 lower-Z nodes

        return topology

    # =========================================================================
    # Build Grid
    # =========================================================================

    def _build_grid(self, topology: Dict, archetypes: List[Dict]) -> VolumetricGrid:
        """Convert topology dict into VolumetricGrid with NodeTensors."""
        grid_size = topology.get('grid_size', [8, 8, 4])
        topo_nodes = topology.get('nodes', [])

        node_tensors = []
        connections = []
        id_map = {}  # node_index -> node_id

        # First pass: create node IDs (sanitised for GLSL identifier safety)
        import re
        def _sanitize_id(name: str) -> str:
            """Strip everything except [a-zA-Z0-9_] so the ID is a valid GLSL identifier."""
            return re.sub(r'[^a-zA-Z0-9_]', '_', name.lower()).strip('_') or 'unknown'

        skipped_indices = set()
        for tn in topo_nodes:
            idx = tn.get('node_index', tn.get('archetype_index', 0))
            if idx < 0 or idx >= len(archetypes):
                # Out-of-range: skip entirely instead of clamping to wrong archetype
                skipped_indices.add(id(tn))
                print(f"  [ARCHITECT] Skipping topo node with out-of-range index {idx} (max {len(archetypes)-1})")
                continue
            arch = archetypes[idx]
            node_id = f"node_{idx}_{_sanitize_id(arch.get('name', 'unknown'))}"
            id_map[idx] = node_id

        # Second pass: build NodeTensors
        for tn in topo_nodes:
            if id(tn) in skipped_indices:
                continue
            idx = tn.get('node_index', tn.get('archetype_index', 0))
            if idx < 0 or idx >= len(archetypes):
                continue

            arch = archetypes[idx]
            pos = tuple(tn.get('grid_position', [0, 0, 0]))
            size = tuple(tn.get('grid_size', [1, 1]))

            category = arch.get('category', 'effect')
            # Pass keywords so the dynamic router can detect generative art intent
            keywords = arch.get('keywords', [])
            # Explicit engine from archetype (set by SemanticReasoner/LLM) takes priority.
            # Fall back to keyword-based inference, then GLSL as last resort.
            engine = (
                arch.get('engine')
                or (arch.get('engine_hints') or {}).get('preferred_engine')
                or infer_engine_from_category(category, keywords=keywords)
            )

            modality = CATEGORY_MODALITY_MAP.get(category, 'texture')
            domain = CATEGORY_DOMAIN_MAP.get(category, 'visual')
            role = arch.get('role', 'process')

            node_id = id_map[idx]

            # Preserve the original meta dict if it exists (includes keywords, parameter_ui, etc.)
            # Otherwise create a new meta dict (not NodeMeta object, so Mason can add fields)
            original_meta = arch.get('meta', {})
            if isinstance(original_meta, dict):
                meta = {
                    'concept_id': original_meta.get('concept_id', arch.get('id', f'arch_{idx}')),
                    'label': original_meta.get('label', arch.get('name', f'Node {idx}')),
                    'level': original_meta.get('level', arch.get('creative_level', 'surface')),
                    'modality': original_meta.get('modality', modality),
                    'domain': original_meta.get('domain', domain),
                    'description': original_meta.get('description', arch.get('description', '')),
                    'category': original_meta.get('category', category),
                    'role': original_meta.get('role', role),
                    'keywords': original_meta.get('keywords', arch.get('keywords', [])),
                }
            else:
                meta = {
                    'concept_id': arch.get('id', f'arch_{idx}'),
                    'label': arch.get('name', f'Node {idx}'),
                    'level': arch.get('creative_level', 'surface'),
                    'modality': modality,
                    'domain': domain,
                    'description': arch.get('description', ''),
                    'category': category,
                    'role': role,
                    'keywords': arch.get('keywords', []),
                }

            texture = TextureHandle(
                node_id=node_id,
                z_layer=pos[2] if len(pos) > 2 else 0
            ) if domain == 'visual' else None

            input_ids = []
            for from_idx in tn.get('input_from', []):
                try:
                    from_idx = int(from_idx)
                except (ValueError, TypeError):
                    continue
                if from_idx in id_map:
                    from_id = id_map[from_idx]
                    if from_id not in input_ids:
                        input_ids.append(from_id)
                        connections.append(Connection(
                            from_node=from_id,
                            from_output=0,
                            to_node=node_id,
                            to_input=len(input_ids) - 1
                        ))

            # Handle parameters - can be list of defs or plain dict
            raw_params = arch.get('parameters', [])
            if isinstance(raw_params, dict):
                # Already a {name: value} dict (predefined nodes)
                params = raw_params
            elif isinstance(raw_params, list):
                # List of parameter definitions [{name, default}, ...]
                params = {}
                for j, p in enumerate(raw_params):
                    if isinstance(p, dict):
                        params[p.get('name', f'p{j}')] = p.get('default', 0)
                    elif isinstance(p, str):
                        params[p] = 0  # String param name with default 0
            else:
                params = {}

            # Semantic purpose from brief decomposition (JEPA-style)
            semantic_map = topology.get("semantic_map", {})
            sem_info = semantic_map.get(idx, {})
            semantic_purpose = sem_info.get("purpose", "")

            nt = NodeTensor(
                id=node_id,
                meta=meta,
                grid_position=pos,
                grid_size=size,
                engine=engine,
                code_snippet="",  # Mason fills this in
                parameters=params,
                input_nodes=input_ids,
                output_texture=texture,
                semantic_purpose=semantic_purpose,
            )
            node_tensors.append(nt)

        grid = VolumetricGrid(
            dimensions=tuple(grid_size),
            nodes=node_tensors,
            connections=connections,
            runtime_hints={
                'essence': topology.get('reasoning', ''),
                'total_z_layers': grid_size[2] if len(grid_size) > 2 else 4
            }
        )

        # Normalize grid positions to form proper 3D blocks
        grid = self._normalize_grid_positions(grid)

        # CRITICAL FIX: If creative_topology didn't wire nodes (0 connections),
        # fall back to volumetric grid-proximity wiring to ensure connectivity.
        if not grid.connections:
            print("  [ARCHITECT] No connections from creative_topology, adding volumetric wiring...")
            grid = self._add_volumetric_connections(grid)

        return grid

    def _normalize_grid_positions(self, grid: VolumetricGrid) -> VolumetricGrid:
        """Normalize grid positions to ensure non-overlapping 3D blocks.

        Groups nodes by Z-layer and assigns non-overlapping X,Y positions.
        """
        # Group nodes by Z layer
        z_layers: Dict[int, List[NodeTensor]] = {}
        for n in grid.nodes:
            z = n.grid_position[2] if len(n.grid_position) > 2 else 0
            if z not in z_layers:
                z_layers[z] = []
            z_layers[z].append(n)

        # Renumber Z layers to be contiguous (0, 1, 2, ...)
        sorted_zs = sorted(z_layers.keys())
        z_remap = {old_z: new_z for new_z, old_z in enumerate(sorted_zs)}

        # For each Z layer, assign non-overlapping X,Y positions
        max_x, max_y = 0, 0
        for old_z in sorted_zs:
            nodes_at_z = z_layers[old_z]
            new_z = z_remap[old_z]

            # Arrange nodes in a grid at this Z level
            # Try to respect original X,Y but resolve conflicts
            occupied: Dict[Tuple[int, int], str] = {}

            for n in nodes_at_z:
                orig_x = n.grid_position[0] if len(n.grid_position) > 0 else 0
                orig_y = n.grid_position[1] if len(n.grid_position) > 1 else 0

                # Find nearest unoccupied position
                x, y = orig_x, orig_y
                attempts = 0
                while (x, y) in occupied and attempts < 100:
                    # Spiral outward to find free slot
                    attempts += 1
                    if attempts % 4 == 1:
                        x += 1
                    elif attempts % 4 == 2:
                        y += 1
                    elif attempts % 4 == 3:
                        x -= 1 if x > 0 else 0
                    else:
                        y -= 1 if y > 0 else 0

                occupied[(x, y)] = n.id
                n.grid_position = (x, y, new_z)
                max_x = max(max_x, x)
                max_y = max(max_y, y)

        # Update grid dimensions
        new_z_count = len(sorted_zs)
        grid.dimensions = (max_x + 1, max_y + 1, new_z_count)

        return grid

    def _add_volumetric_connections(self, grid: VolumetricGrid) -> VolumetricGrid:
        """Add connections based on 3D grid proximity (XYZ neighbors).

        Volumetric connection rules:
        1. Z-1 neighbors: Nodes connect to compatible nodes at the previous Z layer
        2. XY neighbors at same Z: Nodes can blend with adjacent nodes (fallback if no Z-layers)
        3. Respects engine compatibility
        """
        node_by_id = {n.id: n for n in grid.nodes}
        pos_to_node = {}  # (x, y, z) -> node_id

        # Build position map
        for n in grid.nodes:
            pos = tuple(n.grid_position[:3]) if len(n.grid_position) >= 3 else (n.grid_position[0], n.grid_position[1], 0)
            pos_to_node[pos] = n.id

        new_connections = list(grid.connections)
        existing_pairs = {(c.from_node, c.to_node) for c in grid.connections}
        
        # Check if we have multi-layer Z structure
        z_values = set()
        for n in grid.nodes:
            z = n.grid_position[2] if len(n.grid_position) >= 3 else 0
            z_values.add(z)
        has_z_layers = len(z_values) > 1

        for n in grid.nodes:
            x, y, z = n.grid_position[:3] if len(n.grid_position) >= 3 else (n.grid_position[0], n.grid_position[1], 0)
            n_engine = (n.engine or "glsl").strip()

            # Z-1 layer neighbors (primary data flow direction)
            if has_z_layers and z > 0:
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        neighbor_pos = (x + dx, y + dy, z - 1)
                        neighbor_id = pos_to_node.get(neighbor_pos)
                        if neighbor_id and neighbor_id != n.id:
                            neighbor = node_by_id.get(neighbor_id)
                            if neighbor:
                                neighbor_engine = (neighbor.engine or "glsl").strip()
                                if _engines_compatible(neighbor_engine, n_engine):
                                    if (neighbor_id, n.id) not in existing_pairs:
                                        if len(n.input_nodes) < 4:
                                            if neighbor_id not in n.input_nodes:
                                                n.input_nodes.append(neighbor_id)
                                                new_connections.append(Connection(
                                                    from_node=neighbor_id,
                                                    from_output=0,
                                                    to_node=n.id,
                                                    to_input=len(n.input_nodes) - 1
                                                ))
                                                existing_pairs.add((neighbor_id, n.id))

            # XY neighbors at same Z: ENABLED as fallback when no Z-layers exist
            # This ensures connectivity even when stratification fails
            if n_engine in ("glsl", "regl") and len(n.input_nodes) < 2:
                # Reduced strictness: allow any two GLSL nodes to connect spatially
                for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:  # All 4 directions for connectivity
                    neighbor_pos = (x + dx, y + dy, z)
                    neighbor_id = pos_to_node.get(neighbor_pos)
                    if neighbor_id and neighbor_id != n.id:
                        neighbor = node_by_id.get(neighbor_id)
                        if neighbor:
                            neighbor_engine = (neighbor.engine or "glsl").strip()
                            if neighbor_engine in ("glsl", "regl"):
                                if (neighbor_id, n.id) not in existing_pairs:
                                    if neighbor_id not in n.input_nodes:
                                        n.input_nodes.append(neighbor_id)
                                        new_connections.append(Connection(
                                            from_node=neighbor_id,
                                            from_output=0,
                                            to_node=n.id,
                                            to_input=len(n.input_nodes) - 1
                                        ))
                                        existing_pairs.add((neighbor_id, n.id))

        grid.connections = new_connections
        return grid

    # =========================================================================
    # Filter 1: Topology Validation
    # =========================================================================

    def _validate_topology(self, grid: VolumetricGrid) -> Tuple[bool, List[str]]:
        """Filter 1 - validate graph topology."""
        log = []
        valid = True
        node_ids = {n.id for n in grid.nodes}

        for n in grid.nodes:
            for inp in n.input_nodes:
                if inp not in node_ids:
                    log.append(f"WARN: {n.id} references non-existent input {inp}")
                    valid = False

        # Check cycles via DFS
        adj = {n.id: n.input_nodes for n in grid.nodes}
        visited = set()
        rec_stack = set()

        def has_cycle(node_id):
            visited.add(node_id)
            rec_stack.add(node_id)
            for dep in adj.get(node_id, []):
                if dep not in visited:
                    if has_cycle(dep):
                        return True
                elif dep in rec_stack:
                    return True
            rec_stack.discard(node_id)
            return False

        for n in grid.nodes:
            if n.id not in visited:
                if has_cycle(n.id):
                    log.append("ERROR: Cycle detected in graph")
                    valid = False
                    break

        # Check Z ordering (engine compatibility not checked — runtime does auto-conversion)
        node_by_id = {nd.id: nd for nd in grid.nodes}
        for n in grid.nodes:
            z = n.grid_position[2] if len(n.grid_position) > 2 else 0
            for inp_id in n.input_nodes:
                inp_node = node_by_id.get(inp_id)
                if inp_node:
                    inp_z = inp_node.grid_position[2] if len(inp_node.grid_position) > 2 else 0
                    if inp_z > z:
                        log.append(f"WARN: {n.id} at Z={z} takes input from {inp_id} at Z={inp_z} (backwards)")

        # Check for orphan non-source nodes (no inputs but not a source)
        source_keywords = {"noise", "generator", "input", "sampler", "oscillator", "source", "loader", "field"}
        for n in grid.nodes:
            if n.input_nodes:
                continue
            cat = (getattr(n, 'category', '') or n.id or '').lower()
            is_source = any(kw in cat for kw in source_keywords)
            if not is_source:
                log.append(f"WARN: Orphan node {n.id} has no inputs and is not a source")

        # Check graph depth
        all_z = [n.grid_position[2] if len(n.grid_position) > 2 else 0 for n in grid.nodes]
        max_z = max(all_z) if all_z else 0
        if max_z < 2 and len(grid.nodes) > 3:
            log.append(f"WARN: Shallow graph (depth={max_z}), expected >= 2 for {len(grid.nodes)} nodes")

        # Check output node placement
        output_keywords = {"output", "composite", "final", "render", "display"}
        for n in grid.nodes:
            cat = (getattr(n, 'category', '') or n.id or '').lower()
            is_output = any(kw in cat for kw in output_keywords)
            z = n.grid_position[2] if len(n.grid_position) > 2 else 0
            if is_output and z < max_z:
                log.append(f"WARN: Output node {n.id} at Z={z} but maxZ={max_z} (should be at end)")

        if valid and not log:
            log.append("OK: Topology valid")

        return valid, log

    def _repair_topology(self, grid: VolumetricGrid, issues: List[str]) -> VolumetricGrid:
        """Attempt to repair topology issues including engine-incompatible connections."""
        node_ids = {n.id for n in grid.nodes}
        node_by_id = {nd.id: nd for nd in grid.nodes}

        for n in grid.nodes:
            n.input_nodes = [inp for inp in n.input_nodes if inp in node_ids]

        # NOTE: Engine-incompatible connection filtering removed.
        # Rich tensor auto-conversion in the runtime handles cross-engine data flow.
        # Position IS the connection graph (Lego 1:1 snap) — we preserve it as-is.

        # Recompute depths
        depths = {}
        def get_depth(nid, seen=None):
            if seen is None:
                seen = set()
            if nid in depths:
                return depths[nid]
            if nid in seen:
                return 0
            seen.add(nid)
            node = node_by_id.get(nid)
            if not node or not node.input_nodes:
                depths[nid] = 0
                return 0
            d = max(get_depth(inp, seen) for inp in node.input_nodes) + 1
            depths[nid] = d
            return d

        for n in grid.nodes:
            get_depth(n.id)

        for n in grid.nodes:
            d = depths.get(n.id, 0)
            n.grid_position = (n.grid_position[0], n.grid_position[1], d)

        # Force output nodes to maxZ
        output_keywords = {"output", "composite", "final", "render", "display"}
        max_depth = max(depths.values()) if depths else 0
        for n in grid.nodes:
            cat = (getattr(n, 'category', '') or n.id or '').lower()
            is_output = any(kw in cat for kw in output_keywords)
            if is_output and depths.get(n.id, 0) < max_depth:
                new_z = max_depth + 1
                n.grid_position = (n.grid_position[0], n.grid_position[1], new_z)
                depths[n.id] = new_z
                # Ensure output has input from deepest non-output node
                deepest_non_output = [
                    nid for nid, d in depths.items()
                    if d == max_depth and nid != n.id
                    and not any(kw in (getattr(node_by_id.get(nid), 'category', '') or nid or '').lower()
                                for kw in output_keywords)
                ]
                if deepest_non_output and not n.input_nodes:
                    n.input_nodes = deepest_non_output[:2]
                    print(f"  [REPAIR] Moved output {n.id} to Z={new_z}, connected from {n.input_nodes}")

        # Rebuild connections list from input_nodes
        grid.connections = []
        for n in grid.nodes:
            for i, inp_id in enumerate(n.input_nodes):
                if inp_id in node_ids:
                    grid.connections.append(Connection(
                        from_node=inp_id,
                        from_output=0,
                        to_node=n.id,
                        to_input=i
                    ))

        max_z = max((n.grid_position[2] for n in grid.nodes), default=3) + 1
        grid.dimensions = (grid.dimensions[0], grid.dimensions[1], max_z)

        return grid