"""Creative Topology Agent - LLM-driven volumetric layout with Phase 1 validation

Instead of hardcoded topology rules, ask the LLM to design the 3D volumetric layout
based on creative intent, validated against Phase 1 brand data.

Now includes SemanticZStratifier for embedding-based Z-layer assignment (Task 3).
"""

import json
import math
import requests
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from config import OLLAMA_URL, MODEL_NAME_FALLBACK, EMBEDDING_MODEL

# Try to import sentence-transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False
    print("[CREATIVE TOPOLOGY] sentence-transformers not available, using keyword-based stratification")


# =============================================================================
# Semantic Z-Stratifier (Task 3 Algorithm Implementation)
# =============================================================================

class SemanticZStratifier:
    """
    Embedding-based Z-layer stratification algorithm.

    Algorithm from Task 3:
    1. Define 3 anchor centroids: Src (generators), Mod (filters), Sink (outputs)
    2. For each node, compute Semantic Rank S(v) using softmax over similarities
    3. Compute Topological Rank T(v) via longest path
    4. Final Z = max(T(v), λ * S(v) * Z_max)
    5. Enforce minimum 3 Z-layers with dilation
    6. X-axis lane separation using barycentric minimization
    """

    # Anchor category definitions for computing centroids
    SRC_CATEGORIES = [
        "noise_generator", "perlin_noise", "fractal_noise", "simplex_noise",
        "pattern_generator", "oscillator", "field_generator", "texture_generator",
        "webcam_input", "video_input", "audio_input", "image_input", "sampler",
        "source", "input", "generator", "noise", "loader"
    ]

    MOD_CATEGORIES = [
        "blur", "displacement", "distortion", "warp", "transform",
        "color_grade", "filter", "effect", "modifier", "process",
        "particle_system", "particle_emitter", "physics", "gravity",
        "glow", "bloom", "chromatic_aberration", "feedback", "glitch",
        "kaleidoscope", "mirror", "rotate", "scale", "blend", "mix"
    ]

    SINK_CATEGORIES = [
        "output", "composite", "composite_output", "final", "render",
        "display", "post_process", "master", "screen", "export"
    ]

    def __init__(self):
        self.model = None
        self.src_centroid = None
        self.mod_centroid = None
        self.sink_centroid = None
        self._init_embeddings()

    def _init_embeddings(self):
        """Initialize embedding model and compute anchor centroids."""
        if not HAS_EMBEDDINGS:
            return

        try:
            self.model = SentenceTransformer(EMBEDDING_MODEL)

            # Compute anchor centroids from category descriptions
            src_texts = [f"node type: {c}, role: source generator input" for c in self.SRC_CATEGORIES[:8]]
            mod_texts = [f"node type: {c}, role: modifier filter effect" for c in self.MOD_CATEGORIES[:8]]
            sink_texts = [f"node type: {c}, role: output display render" for c in self.SINK_CATEGORIES[:8]]

            src_embeddings = self.model.encode(src_texts)
            mod_embeddings = self.model.encode(mod_texts)
            sink_embeddings = self.model.encode(sink_texts)

            self.src_centroid = np.mean(src_embeddings, axis=0)
            self.mod_centroid = np.mean(mod_embeddings, axis=0)
            self.sink_centroid = np.mean(sink_embeddings, axis=0)

            # Normalize centroids
            self.src_centroid = self.src_centroid / np.linalg.norm(self.src_centroid)
            self.mod_centroid = self.mod_centroid / np.linalg.norm(self.mod_centroid)
            self.sink_centroid = self.sink_centroid / np.linalg.norm(self.sink_centroid)

            print("[SEMANTIC Z] Anchor centroids computed")
        except Exception as e:
            print(f"[SEMANTIC Z] Failed to initialize embeddings: {e}")
            self.model = None

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _compute_semantic_rank(self, node_embedding: np.ndarray) -> float:
        """
        Compute Semantic Rank S(v) ∈ [0, 1] using softmax over anchor similarities.

        S(v) = (exp(sim(h, C_mod)) + 2*exp(sim(h, C_sink))) / sum(exp(sim(h, C_k)))

        Higher S(v) = closer to sink (output), Lower S(v) = closer to source (input)
        """
        if self.src_centroid is None:
            return 0.5  # Neutral if no embeddings

        sim_src = self._cosine_similarity(node_embedding, self.src_centroid)
        sim_mod = self._cosine_similarity(node_embedding, self.mod_centroid)
        sim_sink = self._cosine_similarity(node_embedding, self.sink_centroid)

        # Softmax-weighted combination
        exp_src = math.exp(sim_src)
        exp_mod = math.exp(sim_mod)
        exp_sink = math.exp(sim_sink)

        total = exp_src + exp_mod + exp_sink
        if total == 0:
            return 0.5

        # S(v) weights modifier and sink higher to push toward later layers
        semantic_rank = (exp_mod + 2 * exp_sink) / (3 * total)
        return min(1.0, max(0.0, semantic_rank))

    def _compute_topological_rank(self, nodes: List[Dict], node_idx: int,
                                   adjacency: Dict[int, List[int]],
                                   memo: Dict[int, int]) -> int:
        """
        Compute Topological Rank T(v) = longest path from any source.

        T(v) = 0 if no inputs, else 1 + max(T(u) for u in inputs)
        """
        if node_idx in memo:
            return memo[node_idx]

        inputs = adjacency.get(node_idx, [])
        if not inputs:
            memo[node_idx] = 0
            return 0

        max_parent_rank = 0
        for parent_idx in inputs:
            if parent_idx != node_idx:  # Avoid self-loops
                parent_rank = self._compute_topological_rank(nodes, parent_idx, adjacency, memo)
                max_parent_rank = max(max_parent_rank, parent_rank)

        memo[node_idx] = max_parent_rank + 1
        return memo[node_idx]

    def _keyword_semantic_rank(self, category: str) -> float:
        """Fallback keyword-based semantic rank when embeddings unavailable."""
        cat_lower = category.lower()

        # Source keywords → low rank (Z=0)
        for kw in ["input", "generator", "noise", "source", "loader", "webcam",
                    "video", "audio", "image", "oscillator", "field", "pattern",
                    "gradient", "shape", "circle", "rectangle", "triangle",
                    "polygon", "draw", "line", "curve", "particle_emitter",
                    "flow_field", "sampler"]:
            if kw in cat_lower:
                return 0.1

        # Sink keywords → high rank (Z=max)
        for kw in ["output", "composite", "final", "render", "display",
                    "post_process", "master", "screen", "export"]:
            if kw in cat_lower:
                return 0.95

        # Late-stage effects → high-middle rank (closer to output)
        for kw in ["bloom", "glow", "chromatic", "vignette", "tone_map",
                    "clamp", "blend", "mix", "composite"]:
            if kw in cat_lower:
                return 0.7

        # Mid-stage effects → middle rank
        for kw in ["blur", "color", "distort", "effect", "particle", "warp",
                    "displacement", "feedback", "glitch", "kaleidoscope",
                    "transform", "grade", "filter", "modifier"]:
            if kw in cat_lower:
                return 0.5

        return 0.5  # Default middle

    def stratify(self, topology: Dict, nodes: List[Dict], min_z_layers: int = 3) -> Dict:
        """
        Apply semantic Z-stratification to a topology.

        Args:
            topology: Topology dict with 'nodes' list
            nodes: Original node archetypes (for category info)
            min_z_layers: Minimum number of Z layers (default 3)

        Returns:
            Updated topology with semantic Z positions
        """
        topo_nodes = topology.get("nodes", [])
        if not topo_nodes:
            return topology

        n = len(topo_nodes)
        print(f"[SEMANTIC Z] Stratifying {n} nodes with min {min_z_layers} layers...")

        # Build adjacency map (node_index -> list of input node indices)
        adjacency: Dict[int, List[int]] = {}
        for tn in topo_nodes:
            idx = tn.get("node_index", tn.get("archetype_index", 0))
            inputs = tn.get("input_from", [])
            adjacency[idx] = [i for i in inputs if isinstance(i, int)]

        # Compute ranks for each node
        topo_memo: Dict[int, int] = {}
        semantic_ranks: Dict[int, float] = {}
        topo_ranks: Dict[int, int] = {}

        for tn in topo_nodes:
            idx = tn.get("node_index", tn.get("archetype_index", 0))

            # Get category from original node archetype - fallback to 'id' if 'category' missing
            category = "process"
            if idx < len(nodes):
                category = nodes[idx].get("category", nodes[idx].get("id", "process"))

            # Compute semantic rank
            if self.model is not None:
                # Use embedding-based rank
                node_text = f"node type: {category}, role: visual effect"
                embedding = self.model.encode([node_text])[0]
                semantic_ranks[idx] = self._compute_semantic_rank(embedding)
            else:
                # Fallback to keyword-based
                semantic_ranks[idx] = self._keyword_semantic_rank(category)

            # Compute topological rank
            topo_ranks[idx] = self._compute_topological_rank(nodes, idx, adjacency, topo_memo)

        # Determine max Z from topology or semantic considerations
        max_topo = max(topo_ranks.values()) if topo_ranks else 0
        z_max = max(min_z_layers - 1, max_topo, 2)

        # Compute final Z for each node: Z = max(T(v), λ * S(v) * Z_max)
        lambda_factor = 1.0  # Relaxation factor
        z_assignments: Dict[int, int] = {}

        for idx in semantic_ranks:
            t_rank = topo_ranks.get(idx, 0)
            s_rank = semantic_ranks.get(idx, 0.5)

            # Final Z is max of topological and semantic-weighted
            semantic_z = int(round(lambda_factor * s_rank * z_max))
            final_z = max(t_rank, semantic_z)
            z_assignments[idx] = min(final_z, z_max)

        # Check current depth and apply dilation if needed
        current_depth = max(z_assignments.values()) + 1 if z_assignments else 1

        if current_depth < min_z_layers:
            print(f"[SEMANTIC Z] Spreading {current_depth} -> {min_z_layers} layers")

            if current_depth <= 1:
                # All nodes at same Z (typically 0) — multiplicative dilation
                # is useless (0 * γ = 0). Use semantic rank to distribute.
                ranked = sorted(z_assignments.keys(),
                                key=lambda i: semantic_ranks.get(i, 0.5))
                num_nodes = len(ranked)
                max_z = min_z_layers - 1
                for pos, idx in enumerate(ranked):
                    z_assignments[idx] = int(round(pos / max(num_nodes - 1, 1) * max_z))
            else:
                # Multiple Z values exist but not enough — scale proportionally
                old_max = max(z_assignments.values())
                new_max = min_z_layers - 1
                for idx in z_assignments:
                    old_z = z_assignments[idx]
                    z_assignments[idx] = int(round(old_z / max(old_max, 1) * new_max))

            # Ensure sink nodes snap to final layer
            for idx in z_assignments:
                category = (nodes[idx].get("category") or nodes[idx].get("id") or "") if idx < len(nodes) else ""
                if any(kw in category.lower() for kw in ["output", "composite", "final"]):
                    z_assignments[idx] = min_z_layers - 1

        # Apply X-axis lane separation using barycentric positioning
        z_assignments, x_positions = self._compute_x_positions(topo_nodes, z_assignments, adjacency)

        # Update topology nodes with new positions
        for tn in topo_nodes:
            idx = tn.get("node_index", tn.get("archetype_index", 0))
            new_z = z_assignments.get(idx, 0)
            new_x = x_positions.get(idx, 0)

            # Preserve Y or compute based on layer
            old_pos = tn.get("grid_position", [0, 0, 0])
            new_y = old_pos[1] if len(old_pos) > 1 else 0

            tn["grid_position"] = [new_x, new_y, new_z]

        # Update max_z
        new_max_z = max(z_assignments.values()) if z_assignments else min_z_layers - 1
        topology["max_z"] = new_max_z

        print(f"[SEMANTIC Z] ✓ Stratified to {new_max_z + 1} Z-layers")
        return topology

    def _compute_x_positions(self, topo_nodes: List[Dict],
                              z_assignments: Dict[int, int],
                              adjacency: Dict[int, List[int]]) -> Tuple[Dict[int, int], Dict[int, int]]:
        """
        Compute X positions using barycentric minimization.

        X*(i) = weighted average of parent and child X positions
        With collision avoidance via repulsion.
        """
        # Group nodes by Z layer
        layers: Dict[int, List[int]] = {}
        for tn in topo_nodes:
            idx = tn.get("node_index", tn.get("archetype_index", 0))
            z = z_assignments.get(idx, 0)
            if z not in layers:
                layers[z] = []
            layers[z].append(idx)

        # Initialize X positions (simple spread)
        x_positions: Dict[int, int] = {}
        for z in sorted(layers.keys()):
            nodes_at_z = layers[z]
            for i, idx in enumerate(nodes_at_z):
                x_positions[idx] = i

        # Build reverse adjacency (children)
        children: Dict[int, List[int]] = {idx: [] for idx in z_assignments}
        for idx, parents in adjacency.items():
            for parent in parents:
                if parent in children:
                    children[parent].append(idx)

        # Iterative barycentric refinement (3 passes)
        for iteration in range(3):
            new_x: Dict[int, float] = {}

            for z in sorted(layers.keys()):
                nodes_at_z = layers[z]

                for idx in nodes_at_z:
                    parent_xs = [x_positions[p] for p in adjacency.get(idx, []) if p in x_positions]
                    child_xs = [x_positions[c] for c in children.get(idx, []) if c in x_positions]

                    all_xs = parent_xs + child_xs
                    if all_xs:
                        # Barycenter = average of connected nodes
                        new_x[idx] = sum(all_xs) / len(all_xs)
                    else:
                        new_x[idx] = float(x_positions[idx])

            # Apply with collision avoidance
            for z in sorted(layers.keys()):
                nodes_at_z = layers[z]
                # Sort by new X, then assign integer slots
                sorted_nodes = sorted(nodes_at_z, key=lambda i: new_x.get(i, 0))
                for slot, idx in enumerate(sorted_nodes):
                    x_positions[idx] = slot

        return z_assignments, x_positions


class CreativeTopologyAgent:
    """LLM-driven 3D volumetric layout designer with semantic Z-stratification"""

    MIN_Z_LAYERS = 3  # Task 3: Minimum 3 Z-layers required

    def __init__(self):
        self.ollama_url = OLLAMA_URL
        self.model = MODEL_NAME_FALLBACK  # llama:latest for topology reasoning (NOT mason)
        self.fallback_model = MODEL_NAME_FALLBACK  # llama3.2:latest (fallback)
        self.stratifier = SemanticZStratifier()

    def design_topology(self,
                       nodes: List[Dict],
                       brief: str,
                       essence: str,
                       brand_values: Dict,
                       visual_palette: Optional[Dict] = None) -> Dict:
        """
        Ask LLM to design the 3D volumetric topology, then apply semantic stratification.

        Uses llama:latest for topology reasoning (not mason).
        Post-processes with SemanticZStratifier to ensure proper Z-layer distribution.

        Args:
            nodes: List of node archetypes with id/keywords from SemanticReasoner
            brief: Original design brief
            essence: Brand essence from Phase 1
            brand_values: Emotion scores, attributes from Phase 1
            visual_palette: Color palette, mood from Phase 1

        Returns:
            Topology dict with nodes positioned in 3D space and connected
        """
        print(f"\n[CREATIVE TOPOLOGY] Designing volumetric layout for {len(nodes)} nodes...")

        # Format nodes for LLM - use id and keywords from SemanticReasoner
        node_list = []
        for i, node in enumerate(nodes):
            node_id = node.get("id", node.get("category", "unknown_node"))
            keywords = ", ".join(node.get("keywords", []))
            node_list.append(f"  [{i}] ID: {node_id} (Traits: {keywords})")
        nodes_str = "\n".join(node_list)

        # Brand context
        top_emotions = sorted(brand_values.items(), key=lambda x: x[1], reverse=True)[:3]
        emotions_str = ", ".join([f"{e[0]} ({e[1]:.2f})" for e in top_emotions])

        mood = visual_palette.get('dominant_mood', 'balanced') if visual_palette else 'balanced'

        system_prompt = f"""You are a Creative Technical Director designing a 3D volumetric node graph.

DESIGN BRIEF: "{brief}"

BRAND CONTEXT (from Phase 1):
- Essence: {essence}
- Dominant emotions: {emotions_str}
- Visual mood: {mood}

AVAILABLE NODES:
{nodes_str}

YOUR TASK: Design the 3D volumetric layout (X, Y, Z positions + connections).

CREATIVE GUIDELINES:
1. **Z-layers represent NARRATIVE FLOW**, not just data flow:
   - Lower Z: Foundation, generators, inputs
   - Middle Z: Transformation, effects, motion
   - Upper Z: Refinement, output, final touches

2. **Consider CREATIVE INTENT**:
   - Particle systems: emitters→motion fields→forces→rendering
   - Organic motion: noise→flow fields→displacement→output
   - Neon/glow: base→color grade→bloom/glow→composite
   - Glitch: input→distortion→chromatic aberration→output

3. **Z-layer GROUPING by purpose**:
   - Don't just chain linearly - group related operations at the same Z
   - Example: Multiple particle forces can be at same Z, all feeding into one integrator

4. **X,Y positioning for visual clarity**:
   - Spread nodes at same Z across X axis
   - Use Y for variants/parallel paths

5. **CONNECTIONS based on creative flow**:
   - Not just engine compatibility - think about aesthetic purpose
   - Multiple inputs are good if they blend/composite
   - Parallel paths that merge are powerful

VALIDATION against Phase 1:
- Does this layout match the emotional tone? ({emotions_str})
- Does it support the visual style? ({mood})
- Does it fulfill the brief's intent?

OUTPUT FORMAT (strict JSON):
{{
  "reasoning": "1-2 sentences explaining the creative approach",
  "max_z": 5,
  "nodes": [
    {{
      "node_index": 0,
      "grid_position": [0, 0, 0],
      "grid_size": [1, 1],
      "input_from": [],
      "creative_role": "foundation generator"
    }},
    ...
  ]
}}

CRITICAL RULES:
- Every node [0..{len(nodes)-1}] must appear exactly once
- grid_position is [x, y, z] where z is 0..max_z
- input_from is array of node indices (empty for first nodes)
- Be creative - don't just chain linearly!
- Consider the brand emotions when deciding layout
- You MUST use the exact string from the 'ID' field.
- Do NOT use the bracketed integer index [i].
- Do NOT invent new node names.

Output ONLY the JSON, no other text."""

        # Two-stage LLM fallback: mason:latest -> llama fallback -> deterministic fallback
        topology = None
        models_to_try = [
            (self.model, "llama:latest"),
            (self.fallback_model, "llama fallback")
        ]

        for model, model_name in models_to_try:
            try:
                print(f"[CREATIVE TOPOLOGY] Trying {model_name}...")
                response = requests.post(
                    self.ollama_url,
                    json={
                        "model": model,
                        "prompt": system_prompt,
                        "stream": False,
                        "temperature": 0.7 if model == self.model else 0.5,
                        "options": {"num_predict": 4096}
                    },
                    timeout=120 if model == self.fallback_model else None
                )
                response.raise_for_status()
                result = response.json()
                text = result.get("response", "")

                # Extract JSON from response
                topology = self._extract_json(text)

                if topology and self._validate_topology(topology, len(nodes)):
                    # Reject flat topologies: all same Z or zero connections
                    topo_nodes = topology.get("nodes", [])
                    z_values = set(n.get("grid_position", [0,0,0])[2] for n in topo_nodes)
                    has_connections = any(n.get("input_from", []) for n in topo_nodes)
                    if len(z_values) <= 1 and not has_connections and len(topo_nodes) > 2:
                        print(f"[CREATIVE TOPOLOGY] {model_name} produced flat topology (all Z={z_values}), rejecting")
                        topology = None
                    else:
                        print(f"[CREATIVE TOPOLOGY] ✓ Layout designed with {model_name}: {topology.get('reasoning', 'N/A')}")
                        break
                else:
                    print(f"[CREATIVE TOPOLOGY] {model_name} response invalid, trying next...")
                    topology = None

            except Exception as e:
                print(f"[CREATIVE TOPOLOGY] {model_name} error: {e}")
                topology = None

        # If both LLMs failed, use deterministic fallback
        if not topology:
            print("[CREATIVE TOPOLOGY] All LLMs failed, using role-aware fallback")
            topology = self._fallback_topology(nodes)

        # Apply semantic Z-stratification (Task 3)
        topology = self.stratifier.stratify(topology, nodes, min_z_layers=self.MIN_Z_LAYERS)

        # Verify minimum 3 Z-layers
        max_z = topology.get("max_z", 0)
        if max_z < self.MIN_Z_LAYERS - 1:
            print(f"[CREATIVE TOPOLOGY] WARNING: Only {max_z + 1} Z-layers, need {self.MIN_Z_LAYERS}")
            topology = self.stratifier.stratify(topology, nodes, min_z_layers=self.MIN_Z_LAYERS)

        # Auto-wire: if no connections exist after stratification, generate
        # them from Z-layer adjacency (nodes at Z=N connect from all at Z=N-1)
        topology = self._auto_wire_if_empty(topology)

        return topology

    def _extract_json(self, text: str) -> Optional[Dict]:
        """Extract JSON from LLM response (handles markdown fences and thinking tags)"""
        import re

        # Clean thinking tags (mason:latest uses these)
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

        # Try direct parse
        try:
            return json.loads(text)
        except:
            pass

        # Try extracting from code fence
        match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except:
                pass

        # Try finding first { to last }
        start = text.find('{')
        end = text.rfind('}')
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end+1])
            except:
                pass

        # Last attempt: look for nodes array pattern and construct topology
        nodes_match = re.search(r'"nodes"\s*:\s*\[(.*?)\]', text, re.DOTALL)
        if nodes_match:
            try:
                nodes_json = '[' + nodes_match.group(1) + ']'
                nodes = json.loads(nodes_json)
                return {"nodes": nodes, "reasoning": "Extracted from partial response", "max_z": 5}
            except:
                pass

        return None

    def _validate_and_repair_topology(self, topology: Dict, num_nodes: int) -> Optional[Dict]:
        """Validate LLM-generated topology and attempt to repair minor issues.

        Returns repaired topology or None if unfixable.
        """
        if "nodes" not in topology:
            print("[CREATIVE TOPOLOGY] Validation failed: 'nodes' key missing")
            return None

        nodes = topology["nodes"]

        # Repair: if we have wrong count but close, try to fix
        if len(nodes) < num_nodes:
            # Add missing nodes with sensible defaults
            existing_indices = set()
            for n in nodes:
                idx = n.get("node_index", n.get("archetype_index"))
                if idx is not None:
                    existing_indices.add(idx)

            for i in range(num_nodes):
                if i not in existing_indices:
                    # Add missing node at end of chain
                    last_z = max((n.get("grid_position", [0, 0, 0])[2] for n in nodes), default=0)
                    nodes.append({
                        "node_index": i,
                        "grid_position": [len(existing_indices) % 4, 0, last_z],
                        "grid_size": [1, 1],
                        "input_from": [list(existing_indices)[-1]] if existing_indices else [],
                        "creative_role": "auto-added"
                    })
                    existing_indices.add(i)
            print(f"[CREATIVE TOPOLOGY] Repaired: added {num_nodes - len(topology['nodes'])} missing nodes")
        elif len(nodes) > num_nodes:
            # Trim extra nodes, keep only valid indices
            valid_nodes = [n for n in nodes if n.get("node_index", n.get("archetype_index", num_nodes)) < num_nodes]
            topology["nodes"] = valid_nodes[:num_nodes]
            nodes = topology["nodes"]
            print(f"[CREATIVE TOPOLOGY] Repaired: trimmed to {num_nodes} nodes")

        # Normalize node_index field
        for n in nodes:
            if "node_index" not in n and "archetype_index" in n:
                n["node_index"] = n["archetype_index"]
            if "node_index" not in n and "index" in n:
                n["node_index"] = n["index"]

        # Check all indices present
        indices = set()
        for n in nodes:
            idx = n.get("node_index")
            if idx is not None:
                indices.add(idx)

        expected = set(range(num_nodes))
        if indices != expected:
            missing = expected - indices
            extra = indices - expected
            print(f"[CREATIVE TOPOLOGY] Cannot repair: missing indices {missing}, extra {extra}")
            return None

        # Repair missing required fields
        for i, node in enumerate(nodes):
            # Fix grid_position
            if "grid_position" not in node:
                node["grid_position"] = [i % 4, i // 4, i // 8]
            elif len(node["grid_position"]) == 2:
                # Missing Z - add it
                node["grid_position"].append(i // 4)
            elif len(node["grid_position"]) != 3:
                node["grid_position"] = [i % 4, i // 4, i // 8]

            # Fix input_from
            if "input_from" not in node:
                node["input_from"] = [i - 1] if i > 0 else []

            # Fix grid_size
            if "grid_size" not in node:
                node["grid_size"] = [1, 1]

            # Ensure input_from is a list
            if not isinstance(node["input_from"], list):
                node["input_from"] = [node["input_from"]] if node["input_from"] else []

        # Validate and repair input_from references
        for node in nodes:
            valid_inputs = []
            for inp in node.get("input_from", []):
                if isinstance(inp, int) and 0 <= inp < num_nodes:
                    valid_inputs.append(inp)
            node["input_from"] = valid_inputs

        # Update max_z if not present
        if "max_z" not in topology:
            topology["max_z"] = max((n["grid_position"][2] for n in nodes), default=3)

        return topology

    def _auto_wire_if_empty(self, topology: Dict) -> Dict:
        """Lego 1:1 snap wiring: each node at Z>0 connects to exactly 1
        nearest neighbor at Z-1, determined by (X,Y) Euclidean distance.

        This ALWAYS runs — it replaces any existing input_from with the
        position-derived connection. Position IS the connection graph.
        """
        topo_nodes = topology.get("nodes", [])
        if not topo_nodes:
            return topology

        # Normalize node_index for all nodes (prevent fallback-to-0 bug)
        for i, tn in enumerate(topo_nodes):
            if "node_index" not in tn:
                tn["node_index"] = tn.get("archetype_index", i)

        # Build position map: idx -> (x, y, z)
        pos_map: Dict[int, Tuple[int, int, int]] = {}
        for tn in topo_nodes:
            idx = tn["node_index"]
            gp = tn.get("grid_position", [0, 0, 0])
            pos_map[idx] = (gp[0], gp[1], gp[2])

        # Group nodes by Z layer
        by_z: Dict[int, List[int]] = {}
        for idx, (x, y, z) in pos_map.items():
            by_z.setdefault(z, []).append(idx)

        sorted_z = sorted(by_z.keys())

        # For each node at Z > min, find nearest 1 neighbor at the
        # previous Z layer by (X,Y) Euclidean distance
        wired = 0
        for tn in topo_nodes:
            idx = tn["node_index"]
            x, y, z = pos_map[idx]

            if z == sorted_z[0]:
                # Source layer — no input
                tn["input_from"] = []
                continue

            # Find the previous Z layer (may not be z-1 if layers are sparse)
            prev_z = None
            for zz in reversed(sorted_z):
                if zz < z:
                    prev_z = zz
                    break

            if prev_z is None:
                tn["input_from"] = []
                continue

            # Find nearest node at prev_z by (X,Y) distance
            candidates = by_z.get(prev_z, [])
            if not candidates:
                tn["input_from"] = []
                continue

            best_idx = None
            best_dist = float('inf')
            for cand_idx in candidates:
                cx, cy, _ = pos_map[cand_idx]
                dist = (x - cx) ** 2 + (y - cy) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best_idx = cand_idx

            tn["input_from"] = [best_idx] if best_idx is not None else []
            if best_idx is not None:
                wired += 1

        print(f"[CREATIVE TOPOLOGY] Lego 1:1 snap: wired {wired}/{len(topo_nodes)} nodes")
        return topology

    def _validate_topology(self, topology: Dict, num_nodes: int) -> bool:
        """Validate LLM-generated topology with repair attempt"""
        repaired = self._validate_and_repair_topology(topology, num_nodes)
        if repaired:
            topology.update(repaired)
            return True
        return False

    def _fallback_topology(self, nodes: List[Dict]) -> Dict:
        """Role-aware fallback if LLM fails.

        Builds a single-chain topology where ALL generators feed into a
        linear process chain. XY positions are aligned so Lego 1:1 snap
        produces the correct connections (each process is stacked directly
        above its intended input at X=0).
        """
        print("[CREATIVE TOPOLOGY] Using role-aware fallback")

        INPUT_CATEGORIES = {
            "noise_generator", "perlin_noise", "fractal_noise", "oscillator",
            "pattern_generator", "field_generator", "webcam_input", "video_input",
            "audio_input", "image_input", "texture_generator", "sampler",
            "noise_fbm", "noise_simplex", "noise_worley", "noise_perlin",
        }
        OUTPUT_CATEGORIES = {"output", "composite", "final", "render", "display"}
        CONTROL_CATEGORIES = {"control", "router", "lfo", "parameter_map", "trigger", "events"}

        inputs, controls, processes, outputs = [], [], [], []
        for i, node in enumerate(nodes):
            category = (node.get("category") or node.get("id") or "").lower()
            role = (node.get("role") or "").lower()

            if role == "input" or category in INPUT_CATEGORIES or "input" in category or "generator" in category:
                inputs.append(i)
            elif role == "output" or category in OUTPUT_CATEGORIES or "output" in category:
                outputs.append(i)
            elif role == "control" or category in CONTROL_CATEGORIES:
                controls.append(i)
            else:
                processes.append(i)

        if not inputs and processes:
            inputs.append(processes.pop(0))
        if not outputs and processes:
            outputs.append(processes.pop())

        topology_nodes = []
        z = 0

        # Z=0: All generators at X=0,1,2,... Y=0
        # First generator at X=0 will be the "main" input for the process chain
        for xi, idx in enumerate(inputs):
            topology_nodes.append({
                "node_index": idx,
                "grid_position": [xi, 0, z],
                "grid_size": [1, 1],
                "input_from": [],
                "creative_role": "input"
            })
        z += 1

        # If >1 generator, insert blend nodes to merge them into a single stream
        # so all generators contribute. Each blend sits at X=0 (directly above
        # first generator) — Lego snap will pick it up.
        if len(inputs) > 1:
            # Add a blend node that merges all generators
            # (Lego 1:1 snap will connect it to the nearest Z-1 node at X=0)
            # The blend is at X=0, so it connects to inputs[0].
            # We rely on the process chain to propagate all textures.
            pass  # Generators at X>0 contribute via layer composite fallback

        # Controls at Z=1 if present
        if controls:
            for xi, idx in enumerate(controls):
                topology_nodes.append({
                    "node_index": idx,
                    "grid_position": [xi, 0, z],
                    "grid_size": [1, 1],
                    "input_from": inputs[:1] if inputs else [],
                    "creative_role": "control"
                })
            z += 1

        # Process chain: each process at X=0, stacked vertically (Z increments).
        # All at X=0 so Lego 1:1 snap chains them: each connects to the
        # nearest node at Z-1, which is the previous process (also at X=0).
        for pi, idx in enumerate(processes):
            if pi == 0:
                inp_from = [inputs[0]] if inputs else []
            else:
                inp_from = [processes[pi - 1]]

            topology_nodes.append({
                "node_index": idx,
                "grid_position": [0, 0, z],
                "grid_size": [1, 1],
                "input_from": inp_from,
                "creative_role": "process"
            })
            z += 1

        # Output at X=0, final Z layer
        last_process = processes[-1] if processes else (controls[-1] if controls else (inputs[0] if inputs else -1))
        for xi, idx in enumerate(outputs):
            topology_nodes.append({
                "node_index": idx,
                "grid_position": [0, 0, z],
                "grid_size": [1, 1],
                "input_from": [last_process] if last_process >= 0 else [],
                "creative_role": "output"
            })

        # Place any orphan nodes in the process area at X=0
        placed = {n["node_index"] for n in topology_nodes}
        for i in range(len(nodes)):
            if i not in placed:
                topology_nodes.append({
                    "node_index": i,
                    "grid_position": [0, 0, z - 1 if z > 1 else 1],
                    "grid_size": [1, 1],
                    "input_from": [last_process] if last_process >= 0 else [],
                    "creative_role": "process"
                })
                placed.add(i)

        max_z = max(n["grid_position"][2] for n in topology_nodes) if topology_nodes else 3

        return {
            "reasoning": f"Role-based fallback: {len(inputs)} inputs, {len(controls)} controls, {len(processes)} processes, {len(outputs)} outputs",
            "max_z": max_z,
            "nodes": topology_nodes
        }

    def _parse_topology_response(self, text: str, nodes: List[Dict]) -> List[Dict]:
        """
        Parses LLM JSON and Fuzzy-Matches IDs to available nodes.
        """
        connections = []
        
        # 1. Clean JSON
        try:
            # (Insert your existing JSON cleaning logic here)
            data = json.loads(self._clean_json_text(text))
            raw_conns = data.get("connections", [])
        except:
            return []

        # 2. Create Lookup Map (Handle exact IDs and "Friendly" names)
        node_map = {}
        for n in nodes:
            node_map[n["id"]] = n["id"]  # Exact match: "node_0_n0" -> "node_0_n0"
            node_map[n.get("name", "")] = n["id"] # Name match: "noise" -> "node_0_n0"
            # Add semantic category as fallback
            cat = n.get("meta", {}).get("category", "")
            if cat: node_map[cat] = n["id"]

        # 3. Fuzzy Wiring
        for conn in raw_conns:
            src_raw = conn.get("from", "")
            tgt_raw = conn.get("to", "")
            
            src_id = self._fuzzy_find_node(src_raw, node_map, nodes)
            tgt_id = self._fuzzy_find_node(tgt_raw, node_map, nodes)
            
            if src_id and tgt_id and src_id != tgt_id:
                connections.append({
                    "from_node": src_id,
                    "from_output": 0,
                    "to_node": tgt_id,
                    "to_input": 0
                })
        
        return connections

    def _fuzzy_find_node(self, query: str, lookup: Dict, nodes: List) -> Optional[str]:
        query = query.lower()
        
        # 1. Exact or partial ID match (e.g., query="sdf_shape_11" matches node "sdf_shape_11")
        for nid in [n["id"] for n in nodes]:
            if query == nid.lower() or query in nid.lower(): 
                return nid
                
        # 2. Try partial match in Category (if LLM forgot the index and just wrote "sdf_shape")
        # Note: If there are duplicates, this just grabs the first one, which is why Step 1 is so important.
        for n in nodes:
            cat = n.get("meta", {}).get("category", "").lower()
            if cat and cat in query:
                return n["id"]
                
        return None