"""Node Affinity System — Transformer-Inspired Attention for Blend Weights
--------------------------------------------------------------------------
Computes pairwise affinity between connected nodes using projected
dot-product attention over node metadata embeddings.

Key insight (vs naive self-attention):
  Standard transformers use DIFFERENT projection matrices for Q and K
  (W_Q and W_K) so a token can "ask for" something different than what
  it "is". With one-hot metadata embeddings, a naive dot(Q, K) only
  measures SIMILARITY — it punishes sequential pipeline flow because
  a process node (role=[0,1,0]) has zero dot product with a source
  node (role=[1,0,0]) on the role dimension.

  Fix: We introduce a hardcoded Interaction Matrix M so the score is:
    score = Q_modality · K_modality        (similarity — good for modality/domain)
          + Q_role^T · M_role · K_role     (compatibility — off-diagonal synergies)
          + Q_level^T · M_level · K_level  (compatibility — cross-layer attention)
          + keyword_jaccard_bonus

  M_role encodes pipeline expectations:
    - Output nodes seek Process inputs (weight 1.5)
    - Process nodes seek Source inputs (weight 1.2) + Process chain (weight 1.0)
    - Source nodes don't strongly prefer anything (they generate from nothing)

  M_level encodes abstraction flow:
    - Narrative nodes attend to Flow (weight 1.3)
    - Flow nodes attend to Surface (weight 1.2) + Flow chain (weight 1.0)

  Softmax temperature τ=0.5 sharpens the distribution (sparse one-hot
  vectors produce tightly clustered logits that softmax would otherwise
  flatten to near-uniform).

Pipeline position: Called by pipeline.py after Architect topology, before Mason.
Weights are injected into node.parameters so the runtime applies them
automatically via the existing params→uniforms mapping.
"""

import math
from typing import Dict, List, Tuple, Set


# ── Vocabulary for one-hot encoding ──────────────────────────────────────

MODALITY_VOCAB = {
    "field": 0, "particles": 1, "mesh": 2, "audio": 3,
    "event": 4, "texture": 5, "data": 6
}

DOMAIN_VOCAB = {
    "visual": 0, "physics": 1, "audio": 2, "logic": 3, "control": 4
}

LEVEL_VOCAB = {
    "surface": 0, "flow": 1, "narrative": 2
}

ROLE_VOCAB = {
    "input": 0, "source": 0,    # source = input (same slot)
    "process": 1,
    "output": 2,
    "control": 3,
    "utility": 4
}

# Embedding dimensions
D_MODALITY = len(set(MODALITY_VOCAB.values()))  # 7
D_DOMAIN = len(set(DOMAIN_VOCAB.values()))      # 5
D_LEVEL = len(set(LEVEL_VOCAB.values()))        # 3
D_ROLE = len(set(ROLE_VOCAB.values()))           # 5

# Dimension offsets within the embedding vector
OFF_MODALITY = 0
OFF_DOMAIN = D_MODALITY                          # 7
OFF_LEVEL = D_MODALITY + D_DOMAIN                # 12
OFF_ROLE = D_MODALITY + D_DOMAIN + D_LEVEL       # 15
D_TOTAL = D_MODALITY + D_DOMAIN + D_LEVEL + D_ROLE  # 20

# ── Interaction matrices (the W projection) ──────────────────────────────
#
# M_role[target_role][source_role] = compatibility score
# Encodes "what does this role SEEK from its inputs?"
#
# Key intuition:
#   - Output nodes want processed data → high weight for Process inputs
#   - Process nodes want raw data → high weight for Source, medium for Process chain
#   - Source nodes are generators → slight preference for other sources (parallel gen)

_R = ROLE_VOCAB  # shorthand

M_ROLE: Dict[int, Dict[int, float]] = {
    _R["output"]:  {_R["source"]: 0.8, _R["process"]: 1.5, _R["output"]: 0.3, _R["control"]: 0.6, _R["utility"]: 0.5},
    _R["process"]: {_R["source"]: 1.2, _R["process"]: 1.0, _R["output"]: 0.2, _R["control"]: 0.8, _R["utility"]: 0.6},
    _R["source"]:  {_R["source"]: 0.5, _R["process"]: 0.3, _R["output"]: 0.1, _R["control"]: 0.4, _R["utility"]: 0.3},
    _R["control"]: {_R["source"]: 0.6, _R["process"]: 0.8, _R["output"]: 0.3, _R["control"]: 0.5, _R["utility"]: 0.7},
    _R["utility"]: {_R["source"]: 0.5, _R["process"]: 0.7, _R["output"]: 0.3, _R["control"]: 0.6, _R["utility"]: 0.5},
}

# M_level[target_level][source_level] = compatibility score
# Encodes abstraction flow: narrative ← flow ← surface
_L = LEVEL_VOCAB

M_LEVEL: Dict[int, Dict[int, float]] = {
    _L["narrative"]: {_L["surface"]: 0.6, _L["flow"]: 1.3, _L["narrative"]: 0.8},
    _L["flow"]:      {_L["surface"]: 1.2, _L["flow"]: 1.0, _L["narrative"]: 0.4},
    _L["surface"]:   {_L["surface"]: 0.8, _L["flow"]: 0.5, _L["narrative"]: 0.2},
}

# Softmax temperature — lower = sharper contrast between weights
# With sparse one-hot vectors, raw logits cluster tightly (~0.5-0.8 range).
# τ=0.5 ensures softmax can meaningfully discriminate between inputs.
TEMPERATURE = 0.5


class NodeAffinity:
    """Compute transformer-style attention weights between connected nodes.

    Uses projected dot-product: score = Q^T · M · K, where M encodes
    pipeline compatibility (not just similarity) for Role and Level dims.

    Usage:
        aff = NodeAffinity()
        aff.compute_embeddings(grid.nodes)
        weights = aff.compute_all_blend_weights(grid.nodes, grid.connections)
        aff.inject_weights(grid.nodes, weights)
    """

    def __init__(self, temperature: float = TEMPERATURE):
        self.embeddings: Dict[str, List[float]] = {}
        self.keywords_map: Dict[str, Set[str]] = {}
        self.affinity_cache: Dict[Tuple[str, str], float] = {}
        self.temperature = temperature

    # ── Embedding ────────────────────────────────────────────────────────

    def compute_embeddings(self, nodes) -> Dict[str, List[float]]:
        """Compute d-dimensional embedding for each node from its metadata.

        Args:
            nodes: List of NodeTensor objects (or dicts with meta)

        Returns:
            {node_id: [float] * D_TOTAL}
        """
        self.embeddings = {}
        self.keywords_map = {}
        self.affinity_cache = {}

        for node in nodes:
            node_id = node.id if hasattr(node, 'id') else node.get('id', '')
            meta = node.meta if hasattr(node, 'meta') else node.get('meta', {})

            # Extract metadata fields (handle both NodeMeta and dict)
            if hasattr(meta, 'modality'):
                modality = meta.modality
                domain = meta.domain
                level = meta.level
                role = meta.role
            elif isinstance(meta, dict):
                modality = meta.get('modality', 'texture')
                domain = meta.get('domain', 'visual')
                level = meta.get('level', 'surface')
                role = meta.get('role', 'process')
            else:
                modality, domain, level, role = 'texture', 'visual', 'surface', 'process'

            embed = self._encode(modality, domain, level, role)
            self.embeddings[node_id] = embed

            # Collect keywords for Jaccard bonus
            kws = set()
            if hasattr(node, 'keywords'):
                kws = set(node.keywords or [])
            elif isinstance(node, dict):
                kws = set(node.get('keywords', []))
            self.keywords_map[node_id] = kws

        return self.embeddings

    def _encode(self, modality: str, domain: str, level: str, role: str) -> List[float]:
        """One-hot encode node metadata into a fixed-size vector."""
        vec = [0.0] * D_TOTAL

        # Modality (7 dims, offset 0)
        vec[OFF_MODALITY + MODALITY_VOCAB.get(modality, 5)] = 1.0

        # Domain (5 dims, offset 7)
        vec[OFF_DOMAIN + DOMAIN_VOCAB.get(domain, 0)] = 1.0

        # Level (3 dims, offset 12)
        vec[OFF_LEVEL + LEVEL_VOCAB.get(level, 0)] = 1.0

        # Role (5 dims, offset 15)
        vec[OFF_ROLE + ROLE_VOCAB.get(role, 1)] = 1.0

        return vec

    # ── Helpers to extract sub-vectors ───────────────────────────────────

    @staticmethod
    def _get_active_idx(vec: List[float], offset: int, dim: int) -> int:
        """Find which one-hot index is active in a sub-vector."""
        for i in range(dim):
            if vec[offset + i] > 0.5:
                return i
        return 0

    # ── Affinity: Q^T · M · K (projected dot-product) ───────────────────

    def _keyword_jaccard(self, id_a: str, id_b: str) -> float:
        """Jaccard similarity of keyword sets — bonus affinity for shared tags."""
        kw_a = self.keywords_map.get(id_a, set())
        kw_b = self.keywords_map.get(id_b, set())
        if not kw_a and not kw_b:
            return 0.0
        union = kw_a | kw_b
        if not union:
            return 0.0
        return len(kw_a & kw_b) / len(union)

    def compute_affinity(self, target_id: str, source_id: str) -> float:
        """Projected dot-product affinity: score = Q^T · M · K.

        Three components:
        1. Modality + Domain: standard dot product (similarity IS correct here —
           a visual node should prefer visual inputs over audio inputs)
        2. Role: interaction matrix M_role (compatibility, not similarity —
           a process node SEEKS source inputs, not other process nodes)
        3. Level: interaction matrix M_level (abstraction flow —
           narrative SEEKS flow inputs, flow SEEKS surface)
        + keyword Jaccard bonus
        """
        cache_key = (target_id, source_id)
        if cache_key in self.affinity_cache:
            return self.affinity_cache[cache_key]

        embed_q = self.embeddings.get(target_id)
        embed_k = self.embeddings.get(source_id)

        if not embed_q or not embed_k:
            return 0.0

        # ── Component 1: Modality + Domain similarity (standard dot product) ──
        # Dimensions [0..12) — here similarity IS what we want.
        # A visual+texture node should prefer visual+texture inputs.
        sim_score = sum(
            q * k for q, k in zip(
                embed_q[OFF_MODALITY:OFF_LEVEL],  # modality + domain
                embed_k[OFF_MODALITY:OFF_LEVEL]
            )
        )

        # ── Component 2: Role compatibility (Q^T · M_role · K) ──────────
        # Extract active role indices
        target_role = self._get_active_idx(embed_q, OFF_ROLE, D_ROLE)
        source_role = self._get_active_idx(embed_k, OFF_ROLE, D_ROLE)

        role_score = M_ROLE.get(target_role, {}).get(source_role, 0.5)

        # ── Component 3: Level compatibility (Q^T · M_level · K) ────────
        target_level = self._get_active_idx(embed_q, OFF_LEVEL, D_LEVEL)
        source_level = self._get_active_idx(embed_k, OFF_LEVEL, D_LEVEL)

        level_score = M_LEVEL.get(target_level, {}).get(source_level, 0.5)

        # ── Combine: weighted sum ────────────────────────────────────────
        # sim_score: 0-2 (max when both modality and domain match)
        # role_score: 0.1-1.5 (from interaction matrix)
        # level_score: 0.2-1.3 (from interaction matrix)
        # Total raw range: ~0.3 to ~4.8
        raw_score = sim_score + role_score + level_score

        # Scale by sqrt of effective dimensionality
        d_eff = D_MODALITY + D_DOMAIN + 2  # 2 for the interaction matrix lookups
        score = raw_score / math.sqrt(d_eff)

        # ── Keyword overlap bonus (Jaccard, capped at 0.5) ──────────────
        kw_bonus = self._keyword_jaccard(target_id, source_id) * 0.5
        score += kw_bonus

        self.affinity_cache[cache_key] = score
        return score

    # ── Blend weights (softmax with temperature) ─────────────────────────

    def compute_blend_weights(self, target_id: str, input_ids: List[str]) -> Dict[str, float]:
        """Softmax attention weights for a node's inputs, with temperature.

        For target j with inputs [i1, i2, ...]:
          s_k = Q_j^T · M · K_ik  (projected affinity)
          w_k = exp((s_k - max_s) / τ) / Σ exp((s_m - max_s) / τ)

        Temperature τ < 1.0 sharpens the distribution so it can
        meaningfully discriminate between inputs (sparse one-hot vectors
        produce tightly clustered raw logits that standard softmax flattens).

        Returns:
            {input_id: weight} where weights sum to 1.0
        """
        if not input_ids:
            return {}

        if len(input_ids) == 1:
            return {input_ids[0]: 1.0}

        # Raw attention scores (projected dot-product)
        scores = {in_id: self.compute_affinity(target_id, in_id) for in_id in input_ids}

        # Numerically stable softmax with temperature
        max_s = max(scores.values())
        exp_scores = {
            k: math.exp((v - max_s) / self.temperature)
            for k, v in scores.items()
        }
        total = sum(exp_scores.values())

        if total < 1e-12:
            uniform = 1.0 / len(input_ids)
            return {k: uniform for k in input_ids}

        return {k: round(v / total, 4) for k, v in exp_scores.items()}

    def compute_all_blend_weights(self, nodes, connections) -> Dict[str, Dict[str, float]]:
        """Compute blend weights for every node with 2+ inputs.

        Args:
            nodes: List of NodeTensor objects
            connections: List of Connection objects

        Returns:
            {target_node_id: {source_node_id: weight}}
        """
        # Build input adjacency
        input_map: Dict[str, List[str]] = {}
        for node in nodes:
            nid = node.id if hasattr(node, 'id') else node.get('id', '')
            input_map[nid] = []

        for conn in connections:
            from_id = conn.from_node if hasattr(conn, 'from_node') else conn.get('from_node', '')
            to_id = conn.to_node if hasattr(conn, 'to_node') else conn.get('to_node', '')
            if to_id in input_map and from_id not in input_map[to_id]:
                input_map[to_id].append(from_id)

        # Ensure embeddings exist
        if not self.embeddings:
            self.compute_embeddings(nodes)

        all_weights = {}
        for node_id, inputs in input_map.items():
            if len(inputs) >= 2:
                all_weights[node_id] = self.compute_blend_weights(node_id, inputs)

        return all_weights

    # ── Injection into node parameters ───────────────────────────────────

    def inject_weights(self, nodes, weights: Dict[str, Dict[str, float]]) -> None:
        """Inject affinity blend weights into node parameters.

        For GLSL nodes: adds affinity_0, affinity_1, ... to params dict.
        The existing ShaderExecutor param→uniform loop automatically sets
        u_affinity_0, u_affinity_1, etc. — zero runtime changes needed.

        For Canvas2D: adds affinity_weights dict to params for user code access.
        """
        for node in nodes:
            nid = node.id if hasattr(node, 'id') else node.get('id', '')
            if nid not in weights:
                continue

            w = weights[nid]
            engine = node.engine if hasattr(node, 'engine') else node.get('engine', '')
            params = node.parameters if hasattr(node, 'parameters') else node.get('parameters', {})
            if params is None:
                params = {}

            if engine in ('glsl', 'regl'):
                # Inject as indexed uniforms: affinity_0, affinity_1, ...
                input_ids = node.input_nodes if hasattr(node, 'input_nodes') else node.get('input_nodes', [])
                for i, in_id in enumerate(input_ids):
                    if in_id in w:
                        params[f'affinity_{i}'] = w[in_id]

            elif engine in ('canvas2d', 'p5'):
                # Inject as a weights dict for JS code access
                params['affinity_weights'] = w

            # Write back
            if hasattr(node, 'parameters'):
                node.parameters = params
            elif isinstance(node, dict):
                node['parameters'] = params

    # ── Diagnostics ──────────────────────────────────────────────────────

    def summarize(self, weights: Dict[str, Dict[str, float]]) -> str:
        """Human-readable summary of affinity weights."""
        lines = []
        for target, w in sorted(weights.items()):
            parts = [f"{src}={v:.3f}" for src, v in sorted(w.items(), key=lambda x: -x[1])]
            lines.append(f"  {target}: {', '.join(parts)}")
        return "\n".join(lines) if lines else "  (no multi-input nodes)"
