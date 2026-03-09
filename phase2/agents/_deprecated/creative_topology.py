"""Creative Topology Agent - DAG-first volumetric layout

DAG-First approach: LLM designs only connections (what feeds into what).
Deterministic algorithm (DAGLayout) converts the DAG into XYZ positions.
Lego snap validates (not overrides) the layout.

Semantic Decomposition (JEPA-inspired):
  One LLM call decomposes the design brief into per-node "latent predictions" —
  what each node should semantically contribute. This scoped purpose is used by
  Mason for focused code generation and validation instead of the monolithic brief.

Flow: Brief -> Decompose -> LLM connections (DAG) -> topo sort -> Z-layers -> Sugiyama XY -> positions
"""

import json
import re
from typing import List, Dict, Optional
from config import MODEL_NAME_FALLBACK
from phase2.aider_llm import get_aider_llm
from phase2.agents.dag_layout import DAGLayout, DAGScorer


# Role classification keywords
INPUT_KEYWORDS = {
    "noise_generator", "perlin_noise", "fractal_noise", "oscillator",
    "pattern_generator", "field_generator", "webcam_input", "video_input",
    "audio_input", "image_input", "texture_generator", "sampler",
    "noise_fbm", "noise_simplex", "noise_worley", "noise_perlin",
    "source", "input", "generator", "noise", "loader",
}

OUTPUT_KEYWORDS = {
    "output", "composite", "composite_output", "final", "render",
    "display", "post_process", "master", "screen", "export",
}

CONTROL_KEYWORDS = {
    "control", "router", "lfo", "parameter_map", "trigger", "events",
}


class CreativeTopologyAgent:
    """DAG-first volumetric layout designer.

    LLM designs only the data flow (DAG). Deterministic DAGLayout algorithm
    converts the DAG into XYZ positions where Lego snap naturally recovers
    the correct connections.
    """

    def __init__(self):
        self.model = MODEL_NAME_FALLBACK
        self.fallback_model = MODEL_NAME_FALLBACK
        # Semantic decomposition results: {node_index: {purpose, role}}
        self.semantic_map: Dict[int, Dict] = {}

    def design_topology(self,
                       nodes: List[Dict],
                       brief: str,
                       essence: str,
                       brand_values: Dict,
                       visual_palette: Optional[Dict] = None) -> Dict:
        """Design volumetric topology using multi-candidate DAG scoring.

        0. Decompose brief into per-node semantic purposes (JEPA-style)
        1. Generate 3 candidate DAGs (LLM, linear chain, wide fan)
        2. Score each with deterministic heuristics
        3. Pick best, force output to end of pipe
        4. DAGLayout converts DAG -> positions

        Returns:
            Topology dict with nodes having grid_position, input_from,
            and semantic_map for downstream Mason usage
        """
        num_nodes = len(nodes)
        print(f"\n[CREATIVE TOPOLOGY] Multi-candidate DAG layout for {num_nodes} nodes...")

        # Step 0: Semantic decomposition — one LLM call to assign per-node purpose + role
        self.semantic_map = self._decompose_brief(nodes, brief, essence)
        node_roles = self._classify_roles(nodes)
        scorer = DAGScorer()
        candidates = []

        # Candidate 1: LLM-designed DAG
        llm_dag = self._llm_design_dag(nodes, brief, essence, brand_values, visual_palette)
        llm_dag = DAGLayout.ensure_dag_complete(llm_dag, num_nodes, node_roles)
        candidates.append(("llm", llm_dag))

        # Candidate 2: Deterministic linear chain
        linear_dag = self._fallback_dag(nodes)
        linear_dag = DAGLayout.ensure_dag_complete(linear_dag, num_nodes, node_roles)
        candidates.append(("linear", linear_dag))

        # Candidate 3: Wide fan (all sources -> each process, all processes -> output)
        wide_dag = self._wide_fan_dag(nodes)
        wide_dag = DAGLayout.ensure_dag_complete(wide_dag, num_nodes, node_roles)
        candidates.append(("wide_fan", wide_dag))

        # Score and pick best
        best_label, best_dag, best_score = None, None, -999
        for label, dag in candidates:
            result = scorer.score(dag, num_nodes, node_roles)
            print(f"  [DAG SCORE] {label}: {result}")
            if result["total"] > best_score:
                best_label, best_dag, best_score = label, dag, result["total"]

        print(f"  [DAG SCORE] Winner: {best_label} (score={best_score})")

        # Force output to end of pipe
        best_dag = self._force_output_to_end(best_dag, num_nodes, node_roles)

        # Convert DAG -> positions via deterministic layout
        layout_engine = DAGLayout()
        topology = layout_engine.layout(best_dag, num_nodes)

        # Add grid_size to each node (required by _build_grid)
        for tn in topology.get("nodes", []):
            if "grid_size" not in tn:
                tn["grid_size"] = [1, 1]

        topology["reasoning"] = topology.get(
            "reasoning",
            f"Multi-candidate DAG ({best_label}, score={best_score}): {num_nodes} nodes"
        )
        # Attach semantic decomposition for downstream agents (Mason)
        topology["semantic_map"] = self.semantic_map
        return topology

    def _llm_design_dag(self, nodes: List[Dict], brief: str,
                        essence: str, brand_values: Dict,
                        visual_palette: Optional[Dict] = None) -> List[Dict]:
        """Ask LLM to design only the data flow connections (DAG).

        Returns: [{"from": int, "to": int}, ...]
        """
        # Format nodes for LLM
        node_list = []
        for i, node in enumerate(nodes):
            node_id = node.get("id", node.get("category", "unknown"))
            keywords = ", ".join(node.get("keywords", []))
            category = node.get("category", "")
            role = self.semantic_map[i]["role"] if i in self.semantic_map else self._classify_role_keywords(node)
            node_list.append(f"  [{i}] {node_id} (category={category}, role={role}, traits={keywords})")
        nodes_str = "\n".join(node_list)

        # Brand context
        top_emotions = sorted(brand_values.items(), key=lambda x: x[1], reverse=True)[:3]
        emotions_str = ", ".join([f"{e[0]} ({e[1]:.2f})" for e in top_emotions])
        mood = visual_palette.get('dominant_mood', 'balanced') if visual_palette else 'balanced'

        system_prompt = f"""You are a VFX Pipeline Architect designing data flow for a node graph.

DESIGN BRIEF: "{brief}"

BRAND CONTEXT:
- Essence: {essence}
- Emotions: {emotions_str}
- Mood: {mood}

AVAILABLE NODES:
{nodes_str}

YOUR TASK: Design which nodes feed into which other nodes.
Think about the visual compositing pipeline:
- Which nodes generate content (sources)? They have role=source.
- Which nodes transform/process content? They have role=process.
- Which nodes blend/composite multiple inputs?
- What is the final output node? It has role=output.

OUTPUT FORMAT (JSON only):
{{
  "reasoning": "1-2 sentences explaining the creative data flow",
  "connections": [
    {{"from": 0, "to": 3}},
    {{"from": 1, "to": 3}},
    {{"from": 2, "to": 4}},
    {{"from": 3, "to": 5}},
    {{"from": 4, "to": 5}}
  ]
}}

RULES:
- "from" and "to" are node indices [0..{len(nodes)-1}]
- Source nodes (generators, noise, inputs) have nothing flowing INTO them
- No cycles (A->B->A is invalid)
- Every non-source node must have at least one input
- A node CAN receive multiple inputs (for blending/compositing)
- Think about what visual result the brief demands

Output ONLY the JSON, no other text."""

        # Try LLM with fallback
        aider = get_aider_llm()
        models_to_try = [
            (self.model, "primary"),
            (self.fallback_model, "fallback"),
        ]

        for model, label in models_to_try:
            try:
                print(f"[CREATIVE TOPOLOGY] Trying {label} model for DAG design...")
                text = aider.call(system_prompt, model, think_tokens="4k")
                dag = self._parse_dag_response(text, len(nodes))

                if dag:
                    # Validate: at least 1 connection, no cycles
                    if self._validate_dag(dag, len(nodes)):
                        print(f"[CREATIVE TOPOLOGY] DAG designed with {label}: {len(dag)} connections")
                        return dag
                    else:
                        print(f"[CREATIVE TOPOLOGY] {label} DAG invalid (cycles or empty), trying next...")
                else:
                    print(f"[CREATIVE TOPOLOGY] {label} failed to parse DAG, trying next...")

            except Exception as e:
                print(f"[CREATIVE TOPOLOGY] {label} error: {e}")

        # All LLMs failed -> deterministic DAG fallback
        print("[CREATIVE TOPOLOGY] All LLMs failed, using deterministic DAG fallback")
        return self._fallback_dag(nodes)

    def _parse_dag_response(self, text: str, num_nodes: int) -> Optional[List[Dict]]:
        """Parse LLM response to extract DAG connections."""
        # Clean thinking tags
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

        # Try to extract JSON
        data = None

        # Direct parse
        try:
            data = json.loads(text)
        except Exception:
            pass

        # Code fence
        if not data:
            match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text, re.IGNORECASE)
            if match:
                try:
                    data = json.loads(match.group(1).strip())
                except Exception:
                    pass

        # First { to last }
        if not data:
            start = text.find('{')
            end = text.rfind('}')
            if start >= 0 and end > start:
                try:
                    data = json.loads(text[start:end+1])
                except Exception:
                    pass

        if not data or "connections" not in data:
            return None

        reasoning = data.get("reasoning", "")
        if reasoning:
            print(f"  [DAG REASONING] {reasoning}")

        # Parse connections
        raw_conns = data["connections"]
        dag = []
        for c in raw_conns:
            src = c.get("from")
            tgt = c.get("to")
            if src is None or tgt is None:
                continue
            try:
                src = int(src)
                tgt = int(tgt)
            except (ValueError, TypeError):
                continue
            if 0 <= src < num_nodes and 0 <= tgt < num_nodes and src != tgt:
                dag.append({"from": src, "to": tgt})

        return dag if dag else None

    def _validate_dag(self, dag: List[Dict], num_nodes: int) -> bool:
        """Validate DAG: at least 1 edge, no cycles."""
        if not dag:
            return False

        # Build adjacency for cycle detection
        adj: Dict[int, List[int]] = {i: [] for i in range(num_nodes)}
        for edge in dag:
            adj[edge["from"]].append(edge["to"])

        # DFS cycle detection
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {i: WHITE for i in range(num_nodes)}

        def has_cycle(node: int) -> bool:
            color[node] = GRAY
            for neighbor in adj[node]:
                if color[neighbor] == GRAY:
                    return True
                if color[neighbor] == WHITE and has_cycle(neighbor):
                    return True
            color[node] = BLACK
            return False

        for i in range(num_nodes):
            if color[i] == WHITE:
                if has_cycle(i):
                    print(f"  [DAG VALIDATE] Cycle detected!")
                    return False

        return True

    def _fallback_dag(self, nodes: List[Dict]) -> List[Dict]:
        """Build a deterministic DAG based on role classification.

        Sources -> processes (linear chain) -> outputs.
        """
        roles = self._classify_roles(nodes)
        sources = [i for i, r in roles.items() if r == "source"]
        processes = [i for i, r in roles.items() if r == "process"]
        outputs = [i for i, r in roles.items() if r == "output"]

        # Ensure at least one source and one output
        if not sources and processes:
            sources.append(processes.pop(0))
        if not outputs and processes:
            outputs.append(processes.pop())
        if not sources and not processes and not outputs:
            # Edge case: no nodes
            return []

        dag = []

        # Sources feed into first process (or output if no processes)
        if processes:
            # All sources feed into first process
            for src in sources:
                dag.append({"from": src, "to": processes[0]})

            # Linear chain through processes
            for i in range(len(processes) - 1):
                dag.append({"from": processes[i], "to": processes[i + 1]})

            # Last process(es) feed into outputs
            for out in outputs:
                dag.append({"from": processes[-1], "to": out})
        else:
            # No processes: sources feed directly into outputs
            for src in sources:
                for out in outputs:
                    dag.append({"from": src, "to": out})

        print(f"[CREATIVE TOPOLOGY] Fallback DAG: {len(sources)} sources, "
              f"{len(processes)} processes, {len(outputs)} outputs, "
              f"{len(dag)} edges")
        return dag

    def _wide_fan_dag(self, nodes: List[Dict]) -> List[Dict]:
        """Build a wide-fan DAG: all sources -> every process, all processes -> output(s).

        Creates maximum connectivity for high scoring.
        """
        roles = self._classify_roles(nodes)
        sources = [i for i, r in roles.items() if r == "source"]
        processes = [i for i, r in roles.items() if r == "process"]
        outputs = [i for i, r in roles.items() if r == "output"]

        if not sources and processes:
            sources.append(processes.pop(0))
        if not outputs and processes:
            outputs.append(processes.pop())
        if not sources and not processes and not outputs:
            return []

        dag = []
        existing = set()

        # All sources feed into every process
        for src in sources:
            for proc in processes:
                if (src, proc) not in existing:
                    dag.append({"from": src, "to": proc})
                    existing.add((src, proc))

        # Chain processes linearly too (for depth)
        for i in range(len(processes) - 1):
            pair = (processes[i], processes[i + 1])
            if pair not in existing:
                dag.append({"from": processes[i], "to": processes[i + 1]})
                existing.add(pair)

        # All processes feed into outputs
        for proc in processes:
            for out in outputs:
                if (proc, out) not in existing:
                    dag.append({"from": proc, "to": out})
                    existing.add((proc, out))

        # If no processes, sources -> outputs
        if not processes:
            for src in sources:
                for out in outputs:
                    if (src, out) not in existing:
                        dag.append({"from": src, "to": out})
                        existing.add((src, out))

        print(f"[CREATIVE TOPOLOGY] Wide-fan DAG: {len(sources)} sources, "
              f"{len(processes)} processes, {len(outputs)} outputs, "
              f"{len(dag)} edges")
        return dag

    def _force_output_to_end(self, dag: List[Dict], num_nodes: int,
                              node_roles: Dict[int, str]) -> List[Dict]:
        """Ensure output nodes are DAG sinks receiving from the deepest processes.

        - Remove edges where output feeds INTO another node
        - Connect output to deepest process node(s) if it has no inputs
        """
        outputs = {i for i in range(num_nodes) if node_roles.get(i) == "output"}
        if not outputs:
            return dag

        # Remove outgoing edges from output nodes
        cleaned = [e for e in dag if e["from"] not in outputs]
        removed = len(dag) - len(cleaned)
        if removed:
            print(f"  [FORCE OUTPUT] Removed {removed} outgoing edge(s) from output nodes")

        # Build reverse adj to check which outputs have inputs
        has_input = set()
        for e in cleaned:
            has_input.add(e["to"])

        # Compute topo depth for non-output nodes to find deepest processes
        reverse_adj = {}
        for e in cleaned:
            reverse_adj.setdefault(e["to"], []).append(e["from"])
        layout = DAGLayout()
        depths = layout._assign_z(reverse_adj, num_nodes)

        # Find deepest non-output nodes
        non_output_depths = {i: d for i, d in depths.items() if i not in outputs}
        if not non_output_depths:
            return cleaned

        max_depth = max(non_output_depths.values())
        deepest = [i for i, d in non_output_depths.items() if d == max_depth]

        existing = {(e["from"], e["to"]) for e in cleaned}

        for out in outputs:
            if out not in has_input:
                # Connect to deepest process(es)
                for d in deepest:
                    if (d, out) not in existing:
                        cleaned.append({"from": d, "to": out})
                        existing.add((d, out))
                        print(f"  [FORCE OUTPUT] Connected output {out} <- deepest node {d} (depth={max_depth})")

        return cleaned

    # =========================================================================
    # Semantic Brief Decomposition (JEPA-inspired)
    # =========================================================================

    def _decompose_brief(self, nodes: List[Dict], brief: str,
                         essence: str) -> Dict[int, Dict]:
        """Decompose the design brief into per-node semantic purposes.

        Like JEPA's latent predictions, each node gets a scoped "purpose" that
        describes what it should contribute to the overall visual. This replaces
        monolithic brief validation with focused, per-node semantic targets.

        One LLM call produces:
          - semantic_purpose: 1-2 sentence description of what this node contributes
          - role: "source" | "process" | "output" (data flow role)

        Returns:
            {node_index: {"purpose": str, "role": str}}
        """
        if not brief or not nodes:
            return {}

        # Format node list for the LLM
        node_list = []
        for i, node in enumerate(nodes):
            node_id = node.get("id", node.get("category", f"node_{i}"))
            category = node.get("category", "")
            keywords = ", ".join(node.get("keywords", []))
            node_list.append(
                f"  [{i}] {node_id} (category={category}, keywords={keywords})"
            )
        nodes_str = "\n".join(node_list)

        prompt = f"""You are a VFX Pipeline Architect decomposing a design brief into per-node responsibilities.

DESIGN BRIEF: "{brief}"
ESSENCE: "{essence}"

AVAILABLE NODES:
{nodes_str}

YOUR TASK: For each node, determine:
1. Its SEMANTIC PURPOSE — what specific visual/audio contribution it makes toward the brief.
   Think of this as the node's "job description" within the pipeline.
   Be specific to the brief, not generic. Example:
   - BAD: "generates noise" (too generic)
   - GOOD: "generates organic turbulence patterns that drive the particle flow motion"

2. Its ROLE in the data flow pipeline:
   - "source": generates content from nothing (noise, input, pattern generators)
   - "process": transforms/combines upstream data (blur, blend, distort, color grade)
   - "output": final compositing/display (composite, output, render, display)

OUTPUT FORMAT (JSON only):
{{
  "decomposition": [
    {{"index": 0, "purpose": "...", "role": "source"}},
    {{"index": 1, "purpose": "...", "role": "process"}},
    ...
  ]
}}

RULES:
- Every node must appear exactly once
- At least 1 node must be "source" and 1 must be "output"
- Purpose must reference the brief's specific visual intent, not generic descriptions
- Nodes that generate raw data (noise, patterns, inputs) are "source"
- Nodes that combine/transform (blur, blend, warp, color) are "process"
- The final compositing/output node is "output"
- A node like "point_cloud" generating 3D points IS a "source"

Output ONLY the JSON, no other text."""

        aider = get_aider_llm()
        models_to_try = [
            (self.model, "primary"),
            (self.fallback_model, "fallback"),
        ]

        for model, label in models_to_try:
            try:
                print(f"[SEMANTIC DECOMPOSE] Trying {label} model...")
                text = aider.call(prompt, model, think_tokens="4k")
                result = self._parse_decomposition(text, len(nodes))
                if result:
                    print(f"[SEMANTIC DECOMPOSE] Success with {label}: {len(result)} nodes decomposed")
                    for idx, info in sorted(result.items()):
                        print(f"  [{idx}] role={info['role']}: {info['purpose'][:80]}")
                    return result
            except Exception as e:
                print(f"[SEMANTIC DECOMPOSE] {label} error: {e}")

        # Fallback: keyword-based roles, generic purposes
        print("[SEMANTIC DECOMPOSE] LLM failed, using keyword fallback")
        return self._fallback_decomposition(nodes, brief)

    def _parse_decomposition(self, text: str, num_nodes: int) -> Optional[Dict[int, Dict]]:
        """Parse LLM decomposition response."""
        # Clean thinking tags
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

        data = None

        # Direct parse
        try:
            data = json.loads(text)
        except Exception:
            pass

        # Code fence
        if not data:
            match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text, re.IGNORECASE)
            if match:
                try:
                    data = json.loads(match.group(1).strip())
                except Exception:
                    pass

        # First { to last }
        if not data:
            start = text.find('{')
            end = text.rfind('}')
            if start >= 0 and end > start:
                try:
                    data = json.loads(text[start:end+1])
                except Exception:
                    pass

        if not data or "decomposition" not in data:
            return None

        result = {}
        valid_roles = {"source", "process", "output"}
        for entry in data["decomposition"]:
            idx = entry.get("index")
            purpose = entry.get("purpose", "")
            role = entry.get("role", "process").lower()
            if idx is None or not isinstance(idx, int):
                continue
            if idx < 0 or idx >= num_nodes:
                continue
            if role not in valid_roles:
                role = "process"
            result[idx] = {"purpose": purpose, "role": role}

        # Validate: must have at least 1 source and 1 output
        roles_present = {info["role"] for info in result.values()}
        if "source" not in roles_present or "output" not in roles_present:
            return None

        return result if len(result) == num_nodes else None

    def _fallback_decomposition(self, nodes: List[Dict], brief: str) -> Dict[int, Dict]:
        """Keyword-based fallback decomposition when LLM fails."""
        brief_snippet = brief[:60] if brief else "visual effect"
        result = {}
        for i, node in enumerate(nodes):
            role = self._classify_role_keywords(node)
            category = (node.get("category") or node.get("id") or "").replace("_", " ")
            if role == "source":
                purpose = f"Generate raw {category} data for: {brief_snippet}"
            elif role == "output":
                purpose = f"Composite and output the final result for: {brief_snippet}"
            else:
                purpose = f"Transform visual data via {category} for: {brief_snippet}"
            result[i] = {"purpose": purpose, "role": role}
        return result

    def _classify_roles(self, nodes: List[Dict]) -> Dict[int, str]:
        """Classify each node as source/process/output.

        Uses semantic_map (from LLM decomposition) when available,
        falls back to keyword matching.
        """
        roles = {}
        for i, node in enumerate(nodes):
            # Prefer LLM-decomposed role
            if i in self.semantic_map:
                roles[i] = self.semantic_map[i]["role"]
            else:
                roles[i] = self._classify_role_keywords(node)
        return roles

    def _classify_role_keywords(self, node: Dict) -> str:
        """Classify a single node's role using keyword matching (fallback)."""
        category = (node.get("category") or node.get("id") or "").lower()
        role = (node.get("role") or "").lower()

        if role == "input" or role == "source":
            return "source"
        if role == "output":
            return "output"

        # Check keywords
        for kw in INPUT_KEYWORDS:
            if kw in category:
                return "source"
        for kw in OUTPUT_KEYWORDS:
            if kw in category:
                return "output"

        return "process"
