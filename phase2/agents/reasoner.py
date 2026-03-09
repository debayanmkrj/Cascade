"""Reasoner Agent — Session JSON -> InfluenceGraphIR

Single LLM call produces the complete influence graph: nodes with intents,
typed edges with contracts (must_use/preserve/allow/avoid), channel protocols.
Replaces creative_topology + graph_designer + node_affinity.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import re
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from config import MODEL_NAME_REASONING, MODEL_NAME_FALLBACK
from phase2.aider_llm import get_aider_llm
from phase2.data_types import (
    InfluenceGraphIR, InfluenceNode, InfluenceEdge,
    InfluenceContract, InfluenceType, ChannelProtocol
)
from phase2.agents.architect import infer_engine_from_category, ENGINE_CAN_FEED

# Protocol inference from engine type
ENGINE_DEFAULT_PROTOCOL = {
    "glsl": "COLOR_RGBA",
    "regl": "COLOR_RGBA",
    "canvas2d": "COLOR_RGBA",
    "three_js": "COLOR_RGBA",
    "html_video": "IMAGE_RGBA",
    "webaudio": "AUDIO_FFT",
    "events": "DATA_JSON",
}

# Category-specific protocol overrides (predefined nodes with known semantics)
CATEGORY_PROTOCOL = {
    "p5_tracking": "LANDMARKS",
    "audio_input": "AUDIO_FFT",
    "video_input": "IMAGE_RGBA",
    "webcam_input": "IMAGE_RGBA",
    "image_input": "IMAGE_RGBA",
    "noise_generator": "DENSITY_RGBA",
    "noise_perlin": "DENSITY_RGBA",
    "noise_worley": "DENSITY_RGBA",
    "noise_simplex": "DENSITY_RGBA",
    "noise_fbm": "DENSITY_RGBA",
}

# Edge type inference from engine pairs + protocol
def _infer_influence_type(src_engine: str, tgt_engine: str, src_role: str,
                          src_protocol: str = "") -> str:
    if src_engine == "webaudio" or src_engine == "events":
        return "data_drive"
    if src_protocol in ("AUDIO_FFT", "LANDMARKS", "DATA_JSON"):
        return "data_drive"
    if src_role == "input":
        return "mask_and_emit"
    if src_protocol == "DENSITY_RGBA":
        return "mask_and_emit"
    if src_engine == tgt_engine == "glsl":
        return "composite"
    return "composite"

VALID_INFLUENCE_TYPES = {"mask_and_emit", "warp_only", "color_grade", "composite", "feedback_trails", "data_drive"}
VALID_PROTOCOLS = {"DENSITY_RGBA", "COLOR_RGBA", "IMAGE_RGBA", "AUDIO_FFT", "LANDMARKS", "DATA_JSON"}


class ReasonerAgent:
    """Produces InfluenceGraphIR from Phase 1 session JSON."""

    def __init__(self):
        self.aider = get_aider_llm()

    def design(self, session_json: Dict) -> InfluenceGraphIR:
        phase2_ctx = session_json.get('phase2_context', {})
        archetypes = phase2_ctx.get('node_archetypes', [])
        brief = (
            session_json.get('input', {}).get('prompt_text', '')
            or session_json.get('brief', {}).get('essence', '')
            or phase2_ctx.get('essence', '')
            or ''
        )
        visual_palette = (
            session_json.get('visual_palette')
            or phase2_ctx.get('visual_palette')
            or session_json.get('brief', {}).get('visual_palette')
            or {}
        )

        if not archetypes:
            return InfluenceGraphIR(global_context={"brief": brief}, reasoning="No archetypes")

        print(f"  [REASONER] Designing influence graph for {len(archetypes)} nodes...")

        # Build global context
        global_ctx = {
            "brief": brief,
            "palette": visual_palette.get('primary_colors', []),
            "accent": visual_palette.get('accent_colors', []),
            "shapes": visual_palette.get('shapes', []),
            "motion": visual_palette.get('motion_words', []),
        }

        # Try LLM, fall back to deterministic
        ir = self._llm_design(archetypes, brief, global_ctx)
        if ir and self._validate(ir, len(archetypes)):
            ir.global_context = global_ctx
            print(f"  [REASONER] LLM produced {len(ir.nodes)} nodes, {len(ir.edges)} edges")
            return ir

        print("  [REASONER] LLM failed, using deterministic fallback")
        return self._deterministic_design(archetypes, global_ctx)

    def _llm_design(self, archetypes: List[Dict], brief: str, global_ctx: Dict) -> Optional[InfluenceGraphIR]:
        node_summary = "\n".join([
            f"  [{i}] {a.get('name', f'n{i}')} category={a.get('category', 'effect')} "
            f"engine={a.get('engine') or a.get('meta', {}).get('engine_hint', 'auto')} "
            f"role={a.get('role', 'process')} keywords={a.get('keywords', a.get('meta', {}).get('keywords', []))}"
            for i, a in enumerate(archetypes)
        ])

        palette_str = ", ".join(global_ctx.get("palette", [])) or "auto"

        prompt = f"""You are a VFX Pipeline Architect. Design the influence graph for this visual system.

BRIEF: "{brief[:300]}"
PALETTE: {palette_str}

NODES:
{node_summary}

For each node, specify: engine, role (source/process/output), intent (1 sentence), output_protocol, params list.
For each edge, specify: from/to indices, influence_type, must_use, preserve, allow, avoid.

TOPOLOGY RULES (CRITICAL):
- Create PARALLEL BRANCHES, not a single linear chain
- Sources (input/generator nodes) should fan out to MULTIPLE downstream nodes
- Process nodes that are independent should be at the SAME depth (parallel)
- Only nodes that truly depend on each other should be in sequence
- An output/compositor node should merge multiple branches
- Example for 6 nodes: 0->2, 1->2, 0->3, 2->4, 3->4, 4->5 (fan-out + merge)
- BAD topology: 0->1->2->3->4->5 (pure linear chain — NEVER do this)

INFLUENCE TYPES (pick one per edge):
- mask_and_emit: source density/mask drives target emission
- warp_only: source displaces target UVs
- color_grade: source provides color reference
- composite: standard alpha-over compositing
- feedback_trails: temporal accumulation
- data_drive: non-visual data driving visual params

PROTOCOLS (pick one per node output):
- DENSITY_RGBA (R=density, G=age, B=noise, A=mask)
- COLOR_RGBA (standard RGB + alpha)
- IMAGE_RGBA (camera/image feed)
- AUDIO_FFT (frequency data)
- LANDMARKS (tracking keypoints)
- DATA_JSON (structured data)

ENGINES: glsl, canvas2d, three_js, webaudio, html_video, events

Return ONLY JSON:
{{
  "reasoning": "brief explanation",
  "nodes": [
    {{"index": 0, "engine": "glsl", "role": "source", "intent": "...", "output_protocol": "COLOR_RGBA", "params": ["speed", "scale"]}}
  ],
  "edges": [
    {{"from": 0, "to": 1, "influence_type": "composite", "must_use": ["sample input texture"], "preserve": ["color identity"], "allow": ["blur"], "avoid": ["ignoring input"]}}
  ]
}}"""

        for model in [MODEL_NAME_REASONING, MODEL_NAME_FALLBACK]:
            try:
                text = self.aider.call(prompt, model)
                if not text:
                    continue
                ir = self._parse_response(text, archetypes)
                if ir:
                    return ir
            except Exception as e:
                print(f"  [REASONER] Model {model} failed: {e}")
        return None

    def _parse_response(self, text: str, archetypes: List[Dict]) -> Optional[InfluenceGraphIR]:
        # Extract JSON from response
        match = re.search(r'\{[\s\S]*\}', text)
        if not match:
            return None
        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            return None

        n = len(archetypes)
        nodes = []
        for nd in data.get("nodes", []):
            idx = nd.get("index", 0)
            if idx < 0 or idx >= n:
                continue
            arch = archetypes[idx]
            arch_engine = arch.get("engine") or arch.get("meta", {}).get("engine_hint")
            engine = nd.get("engine", arch_engine or infer_engine_from_category(arch.get("category", "effect")))
            role = nd.get("role", arch.get("role", "process"))
            cat = arch.get("category", "effect")
            protocol = nd.get("output_protocol",
                              CATEGORY_PROTOCOL.get(cat, ENGINE_DEFAULT_PROTOCOL.get(engine, "COLOR_RGBA")))
            if protocol not in VALID_PROTOCOLS:
                protocol = CATEGORY_PROTOCOL.get(cat, ENGINE_DEFAULT_PROTOCOL.get(engine, "COLOR_RGBA"))

            nodes.append(InfluenceNode(
                id=f"node_{idx}_{re.sub(r'[^a-zA-Z0-9_]', '_', arch.get('name', f'n{idx}').lower())}",
                engine=engine,
                role=role,
                intent=nd.get("intent", arch.get("meta", {}).get("description", "")),
                output_protocol=protocol,
                category=arch.get("category", ""),
                keywords=arch.get("keywords", arch.get("meta", {}).get("keywords", [])),
                suggested_params=nd.get("params", []),
                meta=arch.get("meta", {}),
            ))

        if len(nodes) != n:
            # Fill missing nodes
            covered = {nd.get("index", -1) for nd in data.get("nodes", [])}
            for i, arch in enumerate(archetypes):
                if i not in covered:
                    arch_engine = arch.get("engine") or arch.get("meta", {}).get("engine_hint")
                    engine = arch_engine or infer_engine_from_category(arch.get("category", "effect"))
                    nodes.append(InfluenceNode(
                        id=f"node_{i}_{re.sub(r'[^a-zA-Z0-9_]', '_', arch.get('name', f'n{i}').lower())}",
                        engine=engine,
                        role=arch.get("role", "process"),
                        intent=arch.get("meta", {}).get("description", ""),
                        output_protocol=ENGINE_DEFAULT_PROTOCOL.get(engine, "COLOR_RGBA"),
                        category=arch.get("category", ""),
                        keywords=arch.get("keywords", []),
                        meta=arch.get("meta", {}),
                    ))
            # Sort by original index
            nodes.sort(key=lambda nd: int(re.search(r'node_(\d+)_', nd.id).group(1)) if re.search(r'node_(\d+)_', nd.id) else 0)

        edges = []
        node_ids = [nd.id for nd in nodes]
        for ed in data.get("edges", []):
            f, t = ed.get("from", -1), ed.get("to", -1)
            if f < 0 or f >= len(node_ids) or t < 0 or t >= len(node_ids) or f == t:
                continue
            inf_type = ed.get("influence_type", "composite")
            if inf_type not in VALID_INFLUENCE_TYPES:
                inf_type = "composite"

            edges.append(InfluenceEdge(
                from_node=node_ids[f],
                to_node=node_ids[t],
                protocol=nodes[f].output_protocol,
                influence=InfluenceContract(
                    influence_type=inf_type,
                    must_use=ed.get("must_use", []),
                    preserve=ed.get("preserve", []),
                    allow=ed.get("allow", []),
                    avoid=ed.get("avoid", []),
                ),
            ))

        if not edges:
            return None

        # Wire orphan nodes: every non-source must have at least one input edge
        # Strategy: connect orphans to source/input nodes to create parallel branches
        # (not to i-1, which creates linear chains)
        target_ids = {e.to_node for e in edges}
        source_ids = {e.from_node for e in edges}
        source_indices = [j for j, n in enumerate(nodes) if n.role in ("input", "source")]
        if not source_indices:
            # No explicit sources — use nodes that are only sources (appear in from but not to)
            source_indices = [j for j, n in enumerate(nodes) if n.id in source_ids and n.id not in target_ids]
        if not source_indices:
            source_indices = [0]  # Last resort

        orphan_count = 0
        for i, node in enumerate(nodes):
            if node.id in target_ids or node.role in ("input", "source"):
                continue
            # Round-robin across available sources for fan-out
            best_src = source_indices[orphan_count % len(source_indices)]
            if best_src == i:
                best_src = source_indices[(orphan_count + 1) % len(source_indices)]
            if best_src != i:
                edges.append(self._make_edge_from_nodes(nodes, best_src, i))
                orphan_count += 1

        return InfluenceGraphIR(
            global_context={},
            nodes=nodes,
            edges=edges,
            reasoning=data.get("reasoning", ""),
        )

    def _deterministic_design(self, archetypes: List[Dict], global_ctx: Dict) -> InfluenceGraphIR:
        """Fallback: build IR from categories and engine types without LLM.

        Creates BRANCHING topology — sources fan out to parallel process branches,
        which merge at compositor/output nodes. Avoids pure linear chains.
        """
        brief = global_ctx.get("brief", "")
        nodes = []
        sources, processes, outputs = [], [], []

        for i, arch in enumerate(archetypes):
            cat = arch.get("category", "effect")
            arch_engine = arch.get("engine") or arch.get("meta", {}).get("engine_hint")
            engine = arch_engine or infer_engine_from_category(cat, arch.get("keywords", []))
            role = arch.get("role", "process")
            # Use category-specific protocol if available, else engine default
            protocol = CATEGORY_PROTOCOL.get(cat, ENGINE_DEFAULT_PROTOCOL.get(engine, "COLOR_RGBA"))
            # Also check meta for output_protocol override
            meta_proto = arch.get("meta", {}).get("output_protocol")
            if meta_proto and meta_proto in VALID_PROTOCOLS:
                protocol = meta_proto
            name = arch.get("name", f"n{i}")
            safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', name.lower())

            # Use description from SemanticReasoner if available
            # If it's just generic "{cat} node", enrich with brief context
            desc = arch.get("meta", {}).get("description", "")
            if not desc or desc == f"{cat.replace('_', ' ')} node":
                cat_label = cat.replace('_', ' ')
                desc = f"{cat_label} for: {brief[:80]}" if brief else f"{cat_label} node"

            node = InfluenceNode(
                id=f"node_{i}_{safe_name}",
                engine=engine,
                role=role,
                intent=desc,
                output_protocol=protocol,
                category=cat,
                keywords=arch.get("keywords", arch.get("meta", {}).get("keywords", [])),
                suggested_params=[],
                meta=arch.get("meta", {}),
            )
            nodes.append(node)

            if role == "input":
                sources.append(i)
            elif role == "output":
                outputs.append(i)
            else:
                processes.append(i)

        # Ensure at least one source and one output
        if not outputs and processes:
            outputs.append(processes.pop())
            nodes[outputs[0]].role = "output"
        if not sources and processes:
            sources.insert(0, processes.pop(0))
            nodes[sources[0]].role = "input"

        edges = []

        if not processes:
            # No process nodes: direct source->output
            for s in sources:
                for o in outputs:
                    edges.append(self._make_edge(nodes, s, o))
        elif len(processes) <= 2:
            # Few processes: simple fan-out from sources, merge at output
            for s in sources:
                for p in processes:
                    edges.append(self._make_edge(nodes, s, p))
            for p in processes:
                for o in outputs:
                    edges.append(self._make_edge(nodes, p, o))
        else:
            # Many processes: create parallel branches
            # Split processes into branches based on category similarity
            # Each source fans out to a subset of processes
            # Processes share a merger layer before outputs

            # Group processes by engine type to form natural branches
            branch_groups = defaultdict(list)
            for p in processes:
                eng = nodes[p].engine
                branch_groups[eng].append(p)

            # If all same engine, split by position (first half / second half)
            if len(branch_groups) <= 1:
                branch_groups = {}
                mid = len(processes) // 2
                branch_groups["branch_a"] = processes[:mid]
                branch_groups["branch_b"] = processes[mid:]

            branches = list(branch_groups.values())

            # Sources fan out to the first node of each branch
            for s in sources:
                for branch in branches:
                    edges.append(self._make_edge(nodes, s, branch[0]))

            # Within each branch, chain sequentially
            for branch in branches:
                for j in range(len(branch) - 1):
                    edges.append(self._make_edge(nodes, branch[j], branch[j + 1]))

            # Last node of each branch feeds into outputs (merge point)
            for branch in branches:
                last = branch[-1]
                for o in outputs:
                    edges.append(self._make_edge(nodes, last, o))

            # Cross-branch connections: if a source node produces data (audio/events),
            # it should also feed visual nodes as data_drive
            for s in sources:
                if nodes[s].engine in ("webaudio", "events"):
                    for branch in branches:
                        for p in branch:
                            if nodes[p].engine not in ("webaudio", "events") and p != branch[0]:
                                # Don't duplicate the initial fan-out edge
                                edges.append(self._make_edge(nodes, s, p))
                                break  # One data_drive per branch

        reasoning_parts = []
        if sources: reasoning_parts.append(f"{len(sources)} sources")
        reasoning_parts.append(f"{len(processes)} process nodes")
        if outputs: reasoning_parts.append(f"{len(outputs)} outputs")
        branch_count = max(1, len(set(nodes[p].engine for p in processes))) if processes else 0
        reasoning_parts.append(f"~{branch_count} branches")

        return InfluenceGraphIR(
            global_context=global_ctx,
            nodes=nodes,
            edges=edges,
            reasoning=f"Deterministic branching: {' -> '.join(reasoning_parts)}",
        )

    def _make_edge_from_nodes(self, nodes: List[InfluenceNode], fi: int, ti: int) -> InfluenceEdge:
        """Create edge between nodes by index (used for orphan wiring)."""
        return self._make_edge(nodes, fi, ti)

    def _make_edge(self, nodes: List[InfluenceNode], fi: int, ti: int) -> InfluenceEdge:
        src, tgt = nodes[fi], nodes[ti]
        inf_type = _infer_influence_type(src.engine, tgt.engine, src.role, src.output_protocol)

        # Build must_use based on influence type and protocol
        if inf_type == "data_drive":
            if src.output_protocol == "LANDMARKS":
                must_use = [f"read tracking data from {src.id}", "use keypoints to drive visual parameters"]
            elif src.output_protocol == "AUDIO_FFT":
                must_use = [f"read audio data from {src.id}", "use audio level/bass/mid to drive visual parameters"]
            else:
                must_use = [f"read {src.id} data"]
        elif src.output_protocol == "DENSITY_RGBA":
            must_use = [f"sample {src.id} texture", "use R channel as density mask"]
        else:
            must_use = [f"sample {src.id} texture"]

        return InfluenceEdge(
            from_node=src.id,
            to_node=tgt.id,
            protocol=src.output_protocol,
            influence=InfluenceContract(
                influence_type=inf_type,
                must_use=must_use,
                preserve=[],
                allow=[],
                avoid=["ignoring input"],
            ),
        )

    def _validate(self, ir: InfluenceGraphIR, expected_count: int) -> bool:
        if len(ir.nodes) < 2 or not ir.edges:
            return False
        # Require at least N-1 edges for N nodes (connected graph)
        if len(ir.edges) < len(ir.nodes) - 1:
            print(f"  [REASONER] Too few edges: {len(ir.edges)} for {len(ir.nodes)} nodes")
            return False
        # Check for cycles
        adj = defaultdict(list)
        for e in ir.edges:
            adj[e.from_node].append(e.to_node)
        visited, rec = set(), set()
        def dfs(nid):
            visited.add(nid)
            rec.add(nid)
            for child in adj.get(nid, []):
                if child not in visited:
                    if dfs(child):
                        return True
                elif child in rec:
                    return True
            rec.discard(nid)
            return False
        for n in ir.nodes:
            if n.id not in visited:
                if dfs(n.id):
                    print("  [REASONER] Cycle detected — rejecting")
                    return False
        return True
