"""Influence Compiler — InfluenceGraphIR -> BuildSheets + VolumetricGrid

Deterministic transform: no LLM calls. Reuses DAGLayout for positions.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import re
from typing import Dict, List, Tuple
from collections import defaultdict

from phase2.data_types import (
    InfluenceGraphIR, InfluenceNode, InfluenceEdge, BuildSheet,
    NodeTensor, TextureHandle, Connection, VolumetricGrid
)
from phase2.agents.dag_layout import DAGLayout

# Protocol descriptions for Mason prompts
PROTOCOL_DESCRIPTIONS = {
    "DENSITY_RGBA": "R=density (0-1), G=age/trail (0-1), B=noise (0-1), A=mask (0=transparent, 1=opaque)",
    "COLOR_RGBA": "Standard RGBA: RGB=color channels, A=alpha",
    "IMAGE_RGBA": "Camera/image feed: standard RGBA pixel data",
    "AUDIO_FFT": ("Frequency data. In GLSL: access via u_audio_level (0-1 overall), "
                  "u_audio_bass (0-1 low freq), u_audio_mid, u_audio_high uniforms. "
                  "In JS: read window._nodeData['audio'] = {level, bass, mid, high, fftData}"),
    "LANDMARKS": ("Tracking keypoints. In JS: read window._nodeData['tracking'] = "
                  "{trackingData, type, count, keypoints}. Each keypoint has {x, y, confidence}. "
                  "In GLSL: tracking data not directly available — use a data_drive node to convert"),
    "DATA_JSON": "Structured JSON data. In JS: read window._nodeData[sourceNodeId]",
}


class InfluenceCompiler:
    """Compiles InfluenceGraphIR into BuildSheets + VolumetricGrid."""

    def compile(self, ir: InfluenceGraphIR, archetypes: List[Dict]) -> Tuple[List[BuildSheet], VolumetricGrid]:
        # Step 1: Build index maps
        node_map = {n.id: n for n in ir.nodes}
        id_to_idx = {n.id: i for i, n in enumerate(ir.nodes)}

        # Step 2: DAG layout for positions
        dag = []
        for e in ir.edges:
            fi = id_to_idx.get(e.from_node, -1)
            ti = id_to_idx.get(e.to_node, -1)
            if fi >= 0 and ti >= 0:
                dag.append({"from": fi, "to": ti})

        layout = DAGLayout()
        topology = layout.layout(dag, len(ir.nodes))
        topo_nodes = topology.get("nodes", [])

        # Step 3: Build NodeTensors
        node_tensors = []
        pos_map = {}  # node_id -> (x, y, z)
        for tn in topo_nodes:
            idx = tn.get("node_index", tn.get("archetype_index", 0))
            if idx < 0 or idx >= len(ir.nodes):
                continue
            ir_node = ir.nodes[idx]
            pos = tuple(tn.get("grid_position", [0, 0, 0]))
            pos_map[ir_node.id] = pos

            # Get archetype metadata if available
            arch = archetypes[idx] if idx < len(archetypes) else {}
            meta = ir_node.meta or arch.get("meta", {})
            if not isinstance(meta, dict):
                meta = {}

            # Ensure meta has required fields
            meta.setdefault("concept_id", arch.get("id", f"arch_{idx}"))
            meta.setdefault("label", arch.get("name", ir_node.id))
            meta.setdefault("level", "surface")
            meta.setdefault("modality", "texture")
            meta.setdefault("domain", "visual")
            meta.setdefault("description", ir_node.intent)
            meta.setdefault("category", ir_node.category)
            meta.setdefault("role", ir_node.role)
            meta.setdefault("keywords", ir_node.keywords)

            # Collect input node IDs from edges
            input_ids = [e.from_node for e in ir.edges if e.to_node == ir_node.id]

            # Parameters from archetype
            raw_params = arch.get("parameters", {})
            if isinstance(raw_params, list):
                params = {}
                for p in raw_params:
                    if isinstance(p, dict):
                        params[p.get("name", "p")] = p.get("default", 0)
                    elif isinstance(p, str):
                        params[p] = 0
            elif isinstance(raw_params, dict):
                params = raw_params
            else:
                params = {}

            texture = TextureHandle(node_id=ir_node.id, z_layer=pos[2]) if ir_node.engine not in ("webaudio", "events") else None

            nt = NodeTensor(
                id=ir_node.id,
                meta=meta,
                grid_position=pos,
                grid_size=tuple(tn.get("grid_size", [1, 1])),
                engine=ir_node.engine,
                code_snippet="",
                parameters=params,
                input_nodes=input_ids,
                output_texture=texture,
                keywords=ir_node.keywords,
                semantic_purpose=ir_node.intent,
            )
            node_tensors.append(nt)

        # Step 4: Build legacy connections
        connections = []
        for e in ir.edges:
            # Find input index
            target_inputs = [ee.from_node for ee in ir.edges if ee.to_node == e.to_node]
            input_idx = target_inputs.index(e.from_node) if e.from_node in target_inputs else 0
            connections.append(e.to_legacy_connection(input_idx))

        # Compute grid dimensions
        max_x = max((n.grid_position[0] for n in node_tensors), default=0) + 1
        max_y = max((n.grid_position[1] for n in node_tensors), default=0) + 1
        max_z = max((n.grid_position[2] for n in node_tensors), default=0) + 1

        grid = VolumetricGrid(
            dimensions=(max_x, max_y, max_z),
            nodes=node_tensors,
            connections=connections,
            runtime_hints={
                "essence": ir.reasoning,
                "total_z_layers": max_z,
            },
        )

        # Step 5: Build BuildSheets (with Z-axis context)
        build_sheets = self._build_sheets(ir, pos_map, max_z)

        print(f"  [COMPILER] {len(build_sheets)} build sheets, "
              f"{len(node_tensors)} nodes, {len(connections)} connections, "
              f"{max_z} Z-layers")

        return build_sheets, grid

    def _infer_z_role(self, z: int, z_total: int, has_inputs: bool, has_outputs: bool) -> str:
        """Infer semantic role from Z position in the grid."""
        if z_total <= 1:
            return "standalone"
        frac = z / max(z_total - 1, 1)
        if not has_inputs:
            return "source"
        if not has_outputs:
            return "output"
        if frac < 0.25:
            return "source"
        if frac > 0.75:
            return "compositor"
        return "processor"

    def _build_sheets(self, ir: InfluenceGraphIR, pos_map: Dict[str, tuple], z_total: int) -> List[BuildSheet]:
        node_map = {n.id: n for n in ir.nodes}
        edges_by_target = defaultdict(list)
        edges_by_source = defaultdict(list)
        for e in ir.edges:
            edges_by_target[e.to_node].append(e)
            edges_by_source[e.from_node].append(e)

        palette = ir.global_context.get("palette", [])
        motion = ir.global_context.get("motion", [])

        sheets = []
        for node in ir.nodes:
            incoming = edges_by_target.get(node.id, [])
            outgoing = edges_by_source.get(node.id, [])

            # Build input descriptors with source Z-position
            inputs = []
            merged_rules = {"must_use": [], "preserve": [], "allow": [], "avoid": []}

            for i, edge in enumerate(incoming):
                source = node_map.get(edge.from_node)
                input_name = f"u_input{i}" if node.engine in ("glsl", "regl") else f"input_{i}"
                desc = PROTOCOL_DESCRIPTIONS.get(edge.protocol, edge.protocol)
                src_pos = pos_map.get(edge.from_node, (0, 0, 0))
                inputs.append({
                    "name": input_name,
                    "protocol": edge.protocol,
                    "meaning": ", ".join(edge.influence.must_use) if edge.influence.must_use else desc,
                    "source_id": edge.from_node,
                    "source_intent": source.intent if source else "",
                    "source_z": src_pos[2],
                })

                merged_rules["must_use"].extend(edge.influence.must_use)
                merged_rules["preserve"].extend(edge.influence.preserve)
                merged_rules["allow"].extend(edge.influence.allow)
                merged_rules["avoid"].extend(edge.influence.avoid)

            # Style anchor from global context + node-level overrides
            style = {}
            if palette:
                style["palette"] = palette
            if motion:
                style["motion"] = motion
            if node.style_anchor:
                style.update(node.style_anchor)

            # Z-axis context
            pos = pos_map.get(node.id, (0, 0, 0))
            z_role = self._infer_z_role(pos[2], z_total, bool(incoming), bool(outgoing))

            sheets.append(BuildSheet(
                node_id=node.id,
                engine=node.engine,
                intent=node.intent,
                inputs=inputs,
                influence_rules=merged_rules,
                output_protocol=node.output_protocol,
                style_anchor=style,
                params=node.suggested_params,
                grid_position=pos,
                z_total=z_total,
                z_role=z_role,
            ))

        return sheets
