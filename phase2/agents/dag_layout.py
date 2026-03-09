"""
DAG Layout Engine
-----------------
Pure-algorithm module (no LLM). Converts a connection DAG into volumetric
XYZ positions using topological depth (Z) and barycentric optimization (X).

Algorithm:
1. Topological sort -> Z = longest path from any source
2. Barycentric X optimization -> minimize edge crossing
3. Y = 0 (unused for now)

Pipeline position: Called by creative_topology.py after LLM designs DAG.
"""

from typing import Dict, List, Set, Optional
from collections import defaultdict


class DAGLayout:
    """Converts a connection DAG into volumetric XYZ positions."""

    def layout(self, dag: List[Dict], num_nodes: int) -> Dict:
        """Main entry: DAG -> topology dict with positions.

        Args:
            dag: [{"from": 0, "to": 3}, {"from": 1, "to": 3}, ...]
            num_nodes: total number of nodes

        Returns:
            topology dict with nodes having grid_position and input_from
        """
        # Build adjacency (forward: parent->children) and reverse (child->parents)
        adjacency = defaultdict(list)     # node -> list of children
        reverse_adj = defaultdict(list)   # node -> list of parents (inputs)

        for edge in dag:
            src = edge["from"]
            tgt = edge["to"]
            if src != tgt and 0 <= src < num_nodes and 0 <= tgt < num_nodes:
                if tgt not in adjacency[src]:
                    adjacency[src].append(tgt)
                if src not in reverse_adj[tgt]:
                    reverse_adj[tgt].append(src)

        # Ensure all nodes appear in dicts
        for i in range(num_nodes):
            if i not in adjacency:
                adjacency[i] = []
            if i not in reverse_adj:
                reverse_adj[i] = []

        # Step 1: Assign Z via longest-path topological sort
        z_assignments = self._assign_z(reverse_adj, num_nodes)
        print(f"  [DAG LAYOUT] Z assignment: {z_assignments}")

        # Step 2: Assign X via barycentric optimization
        x_positions = self._assign_x(z_assignments, adjacency, reverse_adj, num_nodes)
        print(f"  [DAG LAYOUT] X assignment: {x_positions}")

        # Step 3: Build positions
        positions = {}
        for i in range(num_nodes):
            positions[i] = (x_positions.get(i, 0), 0, z_assignments.get(i, 0))

        # Build topology dict (same format as creative_topology output)
        max_z = max(z_assignments.values()) if z_assignments else 0

        # Build input_from from the DAG
        input_from_map = defaultdict(list)
        for edge in dag:
            src = edge["from"]
            tgt = edge["to"]
            if src != tgt and 0 <= src < num_nodes and 0 <= tgt < num_nodes:
                if src not in input_from_map[tgt]:
                    input_from_map[tgt].append(src)

        topology_nodes = []
        for i in range(num_nodes):
            pos = positions.get(i, (0, 0, 0))
            topology_nodes.append({
                "node_index": i,
                "grid_position": list(pos),
                "input_from": input_from_map.get(i, []),
            })

        print(f"  [DAG LAYOUT] Final: {num_nodes} nodes across {max_z + 1} Z-layers")
        return {
            "nodes": topology_nodes,
            "max_z": max_z,
        }

    def _assign_z(self, reverse_adj: Dict[int, List[int]], num_nodes: int) -> Dict[int, int]:
        """Assign Z = longest path from any source (topological depth).

        Sources (no inputs) get Z=0.
        Everything else gets Z = max(Z of parents) + 1.
        Guaranteed to produce non-flat layouts when DAG has depth.
        """
        depths: Dict[int, int] = {}

        def get_depth(node: int, seen: Optional[Set[int]] = None) -> int:
            if seen is None:
                seen = set()
            if node in depths:
                return depths[node]
            if node in seen:
                return 0  # cycle detected, break it
            seen.add(node)

            parents = reverse_adj.get(node, [])
            if not parents:
                depths[node] = 0
                return 0

            d = max(get_depth(p, seen) for p in parents) + 1
            depths[node] = d
            return d

        for i in range(num_nodes):
            get_depth(i)

        return depths

    def _assign_x(self, z_assignments: Dict[int, int],
                   adjacency: Dict[int, List[int]],
                   reverse_adj: Dict[int, List[int]],
                   num_nodes: int) -> Dict[int, int]:
        """Assign X via barycentric optimization (Sugiyama-style).

        1. Initial spread: nodes at each Z get X = 0, 1, 2, ...
        2. 3 passes: X_new(v) = average of parent + child X positions
        3. Sort by X, assign integer slots (collision avoidance)
        """
        # Group nodes by Z layer
        layers: Dict[int, List[int]] = defaultdict(list)
        for i in range(num_nodes):
            z = z_assignments.get(i, 0)
            layers[z].append(i)

        # Initialize X positions (simple spread)
        x_positions: Dict[int, float] = {}
        for z in sorted(layers.keys()):
            nodes_at_z = layers[z]
            for slot, idx in enumerate(nodes_at_z):
                x_positions[idx] = float(slot)

        # Iterative barycentric refinement (3 passes)
        for _iteration in range(3):
            new_x: Dict[int, float] = {}

            for z in sorted(layers.keys()):
                for idx in layers[z]:
                    parent_xs = [x_positions[p] for p in reverse_adj.get(idx, [])
                                 if p in x_positions]
                    child_xs = [x_positions[c] for c in adjacency.get(idx, [])
                                if c in x_positions]

                    all_xs = parent_xs + child_xs
                    if all_xs:
                        new_x[idx] = sum(all_xs) / len(all_xs)
                    else:
                        new_x[idx] = x_positions[idx]

            # Apply with collision avoidance: sort by new X, assign integer slots
            for z in sorted(layers.keys()):
                nodes_at_z = layers[z]
                sorted_nodes = sorted(nodes_at_z, key=lambda i: new_x.get(i, 0))
                for slot, idx in enumerate(sorted_nodes):
                    x_positions[idx] = float(slot)

        return {k: int(v) for k, v in x_positions.items()}

    @staticmethod
    def ensure_dag_complete(dag: List[Dict], num_nodes: int,
                            node_roles: Dict[int, str]) -> List[Dict]:
        """Fill orphan connections to ensure every non-source has at least 1 input.

        Args:
            dag: existing edges [{"from": int, "to": int}, ...]
            num_nodes: total nodes
            node_roles: {node_index: "source"|"process"|"output"}

        Returns:
            Completed DAG edges
        """
        # Build reverse adj
        has_input = set()
        has_output = set()
        for edge in dag:
            has_input.add(edge["to"])
            has_output.add(edge["from"])

        # Classify by role
        sources = [i for i in range(num_nodes) if node_roles.get(i) == "source"]
        outputs = [i for i in range(num_nodes) if node_roles.get(i) == "output"]

        # If no sources, treat nodes with no inputs as sources
        if not sources:
            sources = [i for i in range(num_nodes) if i not in has_input]

        new_edges = list(dag)
        existing_pairs = {(e["from"], e["to"]) for e in dag}

        # Find orphans: non-source nodes with no inputs
        for i in range(num_nodes):
            role = node_roles.get(i, "process")
            if role == "source":
                continue
            if i in has_input:
                continue

            # Prefer process nodes that already have outputs (creates deeper chains),
            # then sources as fallback
            processes_with_output = [j for j in range(num_nodes)
                                     if j != i and j in has_output
                                     and node_roles.get(j) in ("process", "source")]
            candidates = processes_with_output if processes_with_output else sources
            if not candidates:
                candidates = [j for j in range(num_nodes) if j != i]
            if not candidates:
                continue

            for cand in candidates:
                if cand != i and (cand, i) not in existing_pairs:
                    new_edges.append({"from": cand, "to": i})
                    existing_pairs.add((cand, i))
                    has_input.add(i)
                    has_output.add(cand)
                    print(f"    [DAG COMPLETE] Connected orphan {i} <- {cand}")
                    break

        # Ensure outputs have inputs if they don't
        for out_idx in outputs:
            if out_idx in has_input:
                continue
            # Connect to nearest process or source
            processes = [i for i in range(num_nodes)
                         if node_roles.get(i) == "process" and i != out_idx]
            candidates = processes if processes else sources
            for cand in candidates:
                if cand != out_idx and (cand, out_idx) not in existing_pairs:
                    new_edges.append({"from": cand, "to": out_idx})
                    existing_pairs.add((cand, out_idx))
                    print(f"    [DAG COMPLETE] Connected output {out_idx} <- {cand}")
                    break

        return new_edges


class DAGScorer:
    """Score a DAG by quality heuristics (deterministic, no LLM).

    Scoring dimensions:
    - Depth:        0-3 pts (longest path length, capped to avoid ultra-linear)
    - Width:        -2 to +2 pts (penalize single-column, reward parallel paths)
    - Diversity:    0-2 pts (unique roles used)
    - Connectivity: -2 to +2 pts (orphan penalty)
    - Output:       0-2 pts (output nodes are DAG sinks)
    """

    def score(self, dag: List[Dict], num_nodes: int,
              node_roles: Dict[int, str]) -> Dict:
        depth = self._depth_score(dag, num_nodes)
        width = self._width_score(dag, num_nodes)
        diversity = self._diversity_score(node_roles)
        connectivity = self._connectivity_score(dag, num_nodes, node_roles)
        output_ok = self._output_placement_score(dag, num_nodes, node_roles)
        total = depth + width + diversity + connectivity + output_ok
        return {
            "total": total,
            "depth": depth,
            "width": width,
            "diversity": diversity,
            "connectivity": connectivity,
            "output": output_ok,
        }

    def _depth_score(self, dag: List[Dict], num_nodes: int) -> int:
        """Score based on longest path. Caps at 3 and penalizes ultra-linear chains.

        A graph with depth == num_nodes-1 is a straight line (bad).
        Ideal depth is 3-6 for most graphs.
        """
        if not dag or num_nodes == 0:
            return 0
        reverse_adj: Dict[int, List[int]] = defaultdict(list)
        for edge in dag:
            reverse_adj[edge["to"]].append(edge["from"])
        layout = DAGLayout()
        depths = layout._assign_z(reverse_adj, num_nodes)
        max_depth = max(depths.values()) if depths else 0

        # Penalize ultra-linear: depth >= num_nodes - 2 means almost every node
        # is on its own Z-layer (single-column pipeline)
        if num_nodes > 4 and max_depth >= num_nodes - 2:
            return 1  # Cap at 1 for straight-line chains

        if max_depth <= 1:
            return 0
        if max_depth == 2:
            return 1
        if max_depth == 3:
            return 2
        return 3

    def _width_score(self, dag: List[Dict], num_nodes: int) -> int:
        """Score based on graph width (max nodes at any Z-layer).

        Penalizes single-column layouts. Rewards parallel processing paths.
        -2 if max_width == 1 (pure linear chain)
         0 if max_width == 2
        +2 if max_width >= 3 (real parallel paths)
        """
        if not dag or num_nodes == 0:
            return 0
        reverse_adj: Dict[int, List[int]] = defaultdict(list)
        for edge in dag:
            reverse_adj[edge["to"]].append(edge["from"])
        layout = DAGLayout()
        depths = layout._assign_z(reverse_adj, num_nodes)

        # Count nodes per Z-layer
        layers: Dict[int, int] = defaultdict(int)
        for z in depths.values():
            layers[z] += 1
        max_width = max(layers.values()) if layers else 1

        if max_width >= 3:
            return 2
        if max_width >= 2:
            return 0
        return -2  # Single column

    def _diversity_score(self, node_roles: Dict[int, str]) -> int:
        """Score based on unique roles. 0 if <2, 1 if 2, 2 if 3."""
        unique_roles = set(node_roles.values())
        if len(unique_roles) >= 3:
            return 2
        if len(unique_roles) >= 2:
            return 1
        return 0

    def _connectivity_score(self, dag: List[Dict], num_nodes: int,
                            node_roles: Dict[int, str]) -> int:
        """Score based on orphans. +2 if 0 orphans, 0 if 1-2, -2 if 3+."""
        has_input = set()
        for edge in dag:
            has_input.add(edge["to"])
        orphans = 0
        for i in range(num_nodes):
            if node_roles.get(i) == "source":
                continue
            if i not in has_input:
                orphans += 1
        if orphans == 0:
            return 2
        if orphans <= 2:
            return 0
        return -2

    def _output_placement_score(self, dag: List[Dict], num_nodes: int,
                                node_roles: Dict[int, str]) -> int:
        """Score: +2 if all output nodes are DAG sinks (nothing flows out), 0 otherwise."""
        outputs = [i for i in range(num_nodes) if node_roles.get(i) == "output"]
        if not outputs:
            return 0
        has_outgoing = set()
        for edge in dag:
            has_outgoing.add(edge["from"])
        for out in outputs:
            if out in has_outgoing:
                return 0
        return 2
