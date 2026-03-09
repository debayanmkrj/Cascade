"""
Graph Designer Agent
--------------------
Acts as a VFX Art Director. Reviews the initial node topology, 
critiques the data flow against the design brief, and rewires 
the graph to create better visual compositing before Mason generates code.

Pipeline position: After Architect (Step 1.5), before Mason (Step 2).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import re
from typing import Dict, List, Optional

from phase2.data_types import VolumetricGrid, Connection, NodeTensor
from config import MODEL_NAME_REASONING
from phase2.aider_llm import get_aider_llm


class GraphDesignerAgent:
    """Reviews and rewires node topology for optimal visual flow."""

    def __init__(self, model: str = None, max_retries: int = 3):
        self.model = model or MODEL_NAME_REASONING
        self.max_retries = max_retries

    def critique_and_rewire(self, grid: VolumetricGrid, prompt: str) -> VolumetricGrid:
        """Critiques current topology and rewires connections based on the prompt."""
        print(f"\n[GRAPH DESIGNER] Critiquing topology for brief: '{prompt}'")
        
        # 1. Format current state for the LLM
        nodes_info = []
        for n in grid.nodes:
            # Safely handle meta whether it's a dict or an object
            meta = n.meta if isinstance(n.meta, dict) else (n.meta.__dict__ if hasattr(n.meta, '__dict__') else {})
            category = meta.get("category", "")
            kws = ", ".join(meta.get("keywords", []))
            engine = n.engine or "glsl"
            nodes_info.append(f"  - ID: {n.id} | Engine: {engine} | Category: {category} | Traits: {kws}")
            
        current_conns = []
        for c in grid.connections:
            current_conns.append(f"  - {c.from_node} -> {c.to_node}")

        nodes_str = "\n".join(nodes_info)
        conns_str = "\n".join(current_conns) if current_conns else "  (No valid connections yet)"

        # 2. Build the LLM Prompt
        system_prompt = self._build_prompt(prompt, nodes_str, conns_str)

        # 3. Call LLM with retries
        response_text = None
        for attempt in range(self.max_retries):
            response_text = self._call_llm(system_prompt)
            if not response_text:
                print(f"  [GRAPH DESIGNER] Attempt {attempt+1} failed to get LLM response.")
                continue
                
            new_connections = self._parse_and_apply(response_text, grid)
            if new_connections:
                grid.connections = new_connections
                print("  [GRAPH DESIGNER] ✓ Graph successfully rewired based on visual critique.")
                return grid
                
            print(f"  [GRAPH DESIGNER] Attempt {attempt+1} failed to parse valid connections.")

        print("  [GRAPH DESIGNER] ⚠ Rewiring failed after all retries. Keeping original topology.")
        return grid

    def _build_prompt(self, prompt: str, nodes_str: str, conns_str: str) -> str:
        return f"""You are a Lead VFX Compositor & Graph Architect.
Your job is to review a draft node graph and rewire it to perfectly execute the design brief.

DESIGN BRIEF: "{prompt}"

AVAILABLE NODES:
{nodes_str}

CURRENT DRAFT CONNECTIONS:
{conns_str}

TASK:
1. Visualize how the current data flows. Does it make sense for the design brief?
2. Ensure logical compositing: generators -> modifiers -> mix -> output.
3. Rewire the graph to create the most logical compositing chain for THIS specific brief.

RULES:
- You MUST use the exact ID strings provided in AVAILABLE NODES.
- Ensure all generators eventually flow into modifiers or the final output.
- No cycles (A -> B -> A is banned).
- Your critique MUST reference the actual design brief, not generic examples.
- Output ONLY valid JSON containing a 'critique' string and a 'connections' array.

OUTPUT FORMAT (JSON ONLY):
{{
  "critique": "Brief description of what you changed and why, referencing the design brief.",
  "connections": [
    {{"from": "EXACT_NODE_ID_1", "to": "EXACT_NODE_ID_2"}},
    {{"from": "EXACT_NODE_ID_2", "to": "EXACT_NODE_ID_3"}}
  ]
}}
"""

    def _call_llm(self, prompt: str) -> Optional[str]:
        try:
            aider = get_aider_llm()
            return aider.call(prompt, self.model, think_tokens="4k")
        except Exception as e:
            print(f"  [GRAPH DESIGNER] LLM Error: {e}")
            return None

    def _parse_and_apply(self, text: str, grid: VolumetricGrid) -> Optional[List[Connection]]:
        try:
            # Extract JSON block if markdown formatting is used
            match = re.search(r"```json\s*([\s\S]*?)```", text, re.IGNORECASE)
            if match:
                text = match.group(1)
            else:
                # Try to extract anything resembling a JSON object
                match = re.search(r"\{[\s\S]*\}", text)
                if match:
                    text = match.group(0)

            data = json.loads(text.strip())
            critique = data.get('critique', 'No critique provided.')
            print(f"  [CRITIQUE]: {critique}")
            
            raw_conns = data.get("connections", [])
        except Exception as e:
            print(f"  [GRAPH DESIGNER] JSON Parse Error: {e}")
            return None

        # Validate nodes exist to prevent hallucinated IDs
        valid_ids = {n.id for n in grid.nodes}
        new_conns = []
        
        # Track input slots for each target node
        input_counts = {n.id: 0 for n in grid.nodes}
        
        for c in raw_conns:
            src = c.get("from", "")
            tgt = c.get("to", "")
            
            # Simple fuzzy matching in case LLM drops indices or adds spaces
            src_match = next((nid for nid in valid_ids if src.lower() in nid.lower()), None)
            tgt_match = next((nid for nid in valid_ids if tgt.lower() in nid.lower()), None)
            
            if src_match and tgt_match and src_match != tgt_match:
                # Assign to_input dynamically based on how many connections already go to this target
                current_input_slot = input_counts[tgt_match]
                
                new_conns.append(Connection(
                    from_node=src_match,
                    from_output=0,
                    to_node=tgt_match,
                    to_input=current_input_slot
                ))
                
                input_counts[tgt_match] += 1
                print(f"    -> Wired {src_match} into {tgt_match} (input {current_input_slot})")

        if len(new_conns) > 0:
            return new_conns
            
        print("  [GRAPH DESIGNER] No valid connections parsed from response.")
        return None