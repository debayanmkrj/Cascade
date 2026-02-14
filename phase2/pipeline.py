"""Phase 2 Pipeline - Polyglot Triple Filter Orchestrator

Orchestrates:
1. Load Phase 1 session JSON
2. Architect agent: plan graph topology → Filter 1 (logic validation)
3. Mason agent: generate code snippets → Filter 2 (syntax validation)
   Mason handles all code generation AND fixing via mason:latest
4. Export project JSON for JS runtime
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from config import SESSIONS_DIR, OUTPUT_DIR
from phase2.data_types import VolumetricGrid, ArchitectPlan
from phase2.agents.architect import ArchitectAgent
from phase2.agents.mason import MasonAgent
from phase2.agents.designer import DesignerAgent
from phase2.agents.graph_designer import GraphDesignerAgent


class Phase2Pipeline:
    """Complete Phase 2 pipeline: Phase 1 JSON → executable project JSON."""

    def __init__(self):
        print("[INIT] Phase 2 Pipeline...")
        self.architect = ArchitectAgent()
        self.mason = MasonAgent()
        self.graph_designer = GraphDesignerAgent()
        self.designer = DesignerAgent()
        print("[INIT] Phase 2 Pipeline ready")

    def execute(self, session_json: Dict) -> Dict:
        """Run the full Phase 2 pipeline.

        Args:
            session_json: Phase 1 session data (from saved JSON file or in-memory)

        Returns:
            Project JSON for JS runtime consumption
        """
        print("=" * 60)
        print("PHASE 2: Polyglot Triple Filter Pipeline")
        print("=" * 60)

        # Step 1: Architect plans graph topology
        print("\n[Step 1/3] Architect: Planning graph topology...")
        plan = self.architect.plan(session_json)
        print(f"  Nodes: {len(plan.grid.nodes)}")
        print(f"  Connections: {len(plan.grid.connections)}")
        print(f"  Grid: {plan.grid.dimensions}")
        print(f"  Filter 1 (topology): {'PASS' if plan.topology_valid else 'FAIL'}")
        for msg in plan.validation_log:
            print(f"    {msg}")
        
        

        # Step 2: Mason generates code
        print("\n[Step 2/3] Mason: Generating code snippets...")
        updated_nodes = self.mason.generate_node_code(plan.grid.nodes)
        plan.grid.nodes = updated_nodes
        passed = sum(1 for n in updated_nodes if n.mason_approved)
        total = len(updated_nodes)
        print(f"  Filter 2 (syntax): {passed}/{total} passed")
        for n in updated_nodes:
            status = "PASS" if n.mason_approved else "FAIL"
            print(f"    {n.id} [{n.engine}]: {status}")

        # Mason handles all fixing internally via mason:latest retries

        # Step 2.5: Designer SKIPPED (disabled per user request)
        print("\n[Step 2.5/3] Designer: SKIPPED (disabled)")

        # Step 3: Skip VLM (Filter 3) per user request
        print("\n[Step 3/3] VLM inspection: SKIPPED (disabled)")

        # Mark all architect-approved
        for node in plan.grid.nodes:
            node.architect_approved = plan.topology_valid

        # Export project JSON
        project_json = plan.grid.to_project_json()
        project_json['phase2_meta'] = {
            'architect_valid': plan.topology_valid,
            'architect_reasoning': plan.reasoning,
            'mason_pass_rate': f"{passed}/{total}",
            'timestamp': datetime.now().isoformat(),
            'session_id': session_json.get('session_id', 'unknown')
        }

        print("\n" + "=" * 60)
        print("PHASE 2 COMPLETE")
        print("=" * 60)

        return project_json

    def execute_from_file(self, session_path: str) -> Dict:
        """Load session JSON from file and execute."""
        with open(session_path, 'r') as f:
            session_json = json.load(f)
        return self.execute(session_json)

    def save_project(self, project_json: Dict, session_id: str = "unnamed") -> str:
        """Save project JSON to output directory."""
        output_path = Path(OUTPUT_DIR) / f"project_{session_id}.json"
        with open(output_path, 'w') as f:
            json.dump(project_json, f, indent=2, default=str)
        print(f"  [OUTPUT] Saved project to {output_path}")
        return str(output_path)
