"""Phase 2 Pipeline - Chain of Influence Orchestrator

Orchestrates:
1. Reasoner: session JSON -> InfluenceGraphIR (nodes + typed edges + contracts)
2. Compiler: IR -> BuildSheets + VolumetricGrid
3. Mason: BuildSheet-driven code generation -> Filter 2 (syntax validation)
4. Parameter review
5. Export project JSON for JS runtime
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from typing import Dict, List
from datetime import datetime

from config import SESSIONS_DIR, OUTPUT_DIR, EFFECTIVE_MODEL_REVIEW
from phase2.data_types import VolumetricGrid
from phase2.agents.reasoner import ReasonerAgent
from phase2.agents.influence_compiler import InfluenceCompiler
from phase2.agents.mason import MasonAgent
from phase2.agents.uniform_validator import get_uniform_validator
from phase2.aider_llm import get_aider_llm


class Phase2Pipeline:
    """Phase 2 pipeline: Phase 1 JSON -> executable project JSON via Chain of Influence."""

    def __init__(self):
        print("[INIT] Phase 2 Pipeline (Chain of Influence)...")
        self.reasoner = ReasonerAgent()
        self.compiler = InfluenceCompiler()
        self.mason = MasonAgent()
        self.aider = get_aider_llm()
        print("[INIT] Phase 2 Pipeline ready")

    def _extract_brief(self, session_json: Dict) -> str:
        return (
            session_json.get('input', {}).get('prompt_text', '')
            or session_json.get('brief', {}).get('essence', '')
            or session_json.get('phase2_context', {}).get('essence', '')
            or session_json.get('user_input', {}).get('brief', '')
            or ''
        )

    def _extract_visual_palette(self, session_json: Dict) -> Dict:
        vp = (
            session_json.get('visual_palette')
            or session_json.get('phase2_context', {}).get('visual_palette')
            or session_json.get('brief', {}).get('visual_palette')
            or {}
        )
        if not vp.get('primary_colors'):
            bv = (
                session_json.get('brand_values')
                or session_json.get('brief', {}).get('brand_values')
                or session_json.get('phase2_context', {}).get('brand_values')
                or {}
            )
            cp = bv.get('color_palette', [])
            if cp:
                vp = vp or {}
                vp['primary_colors'] = cp[:3]
                vp['accent_colors'] = cp[3:6] if len(cp) > 3 else []
        return vp

    def execute(self, session_json: Dict) -> Dict:
        """Run the Chain of Influence pipeline."""
        print("=" * 60)
        print("PHASE 2: Chain of Influence Pipeline")
        print("=" * 60)

        brief = self._extract_brief(session_json)
        visual_palette = self._extract_visual_palette(session_json)
        archetypes = (
            session_json.get('phase2_context', {}).get('node_archetypes', [])
            or session_json.get('brief', {}).get('node_archetypes', [])
        )

        if visual_palette:
            colors = visual_palette.get('primary_colors', [])
            print(f"  Palette: {len(colors)} primary, "
                  f"{len(visual_palette.get('accent_colors', []))} accent")
        else:
            print("  Palette: not available")

        # Step 1: Reasoner designs InfluenceGraphIR
        print("\n[Step 1/3] Reasoner: Designing influence graph...")
        ir = self.reasoner.design(session_json)
        print(f"  Nodes: {len(ir.nodes)}")
        print(f"  Edges: {len(ir.edges)}")
        print(f"  Reasoning: {ir.reasoning[:120]}")

        # Step 2: Compile IR -> BuildSheets + Grid
        print("\n[Step 2/3] Compiler: Building sheets + grid...")
        build_sheets, grid = self.compiler.compile(ir, archetypes)
        print(f"  Grid: {grid.dimensions}")
        for sheet in build_sheets:
            rules_count = sum(len(v) for v in sheet.influence_rules.values())
            print(f"    {sheet.node_id} [{sheet.engine}]: "
                  f"{len(sheet.inputs)} inputs, {rules_count} rules, "
                  f"out={sheet.output_protocol}")

        # Step 3: Mason generates code from BuildSheets
        print("\n[Step 3/3] Mason: Generating code from build sheets...")
        sheet_map = {s.node_id: s for s in build_sheets}
        updated_nodes = self.mason.generate_from_build_sheets(
            grid.nodes, sheet_map, visual_palette
        )
        grid.nodes = updated_nodes
        passed = sum(1 for n in updated_nodes if n.mason_approved)
        total = len(updated_nodes)
        print(f"  Filter 2 (syntax): {passed}/{total} passed")
        for n in updated_nodes:
            status = "PASS" if n.mason_approved else "FAIL"
            print(f"    {n.id} [{n.engine}]: {status}")

        # Review failed nodes
        failed = [n for n in updated_nodes if not n.mason_approved]
        if failed:
            print(f"\n  [REVIEW] {len(failed)} node(s) failed. Retrying...")
            for node in failed:
                errors = "; ".join(node.validation_errors) if node.validation_errors else "unknown"
                review_ctx = (
                    f"Node '{node.id}' (engine={node.engine}) failed.\n"
                    f"Errors: {errors}\nCode:\n{node.code_snippet[:500]}"
                )
                review = self.aider.review(review_ctx, EFFECTIVE_MODEL_REVIEW)
                if not review["ok"]:
                    print(f"    [{node.id}] {review['issues']}")
                    sheet = sheet_map.get(node.id)
                    if sheet:
                        retried = self.mason.generate_from_build_sheets(
                            [node], {node.id: sheet}, visual_palette
                        )
                        if retried and retried[0].mason_approved:
                            for i, n in enumerate(updated_nodes):
                                if n.id == node.id:
                                    updated_nodes[i] = retried[0]
                                    break
                            print(f"    [{node.id}] Retry PASSED")
            grid.nodes = updated_nodes
            passed = sum(1 for n in updated_nodes if n.mason_approved)
            print(f"  Filter 2 (after review): {passed}/{total} passed")

        # Parameter Review
        print("\n[Step 3.5] Parameter Review...")
        self._review_parameters(grid.nodes, brief)

        # Mark architect-approved
        for node in grid.nodes:
            node.architect_approved = True

        # Export
        project_json = grid.to_project_json()
        project_json['phase2_meta'] = {
            'pipeline': 'chain_of_influence',
            'reasoner_reasoning': ir.reasoning,
            'influence_edges': len(ir.edges),
            'mason_pass_rate': f"{passed}/{total}",
            'timestamp': datetime.now().isoformat(),
            'session_id': session_json.get('session_id', 'unknown')
        }

        # Embed Phase 1 context so an output JSON is self-contained for Load Output
        inp = session_json.get('input', {})
        ctx = session_json.get('phase2_context', {})
        project_json['design_brief'] = {
            'prompt_text': inp.get('prompt_text', ''),
            'D_global': inp.get('D_global', 0.5),
            'user_mode': inp.get('user_mode', 'aesthetic'),
            'seed': inp.get('seed', 42),
            'essence': ctx.get('essence', brief),
            'visual_palette': ctx.get('visual_palette', visual_palette),
            'creative_levels': ctx.get('creative_levels', {}),
            'divergence': ctx.get('divergence', {}),
            'node_archetypes': [
                {'id': n.get('id', ''), 'category': n.get('category', ''), 'engine': n.get('engine', '')}
                for n in ctx.get('node_archetypes', [])
            ] if ctx.get('node_archetypes') else [],
        }

        print("\n" + "=" * 60)
        print("PHASE 2 COMPLETE")
        print("=" * 60)
        return project_json

    def _review_parameters(self, nodes: List, brief: str) -> None:
        validator = get_uniform_validator()
        for node in nodes:
            if not node.mason_approved or not node.code_snippet:
                continue
            engine = node.engine or "glsl"
            result = validator.validate_and_reconcile(
                node.code_snippet, node.parameters or {}, engine
            )
            if result["needs_fix"]:
                node.parameters = result["fixed_params"]
                print(f"  [PARAM] {node.id}: {result['analysis']}")
            elif result["unused_params"]:
                print(f"  [PARAM] {node.id}: {result['analysis']}")
            else:
                print(f"  [PARAM] {node.id}: OK")

    def execute_from_file(self, session_path: str) -> Dict:
        with open(session_path, 'r') as f:
            session_json = json.load(f)
        return self.execute(session_json)

    def save_project(self, project_json: Dict, session_id: str = "unnamed") -> str:
        dt_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        short_id = session_id[:8] if len(session_id) >= 8 else session_id
        output_path = Path(OUTPUT_DIR) / f"{dt_str}_output_{short_id}.json"
        # Embed canonical path so any future load knows where to write back
        if "phase2_meta" not in project_json:
            project_json["phase2_meta"] = {}
        project_json["phase2_meta"]["output_path"] = str(output_path)
        with open(output_path, 'w') as f:
            json.dump(project_json, f, indent=2, default=str)
        print(f"  [OUTPUT] Saved to {output_path}")
        return str(output_path)
