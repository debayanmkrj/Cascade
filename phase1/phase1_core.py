"""Phase 1 Core Algorithm (Section 7.1 - Complete Pipeline)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Dict, List, Optional
import json
import uuid
from datetime import datetime
import numpy as np

from data_types import (RawInputBundle, NodeBrief, BrandValues, CreativeLevelSpec,
                        DivergenceValues, UILayoutNode, UIControl, NodeArchetype, VisualPalette)
from .brand_extraction import BrandExtractor
from .creative_levels import CreativeLevelEstimator
from .rag_integration import UINodeLibrary
from .ui_generation import UIGenerator
from .node_archetypes import NodeArchetypeGenerator
from .visual_clustering import VisualClusterer
from .visual_palette import VisualPaletteGenerator
import utils


class Phase1Pipeline:
    """
    Complete Phase 1 pipeline for generating NodeBrief from RawInputBundle.

    Implements algorithm from Phase 1 spec section 7.1.
    """

    def __init__(self):
        """Initialize all components"""
        print("Initializing Phase 1 Pipeline...")

        print("  [1/7] Loading brand extractor...")
        self.brand_extractor = BrandExtractor()

        print("  [2/7] Loading creative level estimator...")
        self.level_estimator = CreativeLevelEstimator()

        print("  [3/7] Loading RAG library...")
        self.rag_library = UINodeLibrary()
        utils.set_rag_library(self.rag_library)

        print("  [4/7] Loading UI generator...")
        self.ui_generator = UIGenerator(self.rag_library)

        print("  [5/7] Loading node archetype generator...")
        self.node_generator = NodeArchetypeGenerator(self.rag_library)

        print("  [6/7] Loading visual clusterer...")
        self.visual_clusterer = VisualClusterer()

        print("  [7/7] Loading visual palette generator...")
        self.visual_palette_gen = VisualPaletteGenerator(
            self.brand_extractor.model,
            self.brand_extractor.processor
        )

        print("✓ Phase 1 Pipeline ready\n")

    def execute(self, input_bundle: RawInputBundle) -> NodeBrief:
        """
        Execute complete Phase 1 pipeline.

        Algorithm (from spec section 7.1):
        1. Cluster images with CLIP
        2. Extract brand values from prompt + images
        3. Generate visual palette (shapes + motion)
        4. Estimate creative level weights
        5. Compute divergence per level
        6. Generate UI controls using RAG + Guided Entropy Scaling
        7. Generate node archetypes using RAG
        8. Assemble NodeBrief with grounding report

        Args:
            input_bundle: RawInputBundle with user input

        Returns:
            NodeBrief with complete workflow specification
        """
        print("="*60)
        print("PHASE 1: Node-Based Workflow Blueprint Generation")
        print("="*60)

        # Set random seed for reproducibility
        np.random.seed(input_bundle.seed)

        # Step 0: Cluster images (if present)
        all_images = input_bundle.reference_images + input_bundle.custom_images
        if all_images:
            print("\n[Step 0/8] Clustering images with CLIP...")
            try:
                cluster_result = self.visual_clusterer.cluster_images(all_images, n_clusters=min(6, len(all_images)))
                print(f"  ✓ Clustered {len(all_images)} images into {len(cluster_result['clusters'])} clusters")
                print(f"  ✓ Overall mood: {cluster_result['mood']}")
            except Exception as e:
                print(f"  ! Clustering failed: {e}")
                cluster_result = None
        else:
            print("\n[Step 0/8] No images to cluster, skipping...")
            cluster_result = None

        # Step 1: Extract brand values
        print("\n[Step 1/8] Extracting brand values from prompt + images...")
        brand_values = self.brand_extractor.extract_brand_values(
            input_bundle.prompt_text,
            all_images
        )
        print(f"  ✓ Extracted {len(brand_values.emotions)} emotions, {len(brand_values.brand_attributes)} attributes")
        print(f"  ✓ Visual mood: {brand_values.visual_mood}")
        print(f"  ✓ Color palette: {brand_values.color_palette[:3]}...")

        # Step 1b: Generate visual palette (shapes + motion)
        print("\n[Step 1b/8] Generating visual palette...")
        try:
            visual_palette = self.visual_palette_gen.generate_palette(all_images, brand_values)
            print(f"  ✓ Shapes: {visual_palette.shapes}")
            print(f"  ✓ Motion: {visual_palette.motion_words}")
        except Exception as e:
            print(f"  ! Visual palette generation failed: {e}")
            # Fallback palette
            visual_palette = VisualPalette(
                primary_colors=brand_values.color_palette[:3] if len(brand_values.color_palette) >= 3 else ["#1e293b", "#64748b", "#e2e8f0"],
                accent_colors=brand_values.color_palette[3:6] if len(brand_values.color_palette) > 3 else ["#f59e0b", "#06b6d4", "#a855f7"],
                shapes=["geometric shapes", "circles", "lines"],
                motion_words=["smooth", "dynamic", "flowing"]
            )

        # Step 2: Estimate creative level weights
        print("\n[Step 2/8] Estimating creative level weights...")
        creative_levels = self.level_estimator.estimate_level_weights(
            brand_values,
            input_bundle.user_mode
        )
        print(f"  ✓ Surface: {creative_levels.surface:.2f}")
        print(f"  ✓ Flow: {creative_levels.flow:.2f}")
        print(f"  ✓ Narrative: {creative_levels.narrative:.2f}")

        # Step 3: Compute divergence per level
        print("\n[Step 3/8] Computing divergence per level...")
        divergence = self.level_estimator.compute_divergence_per_level(
            input_bundle.D_global,
            input_bundle.user_mode,
            creative_levels
        )
        print(f"  ✓ D_surface: {divergence.D_surface:.2f}")
        print(f"  ✓ D_flow: {divergence.D_flow:.2f}")
        print(f"  ✓ D_narrative: {divergence.D_narrative:.2f}")

        # Step 4: Generate essence statement
        print("\n[Step 4/8] Generating essence statement...")
        essence = self._generate_essence(
            input_bundle.prompt_text,
            brand_values,
            creative_levels
        )
        print(f"  ✓ Essence: {essence[:80]}...")

        # Step 5: Generate node archetypes FIRST (UI controls need to know about nodes)
        print("\n[Step 5/8] Generating node archetypes...")
        # Pass full brand context (emotions + attributes + mood + palette) for semantic reasoner
        full_brand_context = {
            "emotions": brand_values.emotions,
            "attributes": brand_values.brand_attributes,
            "visual_mood": brand_values.visual_mood,
            "palette": brand_values.color_palette,
        }
        node_archetypes = self.node_generator.generate_node_archetypes(
            essence,
            full_brand_context,
            divergence,
            {'surface': creative_levels.surface, 'flow': creative_levels.flow, 'narrative': creative_levels.narrative},
            visual_palette=visual_palette  # Include image-derived visual features
        )
        print(f"  ✓ Generated {len(node_archetypes)} node archetypes")

        # Workflow description not needed - mason_examples.json provides sufficient context
        workflow_description = " -> ".join([n.id for n in node_archetypes]) if node_archetypes else "empty"

        # Step 6-7: UI generation SKIPPED - will be handled by Phase 2 UI agent
        print("\n[Step 6/8] Skipping UI generation (Phase 2 will handle this dynamically)...")
        ui_controls = []
        ui_layout = None
        grounding_report = {"total_sources": 0, "sources": []}

        # Step 8: Create NodeBrief
        print("\n[Step 8/8] Assembling final NodeBrief...")
        node_brief = NodeBrief(
            essence=essence,
            brand_values=brand_values,
            creative_levels=creative_levels,
            divergence=divergence,
            visual_palette=visual_palette,
            ui_layout=ui_layout,
            ui_controls=ui_controls,
            node_archetypes=node_archetypes,
            node_count=len(node_archetypes),
            node_workflow_description=workflow_description,
            grounding_report=grounding_report,
            seed=input_bundle.seed,
            platform=input_bundle.platform_preference or "generic"
        )
        print(f"  ✓ NodeBrief complete with {len(node_archetypes)} nodes (UI generation deferred to Phase 2)")

        print("\n" + "="*60)
        print("✓ PHASE 1 COMPLETE")
        print("="*60)

        return node_brief

    def _generate_essence(self, prompt_text: str, brand_values: BrandValues,
                         creative_levels: CreativeLevelSpec) -> str:
        """
        Generate essence statement from prompt and brand values.

        Simple version: use prompt text as base, enhance with mood.
        """
        # For now, use prompt text directly
        # In full version, would use LLM to distill essence
        essence_base = prompt_text.strip()

        if brand_values.visual_mood and brand_values.visual_mood != 'neutral':
            essence = f"{essence_base} with {brand_values.visual_mood} aesthetic"
        else:
            essence = essence_base

        return essence[:200]  # Limit length

    def save_session_json(self, input_bundle, node_brief) -> str:
        """Save Phase 1 session as JSON for Phase 2 consumption.
        Returns the path to the saved session file."""
        from config import SESSIONS_DIR

        session_id = str(uuid.uuid4())
        session_data = {
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'input': {
                'prompt_text': input_bundle.prompt_text,
                'D_global': input_bundle.D_global,
                'user_mode': input_bundle.user_mode,
                'platform_preference': input_bundle.platform_preference,
                'seed': input_bundle.seed,
                'reference_images': [
                    {'id': img.id, 'url': img.url, 'mood': img.mood, 'cluster_id': img.cluster_id}
                    for img in input_bundle.reference_images
                ],
                'custom_images': [
                    {'id': img.id, 'url': img.url, 'mood': img.mood, 'cluster_id': img.cluster_id}
                    for img in input_bundle.custom_images
                ]
            },
            'brief': node_brief.to_dict(),
            'phase2_context': {
                'essence': node_brief.essence,
                'node_archetypes': [
                    n.to_dict() if hasattr(n, 'to_dict') else {'id': n.id, 'meta': n.meta}
                    for n in node_brief.node_archetypes
                ],
                'ui_controls': [
                    {
                        'id': c.id,
                        'type': c.type,
                        'label': c.label,
                        'parameters': c.parameters,
                        'bindings': c.bindings,
                        'creative_level': c.creative_level
                    } for c in node_brief.ui_controls
                ],
                'visual_palette': {
                    'primary_colors': node_brief.visual_palette.primary_colors,
                    'accent_colors': node_brief.visual_palette.accent_colors,
                    'shapes': node_brief.visual_palette.shapes,
                    'motion_words': node_brief.visual_palette.motion_words
                },
                'creative_levels': {
                    'surface': node_brief.creative_levels.surface,
                    'flow': node_brief.creative_levels.flow,
                    'narrative': node_brief.creative_levels.narrative
                },
                'divergence': {
                    'D_surface': node_brief.divergence.D_surface,
                    'D_flow': node_brief.divergence.D_flow,
                    'D_narrative': node_brief.divergence.D_narrative
                }
            }
        }

        dt_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        short_id = session_id[:8]
        session_path = Path(SESSIONS_DIR) / f"{dt_str}_session_{short_id}.json"
        with open(session_path, 'w') as f:
            json.dump(session_data, f, indent=2, default=str)

        print(f"  [SESSION] Saved to {session_path}")
        return str(session_path)

    def _build_grounding_report(self, ui_controls: List[UIControl],
                                node_archetypes: List[NodeArchetype]) -> Dict:
        """
        Build grounding report with all sources and confidence scores.

        Returns dict with:
        - total_sources: number of unique sources
        - avg_confidence: average confidence across all components
        - ui_sources: list of sources for UI
        - node_sources: list of sources for nodes
        """
        ui_sources = []
        node_sources = []

        ui_confidences = []
        node_confidences = []

        for control in ui_controls:
            ui_sources.append({
                'component': control.label,
                'source': control.grounding_source,
                'score': control.grounding_score,
                'confidence': control.confidence
            })
            ui_confidences.append(control.confidence)

        for node in node_archetypes:
            node_sources.append({
                'component': node.id,  # NodeTensor uses id instead of name
                'source': getattr(node, 'grounding_source', 'semantic_reasoner'),
                'score': getattr(node, 'grounding_score', 0.8),
                'confidence': getattr(node, 'confidence', 0.85)
            })
            node_confidences.append(getattr(node, 'confidence', 0.85))

        all_confidences = ui_confidences + node_confidences
        avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.5

        # Count unique sources
        unique_sources = set()
        for src in ui_sources + node_sources:
            unique_sources.add(src['source'])

        return {
            'total_sources': len(unique_sources),
            'avg_confidence': avg_confidence,
            'ui_sources': ui_sources,
            'node_sources': node_sources
        }
