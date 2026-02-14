"""UI Generation (Phase 1 - Section 7.2)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from typing import List, Dict
import requests
import json
import re

from config import (MIN_UI_CONTROLS, MAX_UI_CONTROLS, ENTROPY_SCALING_LAMBDA,
                    UI_CONTROL_TYPES, OLLAMA_URL, MODEL_NAME)
from data_types import UIControl, UILayoutNode, DivergenceValues, stable_id
from utils import normalize_control_parameters
from .rag_integration import UINodeLibrary


class UIGenerator:
    """Generate UI controls and layout using RAG + Guided Entropy Scaling"""

    def __init__(self, rag_library: UINodeLibrary):
        self.rag = rag_library

    def generate_ui_for_nodes(self, node_archetypes: List, brand_essence: str,
                               brand_values: Dict, divergence: DivergenceValues,
                               creative_levels: Dict) -> List[UIControl]:
        """
        Generate UI controls mapped to node parameters.

        Algorithm:
        1. Extract all parameters from all nodes
        2. Group parameters by type and functionality
        3. Create one UI control per parameter (or grouped parameters)
        4. Bind controls to node.parameter paths

        Args:
            node_archetypes: List of NodeArchetype objects
            brand_essence: Brand essence
            brand_values: Brand values
            divergence: Divergence values
            creative_levels: Creative levels

        Returns:
            List of UIControl objects with bindings to nodes
        """
        ui_controls = []

        # Extract all node parameters and create controls for them
        for node in node_archetypes:
            for param in node.parameters:
                param_name = param.get('name', 'parameter')
                param_type = param.get('type', 'float')
                param_default = param.get('default', 0.5)
                param_range = param.get('range', [0.0, 1.0])

                # Determine control type from parameter type
                control_type = self._infer_control_type(param_type, param)

                # Create parameters dict
                parameters = self._create_control_parameters(control_type, param)

                # Normalize parameters
                normalized_params = normalize_control_parameters(control_type, parameters)

                # Extract creative_level from node metadata
                creative_level = node.meta.get('creative_level', 'medium') if isinstance(node.meta, dict) else 'medium'

                # Create control
                control = UIControl(
                    id=stable_id("ui", node.id, param_name),
                    type=control_type,
                    label=f"{node.id}: {param_name.replace('_', ' ').title()}",
                    parameters=normalized_params,
                    grounding_score=0.8,  # Default for NodeTensor (no grounding_score attribute)
                    grounding_source=f"Node: {node.id}",
                    confidence=0.85,  # Default for NodeTensor (no confidence attribute)
                    targets=[creative_level],
                    creative_level=creative_level
                )

                # Add binding to node parameter
                control.bindings = [{
                    'node_id': node.id,
                    'node_name': node.id,  # NodeTensor uses id instead of name
                    'parameter': param_name
                }]

                ui_controls.append(control)

        # Apply Guided Entropy Scaling
        ui_controls = self._apply_entropy_scaling(ui_controls, divergence.D_flow)

        # Select top controls based on D_flow
        num_controls = self._compute_ui_count(divergence.D_flow)
        return ui_controls[:num_controls]

    def _infer_control_type(self, param_type: str, param: Dict) -> str:
        """Infer UI control type from parameter type"""
        if param_type == 'float' or param_type == 'int':
            return 'slider'
        elif param_type == 'bool':
            return 'toggle'
        elif param_type == 'color':
            return 'color'
        elif param_type == 'enum':
            return 'dropdown'
        elif param_type == 'vector2':
            return 'xy_pad'
        else:
            return 'slider'

    def _create_control_parameters(self, control_type: str, param: Dict) -> Dict:
        """Create control parameters from node parameter spec"""
        if control_type == 'slider':
            param_range = param.get('range', [0.0, 1.0])
            # Sanitize default value - handle empty strings and None
            default_val = param.get('default', None)
            if default_val is None or default_val == '':
                default_val = (param_range[0] + param_range[1]) / 2
            return {
                'min_val': param_range[0],
                'max_val': param_range[1],
                'default_val': default_val,
                'step': 0.01
            }
        elif control_type == 'toggle':
            default_val = param.get('default', False)
            # Sanitize boolean - handle empty strings
            if default_val == '' or default_val is None:
                default_val = False
            return {'default_state': default_val}
        elif control_type == 'color':
            default_val = param.get('default', '#ffffff')
            # Sanitize color - handle empty strings
            if default_val == '' or default_val is None:
                default_val = '#ffffff'
            return {'default_color': default_val}
        elif control_type == 'dropdown':
            options = param.get('options', ['option1', 'option2'])
            default_val = param.get('default', None)
            # Sanitize dropdown default - handle empty strings
            if default_val == '' or default_val is None:
                default_val = options[0] if options else 'option1'
            return {
                'options': options,
                'default_option': default_val
            }
        elif control_type == 'xy_pad':
            return {
                'x_range': [0.0, 1.0],
                'y_range': [0.0, 1.0],
                'default_x': 0.5,
                'default_y': 0.5
            }
        else:
            return {}

    def generate_ui_candidates(self, brand_essence: str, brand_values: Dict,
                               divergence: DivergenceValues,
                               creative_levels: Dict) -> List[UIControl]:
        """
        Generate UI control candidates using COMPLETE RAG LOOP.

        Algorithm (ENHANCED with full RAG):
        1. Query RAG with context assembly + answer generation
        2. Parse RAG recommendations into control specs
        3. Generate additional controls from raw chunks (diversity)
        4. Apply Guided_Entropy_Scaling based on D_flow
        5. Return ranked candidates

        Args:
            brand_essence: Core brand/concept statement
            brand_values: Brand values dict
            divergence: Divergence values
            creative_levels: Creative level weights

        Returns:
            List of UIControl candidates
        """
        # Step 1: Use COMPLETE RAG interface (retrieval + generation)
        query = f"{brand_essence} user interface controls interaction design"

        print(f"  → Querying RAG for UI controls with generation...")
        rag_answer = self.rag.query_ui_controls_with_generation(query, brand_values, top_k=15)

        print(f"  → RAG returned {len(rag_answer.get('recommendations', []))} recommendations from {rag_answer.get('num_sources', 0)} sources")
        print(f"  → Design principles: {rag_answer.get('design_principles', [])}")

        # Step 2: Parse RAG recommendations into controls
        ui_controls = []

        # 2a: Process RAG-generated recommendations (high confidence)
        for rec in rag_answer.get('recommendations', []):
            try:
                control = self._generate_control_from_rag_recommendation(
                    rec, brand_essence, divergence.D_flow, creative_levels
                )
                if control:
                    # Boost confidence for RAG-generated recommendations
                    control.confidence *= 1.2
                    ui_controls.append(control)
            except Exception as e:
                print(f"    Warning: Failed to parse RAG recommendation: {e}")

        # 2b: Generate additional controls from raw chunks for diversity
        ui_concepts = rag_answer.get('retrieved_chunks', [])

        # Step 2: Generate controls for each concept
        ui_controls = []
        for i, concept in enumerate(ui_concepts):
            try:
                control = self._generate_control_from_concept(
                    concept, brand_essence, divergence.D_flow, creative_levels
                )
                if control:
                    ui_controls.append(control)
            except Exception as e:
                print(f"Error generating control for concept {concept.get('concept', 'unknown')}: {e}")

        # Step 3: Apply Guided Entropy Scaling (rank and weight)
        ui_controls = self._apply_entropy_scaling(ui_controls, divergence.D_flow)

        # Step 4: Select top controls
        num_controls = self._compute_ui_count(divergence.D_flow)
        return ui_controls[:num_controls]

    def _generate_control_from_rag_recommendation(self, recommendation: Dict,
                                                   brand_essence: str, D_flow: float,
                                                   creative_levels: Dict) -> UIControl:
        """
        Generate UI control from RAG-generated recommendation.

        RAG recommendations already have LLM-generated specs, so we just
        need to parse and validate them into UIControl objects.

        Args:
            recommendation: Dict from RAG with control_type, concept, rationale, etc.
            brand_essence: Brand essence
            D_flow: Flow divergence
            creative_levels: Creative levels

        Returns:
            UIControl object
        """
        control_type = recommendation.get('control_type', 'slider')
        concept = recommendation.get('concept', 'control')
        rationale = recommendation.get('rationale', '')
        suggested_params = recommendation.get('suggested_parameters', '')

        # Parse suggested parameters into actual parameter dict
        # This is a simple parser - could be enhanced
        parameters = {}
        if 'range' in suggested_params.lower():
            # Extract range like "0-1" or "0.0-1.0"
            import re
            range_match = re.search(r'(\d+\.?\d*)\s*[-to]+\s*(\d+\.?\d*)', suggested_params)
            if range_match:
                parameters['min_val'] = float(range_match.group(1))
                parameters['max_val'] = float(range_match.group(2))
                parameters['default_val'] = (parameters['min_val'] + parameters['max_val']) / 2
                parameters['step'] = 0.01

        # Use default parameters if parsing failed
        if not parameters:
            if control_type == 'slider' or control_type == 'knob' or control_type == 'fader':
                parameters = {'min_val': 0.0, 'max_val': 1.0, 'default_val': 0.5, 'step': 0.01}
            elif control_type == 'toggle':
                parameters = {'default_state': False}
            elif control_type == 'color':
                parameters = {'default_color': '#ffffff'}

        # Normalize parameters
        normalized_params = normalize_control_parameters(control_type, parameters)

        # Determine creative level
        creative_level = self._determine_creative_level(creative_levels)

        return UIControl(
            id=stable_id("ui", concept, brand_essence),
            type=control_type,
            label=concept.title(),
            parameters=normalized_params,
            grounding_score=0.8,  # High score for RAG-generated
            grounding_source=f"RAG Generated: {rationale[:50]}...",
            confidence=0.8,  # Will be boosted to 0.96 in generate_ui_candidates
            targets=['surface', 'flow'],  # Default targets
            creative_level=creative_level
        )

    def _generate_control_from_concept(self, concept: Dict, brand_essence: str,
                                       D_flow: float, creative_levels: Dict) -> UIControl:
        """Generate a single UI control from a RAG concept using LLM"""
        concept_text = concept.get('text', '')
        concept_name = concept.get('concept', 'control')

        # Temperature based on D_flow
        temperature = 0.3 + (D_flow * 0.8)  # 0.3 to 1.1

        generation_prompt = f"""You are a UI design expert. Generate a UI control specification based on the provided context.

BRAND ESSENCE: {brand_essence}
CONCEPT: {concept_name}
CONTEXT FROM KNOWLEDGE BASE: {concept_text}

Based ONLY on the context above, generate ONE UI control specification that fits this concept.

Output valid JSON:
{{
    "type": "control type inferred from context",
    "label": "Control Name",
    "parameters": {{
        "parameter details inferred from context"
    }},
    "description": "What this control does based on context",
    "targets": ["creative levels this affects"]
}}

JSON OUTPUT:"""

        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    'model': MODEL_NAME,
                    'prompt': generation_prompt,
                    'stream': False,
                    'format': 'json',
                    'temperature': temperature
                },
                timeout=60
            )

            result_text = response.json().get('response', '{}')
            result_text = re.sub(r'<think>.*?</think>', '', result_text, flags=re.DOTALL)

            json_match = re.search(r'\{[\s\S]*\}', result_text)
            if json_match:
                control_spec = json.loads(json_match.group())

                # Get control type
                control_type = control_spec.get('type', 'slider')

                # Normalize parameters
                raw_parameters = control_spec.get('parameters', {})
                normalized_parameters = normalize_control_parameters(control_type, raw_parameters)

                # Get targets (which creative levels this affects)
                targets = control_spec.get('targets', ['surface'])

                # Determine primary creative level
                creative_level = self._determine_creative_level(creative_levels)

                return UIControl(
                    id=stable_id("ui", concept_name, brand_essence),
                    type=control_type,
                    label=control_spec.get('label', concept_name.title()),
                    parameters=normalized_parameters,
                    grounding_score=concept.get('grounding_score', 0.5),
                    grounding_source=concept.get('header', 'Unknown'),
                    confidence=concept.get('similarity', 0.5),
                    targets=targets,
                    creative_level=creative_level
                )
        except Exception as e:
            print(f"LLM generation error for {concept_name}: {e}")

        # Fallback control
        return self._fallback_control(concept_name, concept, creative_levels)

    def _fallback_control(self, concept_name: str, concept: Dict, creative_levels: Dict) -> UIControl:
        """Create fallback control when LLM fails"""
        creative_level = self._determine_creative_level(creative_levels)

        # Normalize default slider parameters
        raw_params = {'min': 0.0, 'max': 1.0, 'default': 0.5, 'step': 0.01}
        normalized_params = normalize_control_parameters('slider', raw_params)

        return UIControl(
            id=stable_id("ui", concept_name, "fallback"),
            type='slider',
            label=concept_name.title(),
            parameters=normalized_params,
            grounding_score=concept.get('grounding_score', 0.3),
            grounding_source=concept.get('header', 'Fallback'),
            confidence=0.3,
            targets=['surface'],
            creative_level=creative_level
        )

    def _determine_creative_level(self, creative_levels: Dict) -> str:
        """Determine primary creative level based on weights"""
        if not creative_levels:
            return "surface"

        # Get the level with highest weight
        max_level = max(creative_levels.items(), key=lambda x: x[1])
        return max_level[0]

    def _apply_entropy_scaling(self, controls: List[UIControl], D_flow: float) -> List[UIControl]:
        """
        Apply Guided Entropy Scaling to rank controls.

        Algorithm (from Phase 1 spec):
        - At low D: favor high-confidence, top-ranked controls
        - At high D: favor diverse, mid-ranked controls
        """
        # Sort by confidence
        controls.sort(key=lambda c: c.confidence, reverse=True)

        # Apply Goldilocks weighting
        for rank, control in enumerate(controls, start=1):
            # Goldilocks weight: favor mid-ranks at high D
            r_star = 1 + D_flow * (min(len(controls), 20) - 1)
            sigma_r = 5.0
            w_gold = np.exp(-((rank - r_star) ** 2) / (2 * sigma_r ** 2))

            # Top-token penalty at high D
            is_top = (rank == 1)
            w_top = 1.0 - (0.7 * D_flow) if is_top else 1.0

            # Update confidence with weights
            control.confidence = control.confidence * w_gold * w_top

        # Re-sort by weighted confidence
        controls.sort(key=lambda c: c.confidence, reverse=True)
        return controls

    def _compute_ui_count(self, D_flow: float) -> int:
        """
        Compute number of UI controls based on D_flow.

        Low D: fewer, focused controls
        High D: more, exploratory controls
        """
        count = MIN_UI_CONTROLS + int((MAX_UI_CONTROLS - MIN_UI_CONTROLS) * D_flow)
        return np.clip(count, MIN_UI_CONTROLS, MAX_UI_CONTROLS)

    def converge_ui(self, candidates: List[UIControl]) -> UILayoutNode:
        """
        Converge UI candidates into hierarchical layout.

        Algorithm (from Phase 1 spec - section 7.2.3):
        1. Group controls by semantic similarity
        2. Create hierarchy with groups
        3. Return root layout node

        Args:
            candidates: List of UI controls

        Returns:
            Root UILayoutNode with hierarchy
        """
        # Simple grouping by control type for now
        groups = {}
        for control in candidates:
            ctrl_type = control.type
            if ctrl_type not in groups:
                groups[ctrl_type] = []
            groups[ctrl_type].append(control)

        # Create group nodes
        children = []
        for group_type, controls in groups.items():
            if len(controls) > 1:
                # Create a group
                group_node = UILayoutNode(
                    id=f"group_{group_type}",
                    type='group',
                    label=group_type.replace('_', ' ').title(),
                    controls=controls,
                    children=[],
                    layout_hint='vertical'
                )
                children.append(group_node)
            else:
                # Single control, add directly
                control_node = UILayoutNode(
                    id=f"node_{controls[0].id}",
                    type='control',
                    controls=controls,
                    children=[]
                )
                children.append(control_node)

        # Root layout
        root = UILayoutNode(
            id='ui_root',
            type='group',
            label='UI Controls',
            controls=[],
            children=children,
            layout_hint='vertical'
        )

        return root
