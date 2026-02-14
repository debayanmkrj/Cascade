"""Design Assistant - Explain/Edit/Reiterate (Spec Section 8)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import re
import requests
from typing import Dict, Optional, Tuple

from config import OLLAMA_URL, MODEL_NAME
from data_types import NodeBrief, EditRequest, ExplanationResponse
from session_manager import SessionManager


class DesignAssistant:
    """
    DesignAssistant agent for:
    1. Explain - help user understand decisions
    2. Edit - apply natural-language edits to NodeBrief
    3. Re-iterate - regenerate with new seed or changed inputs

    Uses llama3.2:latest for fast, controlled edits.
    Session context provides long-trail memory.
    """

    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager
        self.llm_url = OLLAMA_URL
        self.model = MODEL_NAME

    def explain(self, session_id: str, component: str) -> ExplanationResponse:
        """
        Explain a component of the NodeBrief.

        Args:
            session_id: Session to explain
            component: What to explain (e.g., 'creative_levels', 'ui_controls', 'divergence')

        Returns:
            ExplanationResponse with natural language explanation
        """
        session = self.session_manager.get_session(session_id)
        if not session or not session.node_brief:
            return ExplanationResponse(
                component=component,
                explanation="No brief available to explain.",
                grounding_sources=[],
                confidence=0.0
            )

        brief = session.node_brief
        context = self.session_manager.get_context_for_llm(session_id)

        # Build explanation prompt
        component_data = self._get_component_data(brief, component)

        prompt = f"""You are a design assistant explaining a generated workflow blueprint.

SESSION CONTEXT:
{context}

COMPONENT TO EXPLAIN: {component}
DATA:
{json.dumps(component_data, indent=2)}

Explain this component in 2-3 sentences:
- Why was it chosen/generated this way?
- How does it relate to the user's input (prompt, images, mode, divergence)?
- What is its purpose in the workflow?

Be concise and use natural language. Focus on design decisions, not technical jargon.

EXPLANATION:"""

        try:
            response = requests.post(
                self.llm_url,
                json={'model': self.model, 'prompt': prompt, 'stream': False},
                timeout=30
            )
            explanation = response.json().get('response', '').strip()

            return ExplanationResponse(
                component=component,
                explanation=explanation,
                grounding_sources=self._extract_sources(brief, component),
                confidence=0.8
            )
        except Exception as e:
            print(f"Explanation error: {e}")
            return ExplanationResponse(
                component=component,
                explanation=f"Error generating explanation: {str(e)}",
                grounding_sources=[],
                confidence=0.0
            )

    def query(self, session_id: str, query_text: str) -> str:
        """
        General query handler that routes to explain or edit based on query content.
        """
        query_lower = query_text.lower()

        # Check if it's an edit request
        edit_keywords = ['change', 'make', 'add', 'remove', 'modify', 'update', 'set', 'adjust']
        is_edit = any(kw in query_lower for kw in edit_keywords)

        if is_edit:
            modified_brief, response = self.edit(session_id, query_text)
            return response
        else:
            # Treat as explanation/question
            return self._answer_question(session_id, query_text)

    def _answer_question(self, session_id: str, question: str) -> str:
        """Answer general questions about the brief"""
        session = self.session_manager.get_session(session_id)
        if not session or not session.node_brief:
            return "No brief available. Please generate a brief first."

        brief = session.node_brief
        context = self.session_manager.get_context_for_llm(session_id)

        prompt = f"""You are a design assistant helping explain a generated workflow blueprint.

SESSION CONTEXT:
{context}

BRIEF SUMMARY:
Essence: {brief.essence}
UI Controls: {len(brief.ui_controls)}
Nodes: {brief.node_count}
Divergence: Surface={brief.divergence.D_surface:.2f}, Flow={brief.divergence.D_flow:.2f}, Narrative={brief.divergence.D_narrative:.2f}
Colors: {brief.visual_palette.primary_colors[:3]}

USER QUESTION:
"{question}"

Provide a clear, concise answer (2-4 sentences). Focus on design decisions and their rationale.

ANSWER:"""

        try:
            response = requests.post(
                self.llm_url,
                json={'model': self.model, 'prompt': prompt, 'stream': False},
                timeout=30
            )
            response.raise_for_status()
            return response.json().get('response', 'Error: No response from model').strip()
        except Exception as e:
            return f"Error: {str(e)}"

    def edit(self, session_id: str, instruction: str) -> Tuple[NodeBrief, str]:
        """
        Apply natural-language edit to NodeBrief.

        Algorithm (from spec section 8.2b):
        1. LLM interprets instruction as constraint patch
        2. Apply patch to node_brief
        3. Optionally re-run small parts of Phase 1
        4. Return modified brief + response message

        Args:
            session_id: Session to edit
            instruction: Natural language instruction (e.g., "make colors warmer", "add more controls")

        Returns:
            Tuple of (modified_brief, response_message)
        """
        session = self.session_manager.get_session(session_id)
        if not session or not session.node_brief:
            return None, "No brief available to edit."

        brief = session.node_brief
        context = self.session_manager.get_context_for_llm(session_id)

        # Build edit prompt with full brief details
        edit_prompt = f"""You are a design assistant. Apply the user's edit to this workflow blueprint.

SESSION CONTEXT:
{context}

CURRENT BRIEF (FULL DETAILS):
Essence: {brief.essence}

Brand Values:
- Visual Mood: {brief.brand_values.visual_mood}
- Color Palette: {brief.brand_values.color_palette[:5]}
- Top Emotions: {dict(list(brief.brand_values.emotions.items())[:3])}
- Top Attributes: {dict(list(brief.brand_values.brand_attributes.items())[:3])}

Visual Palette:
- Primary Colors: {brief.visual_palette.primary_colors[:5]}
- Accent Colors: {brief.visual_palette.accent_colors[:5]}
- Shapes: {brief.visual_palette.shapes[:5]}
- Motion Words: {brief.visual_palette.motion_words[:5]}

Creative Levels:
- Surface: {brief.creative_levels.surface:.2f}
- Flow: {brief.creative_levels.flow:.2f}
- Narrative: {brief.creative_levels.narrative:.2f}

Divergence:
- D_surface: {brief.divergence.D_surface:.2f}
- D_flow: {brief.divergence.D_flow:.2f}
- D_narrative: {brief.divergence.D_narrative:.2f}

UI Controls: {len(brief.ui_controls)} controls
Node Archetypes: {brief.node_count} nodes
Workflow Description: {brief.node_workflow_description[:100]}...

USER EDIT INSTRUCTION:
"{instruction}"

TASK:
You can edit ANY field in the brief. Interpret the user's instruction and return specific modifications.

EDITABLE FIELDS:
- essence: string (the core design philosophy)
- brand_values: visual_mood, color_palette, emotions (dict), brand_attributes (dict)
- visual_palette: primary_colors (list), accent_colors (list), shapes (list), motion_words (list)
- creative_levels: surface (0-1), flow (0-1), narrative (0-1)
- divergence: D_surface (0-1), D_flow (0-1), D_narrative (0-1)
- node_workflow_description: string (workflow explanation)
- ui_controls: Can add/remove/modify controls (action: "add", "remove", or "modify")
- node_archetypes: Can add/remove/modify nodes (action: "add", "remove", or "modify")

OUTPUT FORMAT (JSON only):
{{
    "changes": {{
        "essence": "new essence text",
        "brand_values": {{"visual_mood": "energetic", "color_palette": ["#FF0000", "#00FF00"], "emotions": {{"excitement": 0.9}}, "brand_attributes": {{"playful": 0.8}}}},
        "visual_palette": {{"primary_colors": ["#FF0000", "#00FF00"], "accent_colors": ["#FFFF00"], "shapes": ["circles"], "motion_words": ["flowing"]}},
        "creative_levels": {{"surface": 0.7, "flow": 0.5, "narrative": 0.3}},
        "divergence": {{"D_surface": 0.8, "D_flow": 0.6, "D_narrative": 0.5}},
        "node_workflow_description": "new workflow description",
        "ui_controls": [{{"action": "add", "label": "Particle Size", "type": "slider", "creative_level": "surface", "parameters": {{"min": 0, "max": 100, "default": 10}}}}, {{"action": "remove", "index": 0}}],
        "node_archetypes": [{{"action": "add", "name": "Color Mixer", "category": "effect", "role": "transform", "description": "Mixes colors dynamically"}}, {{"action": "modify", "index": 0, "name": "New Node Name", "description": "Updated description"}}]
    }},
    "response": "Natural language response explaining what you changed"
}}

IMPORTANT:
- For ui_controls and node_archetypes, you can pass EITHER a single dict OR a list of dicts
- Use action="add" to create new controls/nodes
- Use action="remove" with index to delete existing ones
- Use action="modify" with index to update existing ones

Only include fields you are actually modifying in the changes dict.
Return ONLY valid JSON. NO explanatory text before or after.

JSON OUTPUT:"""

        try:
            response = requests.post(
                self.llm_url,
                json={'model': self.model, 'prompt': edit_prompt, 'stream': False, 'format': 'json'},
                timeout=60
            )

            result_text = response.json().get('response', '{}')

            # Parse JSON
            json_match = re.search(r'\{[\s\S]*\}', result_text)
            if json_match:
                patch = json.loads(json_match.group())
                changes = patch.get('changes', {})
                response_msg = patch.get('response', 'Brief updated.')

                # Debug: print what fields are being changed
                print(f"Design Assistant Edit - Fields being modified: {list(changes.keys())}")
                print(f"Design Assistant Edit - Changes content: {json.dumps(changes, indent=2)}")

                # Apply changes
                modified_brief = self._apply_patch(brief, changes)

                # Verify changes were applied
                print(f"Design Assistant Edit - Brief after patch:")
                print(f"  UI Controls: {len(modified_brief.ui_controls)}")
                print(f"  Node Archetypes: {len(modified_brief.node_archetypes)}")
                print(f"  Essence: {modified_brief.essence[:50]}...")

                # Log edit and save session
                self.session_manager.log_edit(session_id, instruction, changes)
                self.session_manager.update_brief(session_id, modified_brief)
                self.session_manager.save_session(session_id)

                return modified_brief, response_msg
            else:
                return brief, "Could not parse edit instructions. Please try rephrasing."

        except Exception as e:
            print(f"Edit error: {e}")
            return brief, f"Error applying edit: {str(e)}"

    def _apply_patch(self, brief: NodeBrief, changes: Dict) -> NodeBrief:
        """Apply changes dict to NodeBrief"""
        # Create a copy
        import copy
        modified_brief = copy.deepcopy(brief)

        # Apply visual_palette changes
        if 'visual_palette' in changes:
            vp_changes = changes['visual_palette']
            if 'primary_colors' in vp_changes:
                modified_brief.visual_palette.primary_colors = vp_changes['primary_colors']
            if 'accent_colors' in vp_changes:
                modified_brief.visual_palette.accent_colors = vp_changes['accent_colors']
            if 'shapes' in vp_changes:
                modified_brief.visual_palette.shapes = vp_changes['shapes']
            if 'motion_words' in vp_changes:
                modified_brief.visual_palette.motion_words = vp_changes['motion_words']

        # Apply brand_values changes
        if 'brand_values' in changes:
            bv_changes = changes['brand_values']
            if 'color_palette' in bv_changes:
                modified_brief.brand_values.color_palette = bv_changes['color_palette']
            if 'visual_mood' in bv_changes:
                modified_brief.brand_values.visual_mood = bv_changes['visual_mood']
            if 'emotions' in bv_changes:
                modified_brief.brand_values.emotions.update(bv_changes['emotions'])
            if 'brand_attributes' in bv_changes:
                modified_brief.brand_values.brand_attributes.update(bv_changes['brand_attributes'])

        # Apply divergence changes
        if 'divergence' in changes:
            div_changes = changes['divergence']
            if 'D_surface' in div_changes:
                modified_brief.divergence.D_surface = float(div_changes['D_surface'])
            if 'D_flow' in div_changes:
                modified_brief.divergence.D_flow = float(div_changes['D_flow'])
            if 'D_narrative' in div_changes:
                modified_brief.divergence.D_narrative = float(div_changes['D_narrative'])

        # Apply creative_levels changes
        if 'creative_levels' in changes:
            cl_changes = changes['creative_levels']
            if 'surface' in cl_changes:
                modified_brief.creative_levels.surface = float(cl_changes['surface'])
            if 'flow' in cl_changes:
                modified_brief.creative_levels.flow = float(cl_changes['flow'])
            if 'narrative' in cl_changes:
                modified_brief.creative_levels.narrative = float(cl_changes['narrative'])

        # Apply essence changes
        if 'essence' in changes:
            modified_brief.essence = changes['essence']

        # Apply workflow description changes
        if 'node_workflow_description' in changes:
            modified_brief.node_workflow_description = changes['node_workflow_description']

        # UI Controls changes
        if 'ui_controls' in changes:
            ui_changes = changes['ui_controls']

            # Handle list of changes (for adding multiple controls)
            if isinstance(ui_changes, list):
                from data_types import UIControl
                for ui_change in ui_changes:
                    if isinstance(ui_change, dict):
                        action = ui_change.get('action', 'add')

                        if action == 'add':
                            # Add new control
                            new_control = UIControl(
                                id=f"ctrl_{len(modified_brief.ui_controls)}",
                                label=ui_change.get('label', 'New Control'),
                                type=ui_change.get('type', 'slider'),
                                creative_level=ui_change.get('creative_level', 'surface'),
                                parameters=ui_change.get('parameters', {'min': 0.0, 'max': 1.0, 'default': 0.5, 'step': 0.01}),
                                grounding_source='user_edit',
                                confidence=0.9
                            )
                            modified_brief.ui_controls.append(new_control)

                        elif action == 'remove' and 'index' in ui_change:
                            idx = int(ui_change['index'])
                            if 0 <= idx < len(modified_brief.ui_controls):
                                modified_brief.ui_controls.pop(idx)

                        elif action == 'modify' and 'index' in ui_change:
                            idx = int(ui_change['index'])
                            if 0 <= idx < len(modified_brief.ui_controls):
                                control = modified_brief.ui_controls[idx]
                                if 'label' in ui_change:
                                    control.label = ui_change['label']
                                if 'parameters' in ui_change:
                                    control.parameters.update(ui_change['parameters'])

            # Handle single change (dict)
            elif isinstance(ui_changes, dict):
                action = ui_changes.get('action', 'add')

                if action == 'add':
                    # Add new control
                    from data_types import UIControl
                    new_control = UIControl(
                        id=f"ctrl_{len(modified_brief.ui_controls)}",
                        label=ui_changes.get('label', 'New Control'),
                        type=ui_changes.get('type', 'slider'),
                        creative_level=ui_changes.get('creative_level', 'surface'),
                        parameters=ui_changes.get('parameters', {'min': 0.0, 'max': 1.0, 'default': 0.5, 'step': 0.01}),
                        grounding_source='user_edit',
                        confidence=0.9
                    )
                    modified_brief.ui_controls.append(new_control)

                elif action == 'remove' and 'index' in ui_changes:
                    idx = int(ui_changes['index'])
                    if 0 <= idx < len(modified_brief.ui_controls):
                        modified_brief.ui_controls.pop(idx)

                elif action == 'modify' and 'index' in ui_changes:
                    idx = int(ui_changes['index'])
                    if 0 <= idx < len(modified_brief.ui_controls):
                        control = modified_brief.ui_controls[idx]
                        if 'label' in ui_changes:
                            control.label = ui_changes['label']
                        if 'parameters' in ui_changes:
                            control.parameters.update(ui_changes['parameters'])

        # Node Archetypes changes
        if 'node_archetypes' in changes:
            node_changes = changes['node_archetypes']

            # Handle list of changes (for adding multiple nodes)
            if isinstance(node_changes, list):
                from data_types import NodeArchetype
                for node_change in node_changes:
                    if isinstance(node_change, dict):
                        action = node_change.get('action', 'add')

                        if action == 'add':
                            # Add new node
                            new_node = NodeArchetype(
                                id=f"node_{len(modified_brief.node_archetypes)}",
                                name=node_change.get('name', 'New Node'),
                                category=node_change.get('category', 'generator'),
                                role=node_change.get('role', 'custom'),
                                description=node_change.get('description', 'User-added node'),
                                parameters=node_change.get('parameters', {}),
                                grounding_source='user_edit',
                                confidence=0.9
                            )
                            modified_brief.node_archetypes.append(new_node)
                            modified_brief.node_count = len(modified_brief.node_archetypes)

                        elif action == 'remove' and 'index' in node_change:
                            idx = int(node_change['index'])
                            if 0 <= idx < len(modified_brief.node_archetypes):
                                modified_brief.node_archetypes.pop(idx)
                                modified_brief.node_count = len(modified_brief.node_archetypes)

                        elif action == 'modify' and 'index' in node_change:
                            idx = int(node_change['index'])
                            if 0 <= idx < len(modified_brief.node_archetypes):
                                node = modified_brief.node_archetypes[idx]
                                if 'name' in node_change:
                                    node.name = node_change['name']
                                if 'description' in node_change:
                                    node.description = node_change['description']
                                if 'parameters' in node_change:
                                    node.parameters.update(node_change['parameters'])

            # Handle single change (dict)
            elif isinstance(node_changes, dict):
                action = node_changes.get('action', 'add')

                if action == 'add':
                    # Add new node
                    from data_types import NodeArchetype
                    new_node = NodeArchetype(
                        id=f"node_{len(modified_brief.node_archetypes)}",
                        name=node_changes.get('name', 'New Node'),
                        category=node_changes.get('category', 'generator'),
                        role=node_changes.get('role', 'custom'),
                        description=node_changes.get('description', 'User-added node'),
                        parameters=node_changes.get('parameters', {}),
                        grounding_source='user_edit',
                        confidence=0.9
                    )
                    modified_brief.node_archetypes.append(new_node)
                    modified_brief.node_count = len(modified_brief.node_archetypes)

                elif action == 'remove' and 'index' in node_changes:
                    idx = int(node_changes['index'])
                    if 0 <= idx < len(modified_brief.node_archetypes):
                        modified_brief.node_archetypes.pop(idx)
                        modified_brief.node_count = len(modified_brief.node_archetypes)

                elif action == 'modify' and 'index' in node_changes:
                    idx = int(node_changes['index'])
                    if 0 <= idx < len(modified_brief.node_archetypes):
                        node = modified_brief.node_archetypes[idx]
                        if 'name' in node_changes:
                            node.name = node_changes['name']
                        if 'description' in node_changes:
                            node.description = node_changes['description']
                        if 'parameters' in node_changes:
                            node.parameters.update(node_changes['parameters'])

        return modified_brief

    def _get_component_data(self, brief: NodeBrief, component: str) -> Dict:
        """Extract component data from brief"""
        if component == 'creative_levels':
            return {
                'surface': brief.creative_levels.surface,
                'flow': brief.creative_levels.flow,
                'narrative': brief.creative_levels.narrative
            }
        elif component == 'divergence':
            return {
                'D_surface': brief.divergence.D_surface,
                'D_flow': brief.divergence.D_flow,
                'D_narrative': brief.divergence.D_narrative
            }
        elif component == 'ui_controls':
            return {
                'count': len(brief.ui_controls),
                'controls': [
                    {'index': i, 'label': c.label, 'type': c.type, 'level': c.creative_level, 'parameters': c.parameters}
                    for i, c in enumerate(brief.ui_controls[:10])
                ]
            }
        elif component == 'node_archetypes' or component == 'nodes':
            return {
                'count': brief.node_count,
                'archetypes': [
                    {'index': i, 'name': n.name, 'category': n.category, 'role': n.role, 'description': n.description[:100]}
                    for i, n in enumerate(brief.node_archetypes[:10])
                ]
            }
        elif component == 'visual_palette' or component == 'palette':
            return {
                'primary_colors': brief.visual_palette.primary_colors,
                'accent_colors': brief.visual_palette.accent_colors,
                'shapes': brief.visual_palette.shapes,
                'motion_words': brief.visual_palette.motion_words
            }
        elif component == 'brand_values' or component == 'brand':
            return {
                'color_palette': brief.brand_values.color_palette,
                'visual_mood': brief.brand_values.visual_mood,
                'emotions': dict(list(brief.brand_values.emotions.items())[:5]),
                'brand_attributes': dict(list(brief.brand_values.brand_attributes.items())[:5])
            }
        elif component == 'essence':
            return {'essence': brief.essence}
        elif component == 'workflow':
            return {'description': brief.node_workflow_description}
        else:
            return {}

    def _extract_sources(self, brief: NodeBrief, component: str) -> list:
        """Extract grounding sources for component"""
        sources = []
        if component == 'ui_controls':
            sources = list(set([c.grounding_source for c in brief.ui_controls]))
        elif component == 'node_archetypes':
            sources = list(set([n.grounding_source for n in brief.node_archetypes]))
        return sources[:5]
