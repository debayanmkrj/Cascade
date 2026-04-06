"""
Design Assistant Copilot — LLM tool-calling for pipeline control.

Interprets natural language and calls structured tools to modify the session
JSON and/or output project JSON. Uses the reasoning model for tool selection
and the coding model (via Mason) for all code generation.
"""

import copy
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import EFFECTIVE_MODEL_REASONING


# ---------------------------------------------------------------------------
# Tool schema definitions (OpenAI function-calling format)
# ---------------------------------------------------------------------------

COPILOT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "create_node",
            "description": (
                "Add a brand-new node to the pipeline. Call this whenever the user asks to "
                "create, add, make, or build ANY new visual effect, input source, audio processor, "
                "or utility — even if a node with a similar name already exists. "
                "NEVER refuse a create request by saying 'a similar node already exists'. "
                "Mason generates fresh code for each node — always call this."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": (
                            "A descriptive snake_case name derived DIRECTLY from what the user asked for. "
                            "Examples: 'kaleidoscope' → 'kaleidoscope_mirror_effect'; "
                            "'another neon glow' → 'neon_radial_bloom'; "
                            "'underwater' → 'underwater_ripple_distortion'. "
                            "NEVER use generic names like 'color_grade', 'blur', 'effect', 'node'. "
                            "If a node with the same category already exists, suffix with '_v2', '_alt', etc."
                        )
                    },
                    "role": {
                        "type": "string",
                        "enum": ["source", "process", "output"],
                        "description": "Node role: 'source' for inputs, 'process' for effects, 'output' for final compositing"
                    },
                    "description": {
                        "type": "string",
                        "description": "What this node should do — used by Mason to generate the code"
                    },
                    "engine_hint": {
                        "type": "string",
                        "enum": ["glsl", "canvas2d", "three_js", "p5", "webaudio", "html_video"],
                        "description": (
                            "Rendering engine — choose based on what the node does:\n"
                            "- p5: shapes, geometry, generative patterns, particle systems, drawing, sketches, "
                            "organic forms, motion graphics, anything procedurally drawn on a canvas.\n"
                            "- glsl: pixel/fragment shaders, texture effects, colour grading, blur, glow, "
                            "distortion, noise textures, image post-processing.\n"
                            "- three_js: 3D objects, meshes, 3D scenes, perspective geometry.\n"
                            "- canvas2d: simple 2D compositing, image blending.\n"
                            "- webaudio: audio analysis, synthesis, FFT, waveforms.\n"
                            "- html_video: webcam or video file input.\n"
                            "If the user specifies an engine explicitly, always use that. "
                            "When in doubt between p5 and glsl: prefer p5 for anything involving "
                            "drawing shapes or geometry; prefer glsl for pixel-level image effects."
                        )
                    }
                },
                "required": ["category", "role", "description"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "duplicate_node",
            "description": (
                "Make an exact copy of an existing node, appending it to the pipeline. "
                "Use when the user asks to duplicate, copy, or clone a node."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {
                        "type": "string",
                        "description": "The node ID to copy, e.g. 'node_0_n0'"
                    }
                },
                "required": ["node_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_node_params",
            "description": (
                "Change how an existing node behaves by describing the parameter change in plain language. "
                "This re-examines the node's code and regenerates it with the updated parameter values. "
                "Use for: 'make it faster', 'reduce the blur', 'increase glow intensity', 'change colour to red'. "
                "Do NOT pass raw key-value pairs — describe the change as intent."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {
                        "type": "string",
                        "description": "The node ID to update, e.g. 'node_0_n0'"
                    },
                    "intent": {
                        "type": "string",
                        "description": "Plain-language description of the change, e.g. 'increase speed to 0.9 and reduce blur to 1.0'"
                    }
                },
                "required": ["node_id", "intent"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "delete_node",
            "description": "Remove a node from the pipeline and clean up its connections.",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {
                        "type": "string",
                        "description": "The node ID to delete, e.g. 'node_0_n0'"
                    }
                },
                "required": ["node_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "move_node",
            "description": "Change a node's z-layer (compositing depth) in the pipeline.",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {
                        "type": "string",
                        "description": "The node ID to move"
                    },
                    "z_layer": {
                        "type": "integer",
                        "description": "New z-layer index. 0 = bottom/source layer, higher = later in compositing stack."
                    }
                },
                "required": ["node_id", "z_layer"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_visual_concept",
            "description": (
                "Modify the core design concept or prompt. Use when the user wants to change "
                "the overall aesthetic direction, mood, or theme. Triggers a full pipeline re-run."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "new_prompt": {
                        "type": "string",
                        "description": "The new or updated design prompt describing the visual concept"
                    }
                },
                "required": ["new_prompt"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "regenerate_node",
            "description": (
                "Regenerate the code for a specific existing node using Mason. "
                "Use when the user wants to change how a node looks or behaves at the code level. "
                "If the user says 'regenerate this', 'redo this node', or 'change this to X' "
                "without specifying a node, use the currently selected node ID."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {
                        "type": "string",
                        "description": "The node ID to regenerate"
                    },
                    "intent": {
                        "type": "string",
                        "description": "What the regenerated node should do, e.g. 'react to audio bass frequency', 'add chromatic aberration'"
                    }
                },
                "required": ["node_id", "intent"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "respond",
            "description": (
                "Send a response message to the user. ALWAYS call this last after taking all actions. "
                "Summarise what was done in plain, friendly language."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The response to show the user"
                    }
                },
                "required": ["message"]
            }
        }
    },
]


# ---------------------------------------------------------------------------
# DesignCopilot
# ---------------------------------------------------------------------------

class DesignCopilot:
    """
    AI copilot for the Design Assistant panel.

    Receives natural language from the user + current session/project context,
    calls tools via the active reasoning model to take real pipeline actions,
    and returns the updated state. All code generation is delegated to Mason
    which uses the active coding model.
    """

    def __init__(self):
        from phase2.aider_llm import get_aider_llm
        self.aider = get_aider_llm()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def process(
        self,
        user_message: str,
        session_json: Optional[Dict],
        project: Optional[Dict],
        selected_node_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Interpret user message and take pipeline actions via tool-calling.

        Returns:
            {
                "success": bool,
                "response": str,
                "project": dict,          # updated output project JSON
                "session_json": dict,     # updated session JSON
                "pipeline_needed": bool,  # True if Phase 2 should be re-run
            }
        """
        session_json = session_json or {}
        project = project or {}

        system = self._build_system_prompt(session_json, project, selected_node_id)

        state = {
            "session_json": copy.deepcopy(session_json),
            "project": copy.deepcopy(project),
            "pipeline_needed": False,
            "response": "Done.",
            "_actions_taken": 0,
        }

        handlers = self._make_handlers(state)

        try:
            result = self.aider.call_with_tools(
                system_prompt=system,
                user_prompt=user_message,
                model_name=EFFECTIVE_MODEL_REASONING,
                tools=COPILOT_TOOLS,
                tool_handlers=handlers,
                max_turns=8,
            )

            if result.final_text and state["response"] == "Done.":
                final = result.final_text.strip()
                _creation_words = ("create", "add", "make", "build", "new node", "generate", "duplicate", "copy")
                is_create_request = any(w in user_message.lower() for w in _creation_words)
                no_tools_used = not result.tool_results
                if is_create_request and no_tools_used and final.lower() in ("done.", "done", ""):
                    state["response"] = (
                        "I didn't take any action. Please try rephrasing, e.g.: "
                        "'create a kaleidoscope_mirror_effect node using glsl'."
                    )
                else:
                    state["response"] = final

        except Exception as e:
            print(f"[DesignCopilot] LLM error: {e}")
            state["response"] = f"I encountered an error: {e}"

        return {
            "success": True,
            "response": state["response"],
            "project": state["project"],
            "session_json": state["session_json"],
            "pipeline_needed": state["pipeline_needed"],
        }

    # ------------------------------------------------------------------
    # System prompt builder
    # ------------------------------------------------------------------

    def _build_system_prompt(
        self,
        session_json: Dict,
        project: Dict,
        selected_node_id: Optional[str],
    ) -> str:
        lines = [
            "You are the Design Assistant copilot for a visual node-graph pipeline tool.",
            "Interpret the user's request intelligently — no keywords or special syntax needed.",
            "Reason about what the user wants, then take the right actions using tools.",
            "",
        ]

        prompt_text = (
            session_json.get("input", {}).get("prompt_text")
            or project.get("design_brief", {}).get("prompt_text")
            or "(no prompt)"
        )
        lines.append(f'CURRENT DESIGN PROMPT: "{prompt_text}"')
        lines.append("")

        nodes = project.get("nodes", [])
        if nodes:
            lines.append(f"PIPELINE NODES ({len(nodes)} total):")
            for n in nodes:
                nid = n.get("id", "?")
                cat = n.get("category", n.get("meta", {}).get("category", "?"))
                eng = n.get("engine", "?")
                role = n.get("role", n.get("meta", {}).get("role", "?"))
                approved = "✓" if n.get("mason_approved") else "pending"
                lines.append(f"  {nid}: {cat} ({eng}, {role}) {approved}")
        else:
            archetypes = (
                session_json.get("brief", {}).get("node_archetypes")
                or session_json.get("phase2_context", {}).get("node_archetypes")
                or []
            )
            if archetypes:
                lines.append(f"SESSION ARCHETYPES ({len(archetypes)} total, not yet compiled):")
                for a in archetypes:
                    lines.append(f"  {a.get('id','?')}: {a.get('category','?')} ({a.get('role','?')})")
            else:
                lines.append("PIPELINE: empty (no nodes yet)")
        lines.append("")

        conns = project.get("connections", [])
        if conns:
            conn_strs = [f"{c.get('from_node','?')} → {c.get('to_node','?')}" for c in conns[:8]]
            lines.append("CONNECTIONS: " + ", ".join(conn_strs))
            lines.append("")

        # Selected node — explicit default target for regenerate/update
        if selected_node_id:
            focused = next((n for n in nodes if n.get("id") == selected_node_id), None)
            if focused:
                cat = focused.get("category", focused.get("meta", {}).get("category", "?"))
                eng = focused.get("engine", "?")
                desc = focused.get("meta", {}).get("description", "")
                params = focused.get("parameters", {})
                lines.append(f"SELECTED NODE (user is focused on this): {selected_node_id} — {cat} ({eng})")
                if desc:
                    lines.append(f"  Description: {desc}")
                if params:
                    lines.append(f"  Parameters: {json.dumps(params)}")
                lines.append("")

        lines += [
            "TOOL WORKFLOW:",
            "- User says create/add/make/build/duplicate ANYTHING → call the appropriate tool immediately.",
            f"- User says 'regenerate this', 'redo this', 'change this to X' (no node specified) → regenerate_node(node_id='{selected_node_id or 'NODE_ID'}', intent=...)",
            f"- User says 'update params', 'make it faster', 'change colour' (no node specified) → update_node_params(node_id='{selected_node_id or 'NODE_ID'}', intent=...)",
            "- NEVER say 'a node with that name already exists' — always create a new one with a distinct category.",
            "  Example: neon_glow exists + user asks for another neon node → create 'neon_pulse_scanner' or 'neon_radial_bloom'.",
            "- Remove a node → delete_node",
            "- Reorder z-layers → move_node",
            "- Redesign the whole concept → update_visual_concept",
            "- ALWAYS call respond() last to explain what you did.",
            "",
            "IMPORTANT — tool call format:",
            '{"name": "create_node", "parameters": {"category": "kaleidoscope_mirror_effect", "role": "process", "description": "GLSL kaleidoscope that mirrors the image radially", "engine_hint": "glsl"}}',
            "One tool call per message. No prose before or after the JSON.",
            "",
            "Be concise in respond(). Don't ask follow-up questions.",
        ]

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Tool handlers
    # ------------------------------------------------------------------

    def _make_handlers(self, state: Dict) -> Dict[str, Any]:

        def h_create_node(category: str, role: str, description: str,
                          engine_hint: str = "glsl") -> str:
            if state["_actions_taken"] > 0:
                return "ACTION ALREADY DONE. Call respond() now."
            print(f"[DesignCopilot] create_node: category='{category}', engine='{engine_hint}', role='{role}'")

            # Ensure unique category if a duplicate exists
            existing_cats = {n.get("category", "") for n in state["project"].get("nodes", [])}
            base_cat = category
            suffix = 2
            while category in existing_cats:
                category = f"{base_cat}_v{suffix}"
                suffix += 1

            new_node, err = self._gen_new_node(category, role, description, engine_hint, state)
            if err:
                return f"Failed to create '{category}': {err}"

            state["project"].setdefault("nodes", []).append(new_node)

            nodes = state["project"]["nodes"]
            if len(nodes) > 1 and new_node["input_nodes"]:
                state["project"].setdefault("connections", []).append({
                    "from_node": new_node["input_nodes"][0],
                    "to_node": new_node["id"],
                    "type": "texture",
                })

            state["_actions_taken"] += 1
            return f"Node '{category}' ({engine_hint}) created and validated. ACTION COMPLETE — call respond() now."

        def h_duplicate_node(node_id: str) -> str:
            if state["_actions_taken"] > 0:
                return "ACTION ALREADY DONE. Call respond() now."
            nodes = state["project"].get("nodes", [])
            source = next((n for n in nodes if n.get("id") == node_id), None)
            if not source:
                return f"Node '{node_id}' not found."

            new_node = copy.deepcopy(source)
            # Assign unique ID at next z-layer
            z_layer = len(nodes)
            existing_ids = {n["id"] for n in nodes}
            idx = z_layer
            new_id = f"node_{z_layer}_n{idx}"
            while new_id in existing_ids:
                idx += 1
                new_id = f"node_{z_layer}_n{idx}"

            new_node["id"] = new_id
            gp = list(new_node.get("grid_position", [0, 0, 0]))
            if len(gp) >= 3:
                gp[2] = z_layer
            new_node["grid_position"] = gp
            new_node["input_nodes"] = [source["id"]]

            state["project"]["nodes"].append(new_node)
            state["project"].setdefault("connections", []).append({
                "from_node": source["id"],
                "to_node": new_id,
                "type": "texture",
            })
            state["_actions_taken"] += 1
            return f"Duplicated '{node_id}' → '{new_id}'. ACTION COMPLETE — call respond() now."

        def h_update_node_params(node_id: str, intent: str) -> str:
            if state["_actions_taken"] > 0:
                return "ACTION ALREADY DONE. Call respond() now."
            nodes = state["project"].get("nodes", [])
            node = next((n for n in nodes if n.get("id") == node_id), None)
            if not node:
                return f"Node '{node_id}' not found."
            # Delegate to regenerate with the param-change intent so Mason
            # re-examines the actual code and produces valid parameter values.
            result = self._regen_single_node(node_id, intent, state)
            state["_actions_taken"] += 1
            return result + " ACTION COMPLETE — call respond() now."

        def h_delete_node(node_id: str) -> str:
            if state["_actions_taken"] > 0:
                return "ACTION ALREADY DONE. Call respond() now."
            nodes = state["project"].get("nodes", [])
            before = len(nodes)
            state["project"]["nodes"] = [n for n in nodes if n.get("id") != node_id]
            if len(state["project"]["nodes"]) == before:
                return f"Node '{node_id}' not found."
            conns = state["project"].get("connections", [])
            state["project"]["connections"] = [
                c for c in conns
                if c.get("from_node") != node_id and c.get("to_node") != node_id
            ]
            state["_actions_taken"] += 1
            return f"Deleted node {node_id}. ACTION COMPLETE — call respond() now."

        def h_move_node(node_id: str, z_layer: int) -> str:
            if state["_actions_taken"] > 0:
                return "ACTION ALREADY DONE. Call respond() now."
            nodes = state["project"].get("nodes", [])
            node = next((n for n in nodes if n.get("id") == node_id), None)
            if not node:
                return f"Node '{node_id}' not found."
            gp = list(node.get("grid_position", [0, 0, 0]))
            if len(gp) >= 3:
                gp[2] = z_layer
            node["grid_position"] = gp
            state["_actions_taken"] += 1
            return f"Moved {node_id} to z-layer {z_layer}. ACTION COMPLETE — call respond() now."

        def h_update_visual_concept(new_prompt: str) -> str:
            state["session_json"].setdefault("input", {})["prompt_text"] = new_prompt
            state["project"].setdefault("design_brief", {})["prompt_text"] = new_prompt
            state["pipeline_needed"] = True
            return "Visual concept updated. Pipeline will re-run."

        def h_regenerate_node(node_id: str, intent: str) -> str:
            if state["_actions_taken"] > 0:
                return "ACTION ALREADY DONE. Call respond() now."
            state["_actions_taken"] += 1
            result = self._regen_single_node(node_id, intent, state)
            return result + " ACTION COMPLETE — call respond() now."

        def h_respond(message: str) -> str:
            state["response"] = message
            return "Response recorded."

        return {
            "create_node": h_create_node,
            "duplicate_node": h_duplicate_node,
            "update_node_params": h_update_node_params,
            "delete_node": h_delete_node,
            "move_node": h_move_node,
            "update_visual_concept": h_update_visual_concept,
            "regenerate_node": h_regenerate_node,
            "respond": h_respond,
        }

    # ------------------------------------------------------------------
    # Targeted Mason regen for a single node
    # ------------------------------------------------------------------

    def _regen_single_node(self, node_id: str, intent: str, state: Dict) -> str:
        """Run Mason on a single existing node with a new intent.
        Only writes back to project if mason_approved=True and no validation errors.
        """
        try:
            from phase2.agents.mason import MasonAgent
            from phase2.data_types import NodeTensor, BuildSheet

            nodes = state["project"].get("nodes", [])
            node_data = next((n for n in nodes if n.get("id") == node_id), None)
            if not node_data:
                return f"Node '{node_id}' not found in current project."

            meta_raw = node_data.get("meta", {})
            node_tensor = NodeTensor(
                id=node_data["id"],
                meta=meta_raw,
                grid_position=tuple(node_data.get("grid_position", [0, 0, 0])),
                grid_size=tuple(node_data.get("grid_size", [1, 1])),
                engine=node_data.get("engine", "glsl"),
                code_snippet=node_data.get("code_snippet", ""),
                parameters=node_data.get("parameters", {}),
                input_nodes=node_data.get("input_nodes", []),
                keywords=node_data.get("keywords", []),
                semantic_purpose=intent,
                architect_approved=True,
                mason_approved=False,
                validation_errors=[],
            )

            palette = (
                state["session_json"].get("brief", {}).get("visual_palette")
                or state["project"].get("design_brief", {}).get("visual_palette")
                or {}
            )
            sheet = BuildSheet(
                node_id=node_id,
                engine=node_data.get("engine", "glsl"),
                intent=intent,
                inputs=[],
                influence_rules={"must_use": [], "allow": [], "avoid": []},
                output_protocol=meta_raw.get("output_protocol", "COLOR_RGBA"),
                style_anchor=palette,
                params=list(node_data.get("parameters", {}).keys()),
                grid_position=tuple(node_data.get("grid_position", [0, 0, 0])),
                z_total=1,
                z_role=node_data.get("role", "process"),
            )

            mason = MasonAgent()
            updated = mason.generate_from_build_sheets(
                [node_tensor], {node_id: sheet}, palette
            )

            if not updated or not updated[0].code_snippet:
                return f"Mason produced no code for '{node_id}'."

            result_node = updated[0]

            # Validation gate — only apply if fully approved
            if not result_node.mason_approved or result_node.validation_errors:
                errs = result_node.validation_errors
                return (
                    f"Regeneration for '{node_id}' failed validation — not applied to pipeline. "
                    f"Errors: {errs}"
                )

            for n in state["project"]["nodes"]:
                if n.get("id") == node_id:
                    n["code_snippet"] = result_node.code_snippet
                    n["parameters"] = result_node.parameters
                    n["mason_approved"] = True
                    n["validation_errors"] = []
                    break

            return f"Node '{node_id}' regenerated and validated successfully."

        except Exception as e:
            print(f"[DesignCopilot] _regen_single_node error: {e}")
            return f"Regeneration failed: {e}"

    # ------------------------------------------------------------------
    # Generate a brand-new node via Mason
    # ------------------------------------------------------------------

    def _gen_new_node(
        self,
        category: str,
        role: str,
        description: str,
        engine_hint: str,
        state: Dict,
    ):
        """Run Mason on a brand-new node.
        Returns (node_dict, None) on success, or (None, error_str) on failure.
        Only returns a node if mason_approved=True and validation_errors is empty.
        """
        try:
            from phase2.agents.mason import MasonAgent
            from phase2.data_types import NodeTensor, BuildSheet

            nodes = state["project"].get("nodes", [])
            z_layer = len(nodes)
            last_node = nodes[-1] if nodes and role != "source" else None
            input_nodes = [last_node["id"]] if last_node else []

            existing_ids = {n["id"] for n in nodes}
            idx = z_layer
            node_id = f"node_{z_layer}_n{idx}"
            while node_id in existing_ids:
                idx += 1
                node_id = f"node_{z_layer}_n{idx}"

            palette = (
                state["session_json"].get("brief", {}).get("visual_palette")
                or state["project"].get("design_brief", {}).get("visual_palette")
                or {}
            )

            input_dicts = []
            if last_node:
                src_z = last_node.get("grid_position", [0, 0, 0])
                src_z = src_z[2] if len(src_z) > 2 else 0
                src_intent = (last_node.get("meta") or {}).get("description", "")
                input_dicts = [{
                    "name": last_node["id"],
                    "protocol": "COLOR_RGBA",
                    "meaning": "upstream texture from previous z-layer",
                    "source_intent": src_intent,
                    "source_z": src_z,
                }]

            node_tensor = NodeTensor(
                id=node_id,
                meta={
                    "category": category, "role": role, "description": description,
                    "engine_hint": engine_hint, "output_protocol": "COLOR_RGBA",
                    "label": category.replace("_", " ").title(),
                    "keywords": [category],
                },
                grid_position=(0, 0, z_layer),
                grid_size=(1, 1),
                engine=engine_hint,
                code_snippet="",
                parameters={},
                input_nodes=input_nodes,
                keywords=[category],
                semantic_purpose=description,
                architect_approved=True,
                mason_approved=False,
                validation_errors=[],
            )

            sheet = BuildSheet(
                node_id=node_id,
                engine=engine_hint,
                intent=description,
                inputs=input_dicts,
                influence_rules={"must_use": [], "allow": [], "avoid": []},
                output_protocol="COLOR_RGBA",
                style_anchor=palette,
                params=[],
                grid_position=(0, 0, z_layer),
                z_total=1,
                z_role=role,
            )

            mason = MasonAgent()
            updated = mason.generate_from_build_sheets(
                [node_tensor], {node_id: sheet}, palette
            )

            if not updated or not updated[0].code_snippet:
                return None, f"Mason generated no code for '{category}'."

            g = updated[0]

            # Validation gate — only return node if fully approved
            if not g.mason_approved or g.validation_errors:
                return None, (
                    f"Node '{category}' failed validation and was not added to the pipeline. "
                    f"Errors: {g.validation_errors}"
                )

            new_node = {
                "id": node_id,
                "category": category,
                "engine": engine_hint,
                "role": role,
                "meta": node_tensor.meta,
                "parameters": g.parameters,
                "grid_position": list(g.grid_position),
                "code_snippet": g.code_snippet,
                "input_nodes": input_nodes,
                "mason_approved": True,
                "validation_errors": [],
            }
            return new_node, None

        except Exception as e:
            print(f"[DesignCopilot] _gen_new_node error: {e}")
            return None, str(e)


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_copilot_instance = None


def get_design_copilot() -> DesignCopilot:
    global _copilot_instance
    if _copilot_instance is None:
        print("[INIT] Loading Design Copilot...")
        _copilot_instance = DesignCopilot()
    return _copilot_instance
