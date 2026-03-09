"""
Design Assistant Copilot — LLM tool-calling via llama3.2

Replaces the keyword-based _try_programmatic_modification() in app_web.py.
The LLM interprets natural language and calls structured tools to modify
the session JSON and/or output project JSON.
"""

import copy
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import MODEL_NAME


# ---------------------------------------------------------------------------
# Tool schema definitions (OpenAI function-calling format)
# ---------------------------------------------------------------------------

COPILOT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "create_node",
            "description": (
                "Add a new node to the pipeline. Use when the user asks to create, add, make, or build "
                "ANY new visual effect, input source, audio processor, or utility. "
                "Mason generates custom code for this node — call this even if the user's request "
                "sounds like something that could be done with an existing node."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": (
                            "A descriptive snake_case name derived DIRECTLY from what the user asked for. "
                            "Examples: user says 'kaleidoscope' → 'kaleidoscope_mirror_effect'; "
                            "'underwater' → 'underwater_ripple_distortion'; "
                            "'glitch' → 'glitch_scan_lines'; 'particles' → 'particle_burst_emitter'. "
                            "NEVER use generic names like 'color_grade', 'blur', 'effect', 'node', 'process'. "
                            "The category drives Mason's code generation — make it specific."
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
                        "description": "Preferred rendering engine. Use glsl for visual shaders, canvas2d or p5 for JS-based effects, webaudio for audio."
                    }
                },
                "required": ["category", "role", "description"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_node_params",
            "description": (
                "Change parameter values for a specific existing node. Use for adjusting speed, "
                "scale, intensity, color, blur amount, opacity, threshold, etc. Does not regenerate code."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {
                        "type": "string",
                        "description": "The node ID, e.g. 'node_0_n0'"
                    },
                    "parameters": {
                        "type": "object",
                        "description": "Key-value pairs to update, e.g. {\"speed\": 0.8, \"blur\": 2.0, \"opacity\": 0.5}"
                    }
                },
                "required": ["node_id", "parameters"]
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
            "name": "update_palette",
            "description": (
                "Change the visual palette: colors, shapes, or motion words. "
                "Takes effect immediately without a pipeline re-run."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "primary_colors": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of hex color strings for primary palette, e.g. [\"#1e293b\", \"#e2e8f0\"]"
                    },
                    "accent_colors": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of hex accent color strings"
                    },
                    "shapes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of shape keywords, e.g. [\"circles\", \"grid\", \"organic\"]"
                    },
                    "motion_words": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of motion descriptors, e.g. [\"pulsing\", \"flowing\", \"glitchy\"]"
                    }
                }
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
                "Use when the user wants to change how a node looks or behaves at the code level, "
                "not just its parameters."
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
                        "description": "What the regenerated node should do differently, e.g. 'react to audio bass frequency', 'create chromatic aberration'"
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
                "Summarize what was done in plain, friendly language."
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
    calls tools via llama3.2 to take real pipeline actions, and returns the
    updated state.
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
            "_actions_taken": 0,  # guard: prevent duplicate tool calls in one turn
        }

        handlers = self._make_handlers(state)

        try:
            result = self.aider.call_with_tools(
                system_prompt=system,
                user_prompt=user_message,
                model_name=MODEL_NAME,  # bare name — aider_llm adds ollama_chat/ prefix
                tools=COPILOT_TOOLS,
                tool_handlers=handlers,
                max_turns=8,
            )

            # Fallback: if LLM returned plain text but never called respond()
            if result.final_text and state["response"] == "Done.":
                final = result.final_text.strip()
                # Detect vacuous "Done." with no tool calls on a creation request
                _creation_words = ("create", "add", "make", "build", "new node", "generate")
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

        # --- Pipeline summary ---
        prompt_text = (
            session_json.get("input", {}).get("prompt_text")
            or project.get("design_brief", {}).get("prompt_text")
            or "(no prompt)"
        )
        lines.append(f'CURRENT DESIGN PROMPT: "{prompt_text}"')
        lines.append("")

        # Nodes summary
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
            # Fall back to session archetypes if no project nodes yet
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

        # Connections
        conns = project.get("connections", [])
        if conns:
            conn_strs = [f"{c.get('from_node','?')} → {c.get('to_node','?')}" for c in conns[:8]]
            lines.append("CONNECTIONS: " + ", ".join(conn_strs))
            lines.append("")

        # Palette
        vp = (
            session_json.get("brief", {}).get("visual_palette")
            or project.get("design_brief", {}).get("visual_palette")
            or {}
        )
        if vp:
            pc = vp.get("primary_colors", [])
            ac = vp.get("accent_colors", [])
            sh = vp.get("shapes", [])
            mw = vp.get("motion_words", [])
            palette_parts = []
            if pc:
                palette_parts.append(f"primary={pc[:4]}")
            if ac:
                palette_parts.append(f"accent={ac[:3]}")
            if sh:
                palette_parts.append(f"shapes={sh[:4]}")
            if mw:
                palette_parts.append(f"motion={mw[:4]}")
            if palette_parts:
                lines.append("PALETTE: " + ", ".join(palette_parts))
                lines.append("")

        # Focused node (selected in UI)
        if selected_node_id:
            focused = next((n for n in nodes if n.get("id") == selected_node_id), None)
            if focused:
                cat = focused.get("category", focused.get("meta", {}).get("category", "?"))
                eng = focused.get("engine", "?")
                desc = focused.get("meta", {}).get("description", "")
                lines.append(f"USER IS FOCUSED ON: {selected_node_id} — {cat} ({eng})")
                if desc:
                    lines.append(f"  Description: {desc}")
                params = focused.get("parameters", {})
                if params:
                    lines.append(f"  Parameters: {json.dumps(params)}")
                lines.append("")

        # Workflow instructions
        lines += [
            "TOOL WORKFLOW:",
            "- User says create/add/make/build ANYTHING new → create_node immediately",
            "  (Set category from exactly what the user said, e.g. 'kaleidoscope_mirror_effect')",
            "  (Do NOT assume an existing node already does this — always call create_node)",
            "- Adjust existing node params → update_node_params or regenerate_node",
            "- Colors / shapes / motion feel → update_palette",
            "- Remove a node → delete_node",
            "- Reorder z-layers → move_node",
            "- Redesign the whole concept → update_visual_concept",
            "- ALWAYS call respond() last to explain what you did.",
            "",
            "IMPORTANT — tool call format. Respond with a JSON object like this:",
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
        """Return tool name → handler function dict. Handlers mutate state in-place."""

        def h_create_node(category: str, role: str, description: str,
                          engine_hint: str = "glsl") -> str:
            if state["_actions_taken"] > 0:
                return "ACTION ALREADY DONE. Stop calling tools. Call respond() now to tell the user what you did."
            print(f"[DesignCopilot] create_node called: category='{category}', engine='{engine_hint}', role='{role}'")
            print(f"[DesignCopilot]   description: {description[:120]}")
            new_node, err = self._gen_new_node(category, role, description, engine_hint, state)
            if err:
                return f"Failed to create '{category}': {err}"

            state["project"].setdefault("nodes", []).append(new_node)

            # Wire connection from last existing node → new node (skip for source nodes)
            nodes = state["project"]["nodes"]
            if len(nodes) > 1 and new_node["input_nodes"]:
                state["project"].setdefault("connections", []).append({
                    "from_node": new_node["input_nodes"][0],
                    "to_node": new_node["id"],
                    "type": "texture",
                })

            # NOTE: Do NOT add archetype to session_json.
            # Phase 2 Reasoner assigns z-layers by graph topology, not index, so a
            # reconstructed archetype would get a different z-layer → different id →
            # proxy node if Phase 2 is later triggered. Copilot-added nodes live only in
            # project["nodes"] and are cleared on a full Phase 2 re-run, which is correct
            # behaviour for a full concept redesign.
            state["_actions_taken"] += 1
            status = "✓ approved" if new_node["mason_approved"] else "⚠ validation errors"
            return f"Node '{category}' ({engine_hint}) added — {status}. ACTION COMPLETE — call respond() now."

        def h_update_node_params(node_id: str, parameters: Dict) -> str:
            if state["_actions_taken"] > 0:
                return "ACTION ALREADY DONE. Call respond() now."
            nodes = state["project"].get("nodes", [])
            node = next((n for n in nodes if n.get("id") == node_id), None)
            if not node:
                return f"Node '{node_id}' not found in current project."
            node.setdefault("parameters", {}).update(parameters)
            state["_actions_taken"] += 1
            return f"Updated parameters for {node_id}: {list(parameters.keys())}. ACTION COMPLETE — call respond() now."

        def h_delete_node(node_id: str) -> str:
            if state["_actions_taken"] > 0:
                return "ACTION ALREADY DONE. Call respond() now."
            nodes = state["project"].get("nodes", [])
            before = len(nodes)
            state["project"]["nodes"] = [n for n in nodes if n.get("id") != node_id]
            after = len(state["project"]["nodes"])
            if after == before:
                return f"Node '{node_id}' not found."
            # Clean up connections
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
            gp = node.get("grid_position", [0, 0, 0])
            if isinstance(gp, (list, tuple)) and len(gp) >= 3:
                gp = list(gp)
                gp[2] = z_layer
                node["grid_position"] = gp
            state["_actions_taken"] += 1
            return f"Moved {node_id} to z-layer {z_layer}. ACTION COMPLETE — call respond() now."

        def h_update_palette(
            primary_colors: Optional[List[str]] = None,
            accent_colors: Optional[List[str]] = None,
            shapes: Optional[List[str]] = None,
            motion_words: Optional[List[str]] = None,
        ) -> str:
            updated = []
            # Update session brief
            vp = state["session_json"].setdefault("brief", {}).setdefault("visual_palette", {})
            if primary_colors is not None:
                vp["primary_colors"] = primary_colors
                updated.append("primary_colors")
            if accent_colors is not None:
                vp["accent_colors"] = accent_colors
                updated.append("accent_colors")
            if shapes is not None:
                vp["shapes"] = shapes
                updated.append("shapes")
            if motion_words is not None:
                vp["motion_words"] = motion_words
                updated.append("motion_words")
            # Mirror to project design_brief
            proj_brief_vp = state["project"].setdefault("design_brief", {}).setdefault("visual_palette", {})
            proj_brief_vp.update(vp)
            return f"Palette updated: {updated}"

        def h_update_visual_concept(new_prompt: str) -> str:
            state["session_json"].setdefault("input", {})["prompt_text"] = new_prompt
            # Also update design_brief in project
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
            "update_node_params": h_update_node_params,
            "delete_node": h_delete_node,
            "move_node": h_move_node,
            "update_palette": h_update_palette,
            "update_visual_concept": h_update_visual_concept,
            "regenerate_node": h_regenerate_node,
            "respond": h_respond,
        }

    # ------------------------------------------------------------------
    # Targeted Mason regen for a single node
    # ------------------------------------------------------------------

    def _regen_single_node(self, node_id: str, intent: str, state: Dict) -> str:
        """Run Mason on a single existing node with a new intent."""
        try:
            from phase2.agents.mason import MasonAgent
            from phase2.data_types import NodeTensor, BuildSheet, NodeMeta

            nodes = state["project"].get("nodes", [])
            node_data = next((n for n in nodes if n.get("id") == node_id), None)
            if not node_data:
                return f"Node '{node_id}' not found in current project."

            # Build minimal NodeTensor from project node dict
            meta_raw = node_data.get("meta", {})
            node_tensor = NodeTensor(
                id=node_data["id"],
                meta=meta_raw,  # NodeTensor accepts dict for meta
                grid_position=tuple(node_data.get("grid_position", [0, 0, 0])),
                grid_size=tuple(node_data.get("grid_size", [1, 1])),
                engine=node_data.get("engine", "glsl"),
                code_snippet=node_data.get("code_snippet", ""),
                parameters=node_data.get("parameters", {}),
                input_nodes=node_data.get("input_nodes", []),
                keywords=node_data.get("keywords", []),
                semantic_purpose=node_data.get("semantic_purpose", intent),
                architect_approved=True,
                mason_approved=False,
                validation_errors=[],
            )

            # Build minimal BuildSheet from node metadata + user intent
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

            if updated and updated[0].code_snippet:
                # Apply result back to project
                for n in state["project"]["nodes"]:
                    if n.get("id") == node_id:
                        n["code_snippet"] = updated[0].code_snippet
                        n["mason_approved"] = updated[0].mason_approved
                        n["validation_errors"] = updated[0].validation_errors
                        break
                if updated[0].mason_approved:
                    return f"Node {node_id} regenerated successfully."
                else:
                    errs = updated[0].validation_errors
                    return f"Regeneration completed with issues: {errs}"
            return "Regeneration produced no code."

        except Exception as e:
            print(f"[DesignCopilot] regen_single_node error: {e}")
            return f"Regeneration failed: {e}"

    def _gen_new_node(
        self,
        category: str,
        role: str,
        description: str,
        engine_hint: str,
        state: Dict,
    ):
        """Run Mason on a brand-new node. Returns (node_dict, None) or (None, error_str)."""
        try:
            from phase2.agents.mason import MasonAgent
            from phase2.data_types import NodeTensor, BuildSheet

            nodes = state["project"].get("nodes", [])
            z_layer = len(nodes)
            last_node = nodes[-1] if nodes and role != "source" else None
            input_nodes = [last_node["id"]] if last_node else []

            # Unique id that doesn't clash with existing project nodes
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

            # BuildSheet.inputs expects List[Dict] with protocol/meaning/etc keys
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
                "mason_approved": g.mason_approved,
                "validation_errors": g.validation_errors,
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
