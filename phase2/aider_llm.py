"""LLM Wrapper — Shared orchestrator for all Phase 2 (and Phase 1) agents.

Replaces all raw requests.post(OLLAMA_URL) calls with litellm,
providing proper chat message format, retries, and supervisory review.

All LLM API usage is isolated here — if the backend changes,
only this file needs updating.
"""

import json
import re
import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable

import litellm

# Suppress litellm's verbose logging
litellm.suppress_debug_info = True
litellm.set_verbose = False


@dataclass
class ToolCallResult:
    """Result from a single tool invocation within a tool-calling loop."""
    tool_name: str
    arguments: Dict[str, Any]
    result: str


@dataclass
class ToolLoopResult:
    """Aggregate result from call_with_tools()."""
    final_text: str = ""
    tool_results: List[ToolCallResult] = field(default_factory=list)
    turns_used: int = 0
    ok: bool = False  # True if LLM returned text naturally (loop completed)


class AiderLLM:
    """Wraps litellm for LLM calls across the pipeline."""

    def __init__(self):
        # Ensure Ollama base URL is set
        self.api_base = os.environ.get(
            "OLLAMA_API_BASE", "http://127.0.0.1:11434"
        )
        # Cloud mode — activated via USE_CLOUD_LLM=1 env var (rollback: set to 0)
        from config import USE_CLOUD_LLM, CLOUD_API_KEY, CLOUD_API_BASE
        self.cloud_mode = USE_CLOUD_LLM
        self.cloud_api_base = CLOUD_API_BASE
        self.cloud_api_key = CLOUD_API_KEY
        if self.cloud_mode:
            print("[AiderLLM] Cloud mode ACTIVE — routing Mason to Ollama cloud")

    def _get_litellm_params(self, model_name: str) -> dict:
        """Return litellm kwargs for the active backend (local Ollama / cloud).

        Cloud routing is only applied when BOTH cloud_mode is active AND the
        model name ends with '-cloud' (e.g. qwen3-coder:480b-cloud).
        Local fallback models (llama3.2, qwen2.5-coder) always route to local Ollama,
        even in cloud mode, to avoid 401 errors from mismatched model names.
        """
        if self.cloud_mode and (model_name.endswith("-cloud") or model_name.endswith(":cloud")):
            return {
                "model": f"openai/{model_name}",
                "api_base": self.cloud_api_base,
                "api_key": self.cloud_api_key,
            }
        return {
            "model": f"ollama_chat/{model_name}",
            "api_base": self.api_base,
        }

    def call(
        self,
        prompt: str,
        model_name: str,
        fnames: Optional[List[str]] = None,
        think_tokens: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Call LLM via litellm.

        Args:
            prompt: The instruction/prompt text
            model_name: Ollama model name (e.g. "qwen2.5-coder:7b")
            fnames: Unused (kept for API compatibility)
            think_tokens: Unused (kept for API compatibility)
            system_prompt: Optional system message for the LLM

        Returns:
            Response text from the LLM
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        max_attempts = 2
        for attempt in range(max_attempts):
            try:
                response = litellm.completion(
                    **self._get_litellm_params(model_name),
                    messages=messages,
                    stream=False,
                    timeout=300,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                err_str = str(e).lower()
                is_timeout = "timeout" in err_str or "timed out" in err_str

                if is_timeout and attempt < max_attempts - 1:
                    print(f"  [LLM] Timeout on {model_name}, retrying ({attempt + 1}/{max_attempts})...")
                    continue

                print(f"  [LLM] litellm call failed ({model_name}): {e}")
                return ""
        return ""

    @staticmethod
    def _parse_text_tool_call(text: str):
        """Parse a tool call from plain-text JSON.

        Handles:
          {"name": "update_node_code", "arguments": {...}}   (qwen/openai format)
          {"name": "create_node", "parameters": {...}}        (llama3.2 format)
          {"name": "foo", "params": {...}}                    (fallback)
          ```json\n{...}\n```                                 (markdown-fenced)

        Returns (fn_name, args_dict) or None.
        """
        # Strip markdown code fences if present
        stripped = text.strip()
        fence_match = re.search(r'```(?:json)?\s*\n?([\s\S]*?)```', stripped)
        if fence_match:
            stripped = fence_match.group(1).strip()
        # Also extract first {...} block if response has prose around it
        if not stripped.startswith("{"):
            obj_match = re.search(r'(\{[\s\S]*\})', stripped)
            if obj_match:
                stripped = obj_match.group(1)
        try:
            data = json.loads(stripped)
            if isinstance(data, dict) and "name" in data:
                fn_name = data["name"]
                args = data.get("arguments", data.get("parameters", data.get("params", {})))
                if isinstance(args, str):
                    args = json.loads(args)
                if isinstance(args, dict):
                    return fn_name, args
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
        return None

    # Code markers that indicate the LLM dumped code as plain text
    # instead of calling update_node_code
    _CODE_MARKERS = (
        "void main", "fragColor", "gl_FragColor", "gl_Position",
        "uniform ", "varying ", "precision ", "#version",
        "function draw(", "function setup(",
        "export default function", "import p5", "s.setup",
        "s.createCanvas", "new p5(",
    )

    @staticmethod
    def _extract_code_from_text(text: str) -> str:
        """Extract code from a mixed prose+code LLM response.

        Tries (in order):
        1. Markdown code fences (```glsl ... ``` or ``` ... ```)
        2. void main() { ... } block via brace matching
        3. Falls back to full text (let downstream clean it)
        """
        # 1. Markdown fences
        fence_match = re.search(r'```(?:glsl|hlsl|c|cpp|javascript|js)?\s*\n([\s\S]*?)```', text)
        if fence_match:
            return fence_match.group(1).strip()

        # 2. void main() { ... } block with brace matching
        main_match = re.search(r'(void\s+main\s*\(\s*\)\s*\{)', text)
        if main_match:
            start = main_match.start()
            brace_start = text.index('{', main_match.start())
            depth = 0
            for i in range(brace_start, len(text)):
                if text[i] == '{':
                    depth += 1
                elif text[i] == '}':
                    depth -= 1
                    if depth == 0:
                        return text[start:i + 1].strip()

        # 3. Fallback: return full text
        return text

    def call_with_tools(
        self,
        system_prompt: str,
        user_prompt: str,
        model_name: str,
        tools: List[Dict[str, Any]],
        tool_handlers: Dict[str, Callable],
        max_turns: int = 6,
        code_node_id: str = "",
    ) -> ToolLoopResult:
        """Multi-turn tool-calling loop.

        Sends tools schema to the LLM, executes tool calls, feeds results
        back, and repeats until the LLM returns a text response (no tool
        calls) or max_turns is reached.

        If the LLM returns plain text containing code instead of calling
        update_node_code, we auto-intercept and run the handler for it,
        then prompt the LLM to compile-check.

        Args:
            system_prompt: System message with context and workflow
            user_prompt: Initial user instruction
            model_name: Ollama model name (must support tool calling)
            tools: OpenAI-format tool definitions
            tool_handlers: Map of tool_name -> handler callable
            max_turns: Maximum LLM round-trips before stopping
            code_node_id: Node ID for implicit code capture (e.g. "n0")

        Returns:
            ToolLoopResult with final_text, tool_results, turns_used, ok
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        result = ToolLoopResult()

        for turn in range(max_turns):
            result.turns_used = turn + 1
            try:
                response = litellm.completion(
                    **self._get_litellm_params(model_name),
                    messages=messages,
                    tools=tools,
                    stream=False,
                    timeout=300,
                )
            except Exception as e:
                err_str = str(e).lower()
                if "tool" in err_str or "function" in err_str or "not supported" in err_str:
                    print(f"  [LLM] Model {model_name} does not support tool calling, aborting loop")
                else:
                    print(f"  [LLM] Tool-call failed on turn {turn + 1} ({model_name}): {e}")
                break

            choice = response.choices[0]
            msg = choice.message

            # Case 1: LLM returned tool calls — execute and feed results back
            if msg.tool_calls:
                messages.append(msg.model_dump())

                for tc in msg.tool_calls:
                    fn_name = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments
                    except (json.JSONDecodeError, TypeError):
                        args = {"raw": tc.function.arguments}

                    handler = tool_handlers.get(fn_name)
                    if handler:
                        try:
                            tool_output = handler(**args)
                        except Exception as handler_err:
                            tool_output = f"Error: {handler_err}"
                    else:
                        tool_output = f"Error: Unknown tool '{fn_name}'"

                    tcr = ToolCallResult(tool_name=fn_name, arguments=args, result=str(tool_output))
                    result.tool_results.append(tcr)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": str(tool_output),
                    })
                continue  # Next turn

            # Case 2: LLM returned text — may be a JSON tool call or raw code
            text = msg.content or ""
            handled = False

            # 2a: Ollama/llama returns tool calls as JSON text (possibly fenced)
            #     e.g. {"name": "update_node_code", "arguments": {...}}
            #     or   {"name": "create_node", "parameters": {...}}
            if "{" in text and '"name"' in text:
                parsed_call = self._parse_text_tool_call(text)
                if parsed_call:
                    fn_name, args = parsed_call
                    handler = tool_handlers.get(fn_name)
                    if handler:
                        print(f"  [LLM] Parsed text tool call: {fn_name}() (turn {turn + 1})")
                        try:
                            tool_output = handler(**args)
                        except Exception as handler_err:
                            tool_output = f"Error: {handler_err}"

                        tcr = ToolCallResult(tool_name=fn_name, arguments=args, result=str(tool_output))
                        result.tool_results.append(tcr)

                        # Feed back and prompt next step
                        messages.append({"role": "assistant", "content": text})
                        if fn_name == "update_node_code" and code_node_id:
                            messages.append({
                                "role": "user",
                                "content": f"Code updated. Now call compile_and_get_errors(\"{code_node_id}\") to verify.",
                            })
                        elif fn_name == "ponder" and code_node_id:
                            messages.append({
                                "role": "user",
                                "content": "Good. Now proceed with your next action.",
                            })
                        else:
                            messages.append({
                                "role": "user",
                                "content": f"Tool returned: {str(tool_output)[:300]}",
                            })
                        handled = True

            # 2b: Raw code dumped as text (no JSON wrapper)
            if not handled and text and code_node_id and "update_node_code" in tool_handlers:
                has_code = any(marker in text for marker in self._CODE_MARKERS)
                if has_code:
                    extracted = self._extract_code_from_text(text)
                    print(f"  [LLM] Implicit code capture for {code_node_id} (turn {turn + 1})")
                    handler = tool_handlers["update_node_code"]
                    try:
                        tool_output = handler(node_id=code_node_id, new_code=extracted)
                    except Exception as handler_err:
                        tool_output = f"Error: {handler_err}"

                    tcr = ToolCallResult(
                        tool_name="update_node_code",
                        arguments={"node_id": code_node_id, "new_code": "(implicit)"},
                        result=str(tool_output),
                    )
                    result.tool_results.append(tcr)

                    messages.append({"role": "assistant", "content": text})
                    messages.append({
                        "role": "user",
                        "content": (
                            f"I extracted your code and updated it. Now check for errors by responding with:\n"
                            f"{{\"name\": \"compile_and_get_errors\", \"arguments\": {{\"node_id\": \"{code_node_id}\"}}}}"
                        ),
                    })
                    handled = True

            if handled:
                continue  # Next turn

            # Case 3: Text with no code — loop is done
            result.final_text = text
            result.ok = True
            break

        return result

    def review(self, context: str, model_name: str) -> Dict[str, Any]:
        """Review agent output and return assessment.

        Args:
            context: Description of what was produced + the output itself
            model_name: Model to use for review (e.g. llama3.2:latest)

        Returns:
            {"ok": bool, "issues": list[str], "suggestions": list[str]}
        """
        prompt = f"""You are a QA reviewer for a visual node-graph pipeline.
Review the following output and assess quality.

{context}

Return JSON ONLY:
{{"ok": true/false, "issues": ["list of problems"], "suggestions": ["list of fixes"]}}"""

        raw = self.call(prompt, model_name)

        # Parse JSON from response
        try:
            match = re.search(r'\{[\s\S]*\}', raw)
            if match:
                data = json.loads(match.group())
                return {
                    "ok": bool(data.get("ok", True)),
                    "issues": data.get("issues", []),
                    "suggestions": data.get("suggestions", []),
                }
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback — if we can't parse, assume OK (don't block pipeline)
        return {"ok": True, "issues": [], "suggestions": []}


# Singleton
_instance: Optional[AiderLLM] = None


def get_aider_llm() -> AiderLLM:
    """Get or create singleton AiderLLM instance."""
    global _instance
    if _instance is None:
        _instance = AiderLLM()
    return _instance
