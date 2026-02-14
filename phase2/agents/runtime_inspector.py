"""
Runtime Inspector Agent - WebGL Field Medic
-------------------------------------------
Receives ACTUAL runtime errors from the browser (via Flask).
Uses LLM to fix specific driver/GPU issues that Mason could not predict.

Aligned with Mason's prompt engineering and cleaning logic.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import re
import json
import hashlib
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime

from config import OLLAMA_URL, MODEL_NAME_CODING


class RuntimeInspector:
    """LLM-based fixer for runtime errors reported from the browser.

    Key difference from Mason:
    - Works with ACTUAL runtime errors from WebGL/JS execution in the browser
    - These are errors Mason (running on CPU) can never see (e.g., precision issues,
      uninitialized varyings, GPU-specific loop limits)
    - Uses the same prompt engineering and cleaning as Mason for consistency
    
    Passthrough Retry Strategy:
    - When a passthrough node fails at runtime, this inspector will:
      1. Clear the passthrough flag and retry LLM-based generation with error context
      2. If LLM retries exhausted, trigger fallback node generation (semantic replacement)
      3. This prevents both blank screens and infinite loops
    """

    MAX_RETRIES_PER_ERROR = 2
    MAX_PASSTHROUGH_RETRIES = 1  # Extra retry attempt for passthrough nodes

    def __init__(self):
        self.ollama_url = OLLAMA_URL
        self.model = MODEL_NAME_CODING
        self.error_attempts = {}   # {hash: count}
        self.passthrough_retries = {}  # {node_id: count} - track passthrough-specific retries
        self.fixed_errors = set()  # {hash}
        self.fix_history = []
        self.fallback_node_callback = None  # Will be set by external orchestrator

    def set_fallback_callback(self, callback):
        """Set callback for fallback node generation when all retries exhausted."""
        self.fallback_node_callback = callback

    def _compute_error_hash(self, code: str, error_info: Dict[str, Any]) -> str:
        """Create a fingerprint for this specific crash."""
        fingerprint = f"{code[:50]}{error_info.get('message', '')}"
        return hashlib.md5(fingerprint.encode()).hexdigest()

    def fix_runtime_error(self, code: str, error_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Main entry point. Takes broken code + Browser Error -> Returns Fixed Code or Fallback Action.

        Args:
            code: The GLSL/JS code that failed in the browser
            error_info: Error context from browser:
                - message: str (the actual error message)
                - node_id: str
                - engine: str
                - parameters: dict
                - input_nodes: list
                - is_passthrough: bool (optional, set by Mason if node was passthrough)
                - category: str (optional, for fallback node generation)
                - keywords: list (optional, for semantic matching in fallback)

        Returns:
            Dict with:
            - "action": "fix" | "passthrough_retry" | "fallback" | "skip"
            - "fixed_code": str (if action == "fix")
            - "error_msg": str (reason for skipping/fallback)
            - If action == "passthrough_retry": caller should signal Mason to regenerate
            - If action == "fallback": caller should generate a semantic replacement node
        """
        node_id = error_info.get('node_id', 'unknown')
        error_msg = error_info.get('message', error_info.get('error_message', 'Unknown'))
        engine = error_info.get('engine', 'glsl')
        is_passthrough = error_info.get('is_passthrough', False)
        category = error_info.get('category', 'unknown')
        keywords = error_info.get('keywords', [])

        # 1. PASSTHROUGH NODE HANDLING
        # ==============================
        if is_passthrough:
            passthrough_count = self.passthrough_retries.get(node_id, 0)
            
            if passthrough_count >= self.MAX_PASSTHROUGH_RETRIES:
                print(f"[RuntimeInspector] PASSTHROUGH EXHAUSTED for {node_id} after {passthrough_count} attempts")
                print(f"[RuntimeInspector] Triggering FALLBACK NODE GENERATION for '{category}' ({', '.join(keywords)})")
                
                # Return fallback action - orchestrator will generate semantic replacement
                self.passthrough_retries[node_id] = passthrough_count + 1
                return {
                    "action": "fallback",
                    "node_id": node_id,
                    "category": category,
                    "keywords": keywords,
                    "engine": engine,
                    "error_msg": f"Passthrough node failed at runtime after {passthrough_count} attempts: {error_msg[:80]}"
                }
            
            # Attempt to fix passthrough by retrying with error context
            print(f"[RuntimeInspector] PASSTHROUGH NODE {node_id} failed at runtime - retrying with error context")
            print(f"  Error: {error_msg[:100]}")
            self.passthrough_retries[node_id] = passthrough_count + 1
            
            # Return passthrough_retry action - caller should signal Mason to regenerate with error context
            return {
                "action": "passthrough_retry",
                "node_id": node_id,
                "error_msg": error_msg,
                "engine": engine,
                "parameters": error_info.get('parameters', {})
            }

        # 2. NORMAL ERROR DEDUPLICATION (for non-passthrough nodes)
        # ===========================================================
        error_hash = self._compute_error_hash(code, error_info)
        attempts = self.error_attempts.get(error_hash, 0)

        if error_hash in self.fixed_errors:
            print(f"[RuntimeInspector] Already fixed error {error_hash[:8]}, skipping")
            return {
                "action": "skip",
                "error_msg": "Already fixed this error previously"
            }

        if attempts >= self.MAX_RETRIES_PER_ERROR:
            print(f"[RuntimeInspector] Gave up on error {error_hash[:8]} after {attempts} attempts")
            return {
                "action": "skip",
                "error_msg": f"Max retries ({self.MAX_RETRIES_PER_ERROR}) exhausted for this error"
            }

        self.error_attempts[error_hash] = attempts + 1
        print(f"[RuntimeInspector] Fixing runtime error for {node_id} (attempt {attempts + 1}): {error_msg[:100]}")

        # 2. Build the "Field Medic" Prompt
        parameters = error_info.get('parameters', {})
        param_list = ", ".join([f"u_{k}" for k in parameters.keys()]) if parameters else "none"
        input_nodes = error_info.get('input_nodes', [])
        input_samplers = ", ".join([f"u_{nid}" for nid in input_nodes]) if input_nodes else "none"

        if engine in ("glsl", "regl"):
            prompt = f"""You are a WebGL Runtime Debugger.
The following GLSL ES 3.00 code compiled on the server but FAILED in the browser.

NODE: {node_id}
PARAMETER UNIFORMS: {param_list}
INPUT SAMPLERS: {input_samplers}
STANDARD UNIFORMS: u_time (float), u_resolution (vec2), u_input0 (sampler2D)

CODE:
```glsl
{code}
```

BROWSER ERROR:
{error_msg}

TASK:
1. Analyze the error (look for precision issues, uninitialized varyings, loop limits, or driver bugs).
2. Fix the code strictly. Do not change the visual look if possible.
3. Use fragColor (NOT gl_FragColor). Use texture() (NOT texture2D).
4. Parameters must use u_ prefix.
5. Must have void main() function.

Return ONLY the fixed GLSL code. No explanation, no markdown fences."""
        elif engine in ("three_js", "canvas2d", "events", "webaudio", "p5"):
            prompt = f"""You are a JavaScript Runtime Debugger.
The following {engine} code failed in the browser.

NODE: {node_id}
PARAMETERS: {json.dumps(parameters, indent=2)}

CODE:
```javascript
{code}
```

BROWSER ERROR:
{error_msg}

Fix the code. Return ONLY the fixed code. No explanation.
CRITICAL RULES:
1. "array declared but never populated outside draw()" → Populate arrays immediately after declaration:
   let arr = []; for (let i = 0; i < count; i++) arr.push({...});
2. "nothing will render" → Verify your draw() function calls ctx.arc(), ctx.fill(), ctx.stroke(), etc.
3. Use ctx for all drawing operations: ctx.beginPath(), ctx.arc(), ctx.fillStyle, ctx.fill()
4. Test: Does your code have at least one ctx.arc() or ctx.fillRect() call?
"""
        else:
            return {
                "action": "skip",
                "error_msg": f"Unsupported engine: {engine}"
            }

        # 3. Call LLM to attempt fix
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,
                        "num_predict": 4096,
                    }
                },
                timeout=60
            )

            if response.status_code != 200:
                print(f"[RuntimeInspector] LLM error: HTTP {response.status_code}")
                return {
                    "action": "skip",
                    "error_msg": f"LLM HTTP error: {response.status_code}"
                }

            raw_text = response.json().get("response", "")
            fixed_code = self._clean_code(raw_text, engine)

            # If LLM returns empty or garbage, abort
            if len(fixed_code) < 30:
                print(f"[RuntimeInspector] LLM returned too short ({len(fixed_code)} chars)")
                return {
                    "action": "skip",
                    "error_msg": f"LLM returned insufficient code ({len(fixed_code)} chars)"
                }

            self.fixed_errors.add(error_hash)
            self._record_fix(node_id, error_msg, error_hash)
            print(f"[RuntimeInspector] Fixed code for {node_id} ({len(fixed_code)} chars)")
            return {
                "action": "fix",
                "fixed_code": fixed_code,
                "fixed_parameters": error_info.get("parameters", {})
            }

        except Exception as e:
            print(f"[RuntimeInspector] LLM Error: {e}")
            return {
                "action": "skip",
                "error_msg": f"LLM exception: {str(e)[:80]}"
            }

    def _clean_code(self, text: str, engine: str = "glsl") -> str:
        """Clean LLM output using Mason-aligned cleaning logic."""
        if not text:
            return ""

        # Remove thinking tags
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

        # Remove markdown fences
        text = re.sub(r'```(?:glsl|javascript|js|c|cpp)?\s*\n?', '', text)
        text = re.sub(r'```\s*$', '', text, flags=re.MULTILINE)

        # Remove common hallucination prefixes
        text = re.sub(r'^(?:Here\'s|Here is|Fixed|The fixed)[^:\n]*:\s*', '', text, flags=re.IGNORECASE)

        # Remove special LLM tokens
        text = re.sub(r'<\|[^>]*\|>', '', text)
        text = re.sub(r'<[^>]*begin[^>]*>', '', text, flags=re.IGNORECASE)

        text = text.strip()

        # GLSL-specific cleaning (aligned with Mason)
        if engine in ("glsl", "regl"):
            # Strip forbidden header lines (runtime injects these)
            text = re.sub(r'#version\s+\d+\s*(es)?', '', text)
            text = re.sub(r'precision\s+(highp|mediump|lowp)\s+float\s*;', '', text)
            text = re.sub(r'\bout\s+vec4\s+fragColor\s*;', '', text)
            text = re.sub(r'\bin\s+vec2\s+v_uv\s*;', '', text)
            text = re.sub(r'\buniform\s+float\s+u_time\s*;', '', text)
            text = re.sub(r'\buniform\s+vec2\s+u_resolution\s*;', '', text)
            text = re.sub(r'\buniform\s+sampler2D\s+u_input0\s*;', '', text)

            # Fix WebGL1 patterns
            text = text.replace('gl_FragColor', 'fragColor')
            text = re.sub(r'texture2D\s*\(', 'texture(', text)

            # Fix escape sequences
            text = text.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '')

        return text.strip()

    def _record_fix(self, node_id: str, error_msg: str, error_hash: str):
        """Record fix for debugging/analytics."""
        self.fix_history.append({
            "node_id": node_id,
            "error": error_msg[:100],
            "hash": error_hash[:8],
            "timestamp": datetime.now().isoformat()
        })

    def analyze_and_fix(self, error_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Legacy entry point for backward compatibility with app_web.py.
        Now handles passthrough retry and fallback actions in addition to fixes.
        
        Returns dict with action indicating what caller should do:
        - "fix": Apply fixed_code
        - "passthrough_retry": Signal Mason to regenerate with error context
        - "fallback": Generate semantic replacement node
        - "skip": Cannot fix, continue with existing code
        """
        code = error_info.get("code_snippet", "")
        result = self.fix_runtime_error(code, error_info)

        if not result:
            return {
                "success": False,
                "action": "skip",
                "analysis": "No result from runtime inspector"
            }

        action = result.get("action")

        if action == "fix":
            return {
                "success": True,
                "action": "fix",
                "fixed_code": result.get("fixed_code"),
                "fixed_parameters": result.get("fixed_parameters", error_info.get("parameters", {})),
                "analysis": "Runtime fix applied",
                "fix_type": "code"
            }
        elif action == "passthrough_retry":
            return {
                "success": False,
                "action": "passthrough_retry",
                "node_id": result.get("node_id"),
                "error_msg": result.get("error_msg"),
                "analysis": "Passthrough node failed; retrying with error context",
                "fix_type": "passthrough_retry"
            }
        elif action == "fallback":
            return {
                "success": False,
                "action": "fallback",
                "node_id": result.get("node_id"),
                "category": result.get("category"),
                "keywords": result.get("keywords"),
                "engine": result.get("engine"),
                "analysis": result.get("error_msg"),
                "fix_type": "fallback_generation"
            }
        else:  # action == "skip" or unknown
            return {
                "success": False,
                "action": "skip",
                "analysis": result.get("error_msg", "Could not fix runtime error"),
                "fix_type": "none"
            }

    def batch_fix(self, errors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fix multiple errors, categorizing results by action type."""
        fixes = []
        retries = []
        fallbacks = []
        
        for error in errors:
            result = self.analyze_and_fix(error)
            
            if result.get("action") == "fix" and result.get("success"):
                fixes.append({
                    "node_id": error.get("node_id"),
                    "action": "fix",
                    "fixed_code": result["fixed_code"],
                    "fixed_parameters": result["fixed_parameters"],
                    "analysis": result["analysis"]
                })
            elif result.get("action") == "passthrough_retry":
                retries.append({
                    "node_id": result.get("node_id"),
                    "action": "passthrough_retry",
                    "error_msg": result.get("error_msg"),
                    "analysis": result["analysis"]
                })
            elif result.get("action") == "fallback":
                fallbacks.append({
                    "node_id": result.get("node_id"),
                    "action": "fallback",
                    "category": result.get("category"),
                    "keywords": result.get("keywords"),
                    "engine": result.get("engine"),
                    "analysis": result["analysis"]
                })
        
        return {
            "fixes": fixes,
            "retries": retries,
            "fallbacks": fallbacks
        }

    def get_fix_history(self) -> List[Dict]:
        """Get history of all fixes applied."""
        return self.fix_history

    def reset_error_tracking(self, node_id: str = None):
        """Reset error tracking for a specific node or all nodes."""
        if node_id:
            keys_to_remove = [k for k in self.error_attempts.keys()]
            for k in keys_to_remove:
                del self.error_attempts[k]
            if node_id in self.passthrough_retries:
                del self.passthrough_retries[node_id]
            self.fixed_errors.clear()
            print(f"[RuntimeInspector] Reset error tracking for {node_id}")
        else:
            self.error_attempts.clear()
            self.passthrough_retries.clear()
            self.fixed_errors.clear()
            print(f"[RuntimeInspector] Reset all error tracking")

    def get_error_stats(self) -> Dict[str, Any]:
        """Get statistics about error handling."""
        return {
            "total_attempts": sum(self.error_attempts.values()),
            "unique_errors": len(self.error_attempts),
            "fixed_errors": len(self.fixed_errors),
            "fix_history_count": len(self.fix_history)
        }


# Singleton instance for Flask integration
_runtime_inspector = None


def get_runtime_inspector() -> RuntimeInspector:
    """Get or create singleton RuntimeInspector instance."""
    global _runtime_inspector
    if _runtime_inspector is None:
        _runtime_inspector = RuntimeInspector()
    return _runtime_inspector
