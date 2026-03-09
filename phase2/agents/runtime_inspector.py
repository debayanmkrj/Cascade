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
from typing import Dict, List, Any, Optional
from datetime import datetime

from config import EFFECTIVE_MODEL_CODING
from phase2.aider_llm import get_aider_llm


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

    MAX_RETRIES_PER_ERROR = 5
    MAX_PASSTHROUGH_RETRIES = 1  # Extra retry attempt for passthrough nodes

    def __init__(self):
        self.llm = get_aider_llm()
        self.model = EFFECTIVE_MODEL_CODING
        print(f"[RuntimeInspector] Using model: {self.model}")
        self.error_attempts = {}   # {hash: count}
        self.passthrough_retries = {}  # {node_id: count} - track passthrough-specific retries
        self.fixed_errors = set()  # {hash}
        self.fix_history = []
        self.fallback_node_callback = None  # Will be set by external orchestrator

    def set_fallback_callback(self, callback):
        """Set callback for fallback node generation when all retries exhausted."""
        self.fallback_node_callback = callback

    def _compute_error_hash(self, code: str, error_info: Dict[str, Any]) -> str:
        """Create a fingerprint scoped to the node + full code content."""
        node_id = error_info.get('node_id', '')
        err = error_info.get('message', '') or error_info.get('error_message', '')
        # Use full code hash — preamble lines (import p5...) are identical across nodes
        fingerprint = f"{node_id}{hashlib.md5(code.encode()).hexdigest()}{err[:60]}"
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
        error_msg = error_info.get('message') or error_info.get('error_message') or 'Unknown error'
        engine = error_info.get('engine', 'glsl')

        # WebGL2 drivers sometimes return null info log with no useful message.
        # Substitute a diagnostic checklist so the LLM knows what to look for.
        _null_patterns = ("null", "(gpu driver returned no error details", "shader compile failed:\nnull")
        if engine in ("glsl", "regl") and any(error_msg.lower().strip().endswith(p) for p in _null_patterns):
            error_msg = (
                "WebGL2 rejected this shader with no error message (GPU driver returned null info log). "
                "Audit the code for these common WebGL2/GLSL ES 3.00 issues:\n"
                "- Scalar swizzle: accessing .r/.g/.b/.x/.y on a float variable (only valid on vec types)\n"
                "- Function redeclaration: redefining hash(), noise(), fbm(), rgb2hsv(), etc. "
                  "(these are already declared in the shader prefix — do NOT redeclare them)\n"
                "- Array initializer syntax: float arr[] = {0.0, 1.0} is C syntax not GLSL; "
                  "use float arr[2]; arr[0]=0.0; arr[1]=1.0;\n"
                "- gl_FragColor instead of fragColor (WebGL2 requires fragColor out variable)\n"
                "- texture2D() instead of texture() (WebGL2 uses texture())\n"
                "- Struct or array initialization with braces {} — use constructors instead\n"
                "Fix any of the above issues found in the code."
            )
            print(f"[RuntimeInspector] Null error log detected for {node_id} — using diagnostic prompt")
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
            # Mark as "fixed" so subsequent re-reports hit the fast skip path
            self.fixed_errors.add(error_hash)
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

AVAILABLE GLSL FUNCTIONS (already defined in shader prefix — do NOT redefine):
hash(float), hash(vec2), hash(vec3), hash2(vec2), hash3(vec2), hash3(vec3),
noise(vec2), snoise(vec2), snoise(vec3), fbm(vec2),
worley(vec2), simplex(vec2), simplex(vec3), perlin(vec2), perlin(vec3), voronoi(vec2)
Do NOT use any noise functions outside this list. Do NOT redefine these.

TASK:
1. Analyze the error (look for precision issues, uninitialized varyings, loop limits, or driver bugs).
2. Fix the code strictly. Do not change the visual look if possible.
3. Use fragColor (NOT gl_FragColor). Use texture() (NOT texture2D).
4. Parameters must use u_ prefix.
5. Must have void main() function.

Return ONLY the fixed GLSL code. No explanation, no markdown fences."""
        elif engine in ("three_js", "canvas2d", "events", "webaudio", "p5"):
            pixel_loop_rules = ""
            if "pixel loop" in error_msg.lower() or "pixel-loop" in error_msg.lower():
                pixel_loop_rules = """
CRITICAL — PIXEL LOOP REMOVAL (this is why the code was rejected):
The code contains nested for loops iterating over every pixel (for x<width; x++ inside for y<height; y++).
This creates 262,144+ draw calls per frame and freezes the browser for 15+ seconds. YOU MUST REMOVE THEM.

REPLACE pixel-filling loops with:
  s.background(r, g, b);  // fills entire canvas in 1 call

REPLACE per-pixel starfield/noise loops with:
  // Pre-allocate in init — fixed count, NEVER inside s.draw()
  const stars = Array.from({length: 300}, () => ({ x: Math.random()*width, y: Math.random()*height, r: Math.random()*2+0.5 }));
  // In s.draw():
  for (const star of stars) { s.ellipse(star.x, star.y, star.r); }  // 300 calls max

RULES:
- Total draw calls per s.draw() frame MUST be under 1000.
- NO nested for loops where both bounds reference width or height with increment 1.
- x += stride (where stride >= 4) is acceptable.
- Starfields: pre-allocate array of ≤500 random positions ONCE in init, reuse every frame.
"""

            p5_rules = """
P5.JS RULES:
- inputs[0], inputs[1] etc. are raw HTMLCanvasElement objects.
- NEVER call s.image(inputs[0], ...) — use s.drawingContext.drawImage(inputs[0], 0, 0, width, height).
- Guard all input usage: if (inputs[0]) s.drawingContext.drawImage(inputs[0], 0, 0, width, height);
- ALL p5 calls must use s. prefix (s.ellipse, s.fill, s.background, etc.)
- NEVER push() to arrays inside s.draw() — pre-allocate at fixed size and update in-place.
""" if engine == "p5" else ""

            prompt = f"""You are a JavaScript Runtime Debugger. The following {engine} code failed in the browser.

NODE: {node_id}
PARAMETERS: {json.dumps(parameters, indent=2)}

CODE:
```javascript
{code}
```

BROWSER ERROR:
{error_msg}
{pixel_loop_rules}{p5_rules}
Fix the code. Return ONLY the complete fixed code. No explanation, no markdown fences."""
        else:
            return {
                "action": "skip",
                "error_msg": f"Unsupported engine: {engine}"
            }

        # 3. Call LLM to attempt fix via AiderLLM (routes to cloud or local automatically)
        try:
            raw_text = self.llm.call(prompt, self.model)
            fixed_code = self._clean_code(raw_text, engine)

            # If LLM returns empty or garbage, abort
            if len(fixed_code) < 30:
                print(f"[RuntimeInspector] LLM returned too short ({len(fixed_code)} chars)")
                return {
                    "action": "skip",
                    "error_msg": f"LLM returned insufficient code ({len(fixed_code)} chars)"
                }

            # Do NOT add to fixed_errors here — we don't know yet if the fix will pass
            # hot-reload validation. Let error_attempts count handle the retry cap.
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
