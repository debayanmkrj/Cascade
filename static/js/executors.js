// Executors implement the "Universal Tensor Node" contract.
//
// Universal external interface:
//   - main input: a texture (RGBA render target) from prev Z layer (u_input0 in GLSL)
//   - side inputs: textures from named upstream nodes (u_<nodeId> in GLSL)
//   - data input: JSON-ish uniform dictionary (time/mouse/audio/event state) for JS engines
//   - output: a texture stored in TextureHub (render or data)
//
// The system supports multiple internal engines:
//   - glsl / regl       -> ShaderExecutor (WebGL2 fragment body)
//   - three_js          -> ThreeExecutor (Three.js scene to texture)
//   - webaudio          -> WebAudioExecutor (Audio -> 1xN RED8 data texture)
//   - events            -> EventExecutor (State -> 1x1 RGBA8 data texture + JSON)
//   - html_video        -> VideoExecutor (Video -> texture)
//   - canvas2d          -> Canvas2DExecutor (Canvas2D -> texture)
//   - js_module         -> JSModuleExecutor (ESM module with imports; draws to canvas -> texture)
//
// Validator implications:
//   - body-only JS nodes should not include imports/require/html.
//   - js_module is for "point 6": full code blocks with imports.

// (TextureHub loaded via separate script tag)

// -----------------------------------------------------------------------------
// GLSL utilities injected into every shader.
//
// Mason prompts reference snoise(), fbm(), hash(), perlin(), worley(), simplex(), voronoi().
// All are provided here so runtime matches Mason's validation set.
// -----------------------------------------------------------------------------
const GLSL_UTILS = `
// --- Constants ---
#define PI 3.14159265359

// --- Hash / noise / fbm (cheap) ---
float hash(float n) { return fract(sin(n) * 43758.5453123); }
float hash(vec2 p) { return hash(dot(p, vec2(127.1, 311.7))); }
float hash(vec3 p) { return hash(dot(p, vec3(127.1, 311.7, 74.7))); }

vec2 hash2(vec2 p) {
  p = vec2(dot(p, vec2(127.1, 311.7)),
           dot(p, vec2(269.5, 183.3)));
  return -1.0 + 2.0 * fract(sin(p) * 43758.5453123);
}

float noise(vec2 x) {
  vec2 i = floor(x);
  vec2 f = fract(x);

  vec2 u = f * f * (3.0 - 2.0 * f);

  float a = dot(hash2(i + vec2(0.0, 0.0)), f - vec2(0.0, 0.0));
  float b = dot(hash2(i + vec2(1.0, 0.0)), f - vec2(1.0, 0.0));
  float c = dot(hash2(i + vec2(0.0, 1.0)), f - vec2(0.0, 1.0));
  float d = dot(hash2(i + vec2(1.0, 1.0)), f - vec2(1.0, 1.0));

  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y) * 0.5 + 0.5; // -> [0,1]
}

// simplex-noise style alias: snoise in [-1, 1]
float snoise(vec2 p) { return noise(p) * 2.0 - 1.0; }
float snoise(vec3 p) { return snoise(p.xy + p.z); }

float fbm(vec2 x) {
  float v = 0.0;
  float a = 0.5;
  vec2 shift = vec2(100.0);
  mat2 rot = mat2(cos(0.5), sin(0.5), -sin(0.5), cos(0.5));
  for (int i = 0; i < 5; ++i) {
    v += a * noise(x);
    x = rot * x * 2.0 + shift;
    a *= 0.5;
  }
  return v;
}

// --- Hash3 overloads ---
vec3 hash3(vec3 p) {
  p = vec3(dot(p, vec3(127.1, 311.7, 74.7)),
           dot(p, vec3(269.5, 183.3, 246.1)),
           dot(p, vec3(113.5, 271.9, 124.6)));
  return -1.0 + 2.0 * fract(sin(p) * 43758.5453123);
}
vec3 hash3(vec2 p) { return hash3(vec3(p, 0.0)); }

// --- Worley / cellular noise ---
float worley(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  float minDist = 1.0;
  for (int y = -1; y <= 1; y++) {
    for (int x = -1; x <= 1; x++) {
      vec2 neighbor = vec2(float(x), float(y));
      vec2 cellPos = i + neighbor;
      vec2 noise_pt = hash2(cellPos);
      vec2 pt = neighbor + noise_pt;
      float dist = length(f - pt);
      minDist = min(minDist, dist);
    }
  }
  return minDist;
}

// --- Aliases that LLMs commonly generate ---
float simplex(vec2 p) { return fbm(p); }
float simplex(vec3 p) { return fbm(p.xy) * 0.5 + fbm(p.yz) * 0.3; }
float perlin(vec2 p) { return fbm(p); }
float perlin(vec3 p) { return fbm(p.xy); }
float voronoi(vec2 p) { return worley(p); }

// --- Color space conversions (LLMs call these constantly) ---
vec3 rgb2hsv(vec3 c) {
  vec4 K = vec4(0.0, -1.0/3.0, 2.0/3.0, -1.0);
  vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
  vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
  float d = q.x - min(q.w, q.y);
  float e = 1.0e-10;
  return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}
vec3 hsv2rgb(vec3 c) {
  vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
  vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
  return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}
vec3 hsl2rgb(vec3 c) {
  float a = c.y * min(c.z, 1.0 - c.z);
  vec3 k = mod(vec3(0.0, 8.0, 4.0) + c.x * 12.0, 12.0);
  return c.z - a * max(vec3(-1.0), min(min(k - 3.0, 9.0 - k), vec3(1.0)));
}

// --- Type-safe vector constructors (for LLM-generated code) ---
// These handle common type mismatches that LLMs make
vec3 s_vec3(float a) { return vec3(a); }
vec3 s_vec3(float a, float b, float c) { return vec3(a, b, c); }
vec3 s_vec3(vec2 xy, float z) { return vec3(xy, z); }
vec3 s_vec3(float x, vec2 yz) { return vec3(x, yz); }

vec4 s_vec4(float a) { return vec4(a); }
vec4 s_vec4(float a, float b, float c, float d) { return vec4(a, b, c, d); }
vec4 s_vec4(vec3 rgb, float a) { return vec4(rgb, a); }
vec4 s_vec4(float r, vec3 gba) { return vec4(r, gba); }
vec4 s_vec4(vec2 xy, vec2 zw) { return vec4(xy, zw); }
vec4 s_vec4(vec2 xy, float z, float w) { return vec4(xy, z, w); }
vec4 s_vec4(float x, vec2 yz, float w) { return vec4(x, yz, w); }
vec4 s_vec4(float x, float y, vec2 zw) { return vec4(x, y, zw); }
`;

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

function clamp01(x) {
  return Math.max(0, Math.min(1, x));
}

function sanitizeLLMArtifacts(code) {
  if (!code) return '';
  let out = String(code);

  // Remove common special tokens emitted by some LLMs
  out = out.replace(/<\|[^>]*\|>/g, '');
  // Some models emit fullwidth bars and other separators around those tokens
  out = out.replace(/<｜[^>]*｜>/g, '');
  // Remove stray "begin of sentence" artefacts (often with unicode bars/underscores)
  out = out.replace(/begin\s*▁of\s*▁sentence/gi, '');
  // Remove other unusual token separators seen in logs
  out = out.replace(/[▁｜]/g, '');

  // Normalize smart quotes that can break JS parsing
  out = out.replace(/[“”]/g, '"').replace(/[‘’]/g, "'");
  return out;
}

function isBareImport(spec) {
  return !!spec && !spec.startsWith('.') && !spec.startsWith('/') && !spec.startsWith('http:') && !spec.startsWith('https:') && !spec.startsWith('data:');
}

function resolveImport(spec) {
  // User-provided map (recommended for local preinstall)
  const map = globalThis.__MASON_IMPORTS__;
  if (map && typeof map === 'object' && map[spec]) return map[spec];

  // Built-in defaults (CDN fallback)
  // NOTE: esm.sh resolves npm packages as ESM.
  return `https://esm.sh/${spec}`;
}

// Stale CDN URL patterns → canonical esm.sh equivalents.
// Covers old saved projects that hardcoded cdn.jsdelivr.net paths.
const _CDN_REWRITES = [
  [/https:\/\/cdn\.jsdelivr\.net\/npm\/p5@[\d.]+[^\s'"]]*/g, 'https://esm.sh/p5@1.9.0'],
  [/https:\/\/unpkg\.com\/p5@[\d.]+[^\s'"]*/g,               'https://esm.sh/p5@1.9.0'],
];

function normaliseImportUrls(code) {
  let out = code;
  for (const [pattern, replacement] of _CDN_REWRITES) {
    out = out.replace(pattern, replacement);
  }
  return out;
}

function rewriteBareImports(code) {
  let out = normaliseImportUrls(String(code));

  // import ... from 'pkg'
  out = out.replace(/from\s+['"]([^'"]+)['"]/g, (m, spec) => {
    if (!isBareImport(spec)) return m;
    return `from '${resolveImport(spec)}'`;
  });

  // import 'pkg';
  out = out.replace(/import\s+['"]([^'"]+)['"]\s*;/g, (m, spec) => {
    if (!isBareImport(spec)) return m;
    return `import '${resolveImport(spec)}';`;
  });

  // dynamic import('pkg')
  out = out.replace(/import\s*\(\s*['"]([^'"]+)['"]\s*\)/g, (m, spec) => {
    if (!isBareImport(spec)) return m;
    return `import('${resolveImport(spec)}')`;
  });

  return out;
}

// -----------------------------------------------------------------------------
// Base
// -----------------------------------------------------------------------------

class BaseExecutor {
  constructor(spec, hub) {
    this.id = spec.id;
    this.engine = spec.engine;
    this.codeSnippet = spec.code_snippet || '';
    this.params = spec.parameters || {};
    this.zLayer = spec.z_layer ?? 0;
    this.inputNodes = spec.input_nodes || [];
    this.hub = hub;
    this.initialized = false;
    this.outputKind = spec.output_kind || null; // 'render' | 'data' | null
  }

  async init() { this.initialized = true; }
  async execute(_textures, _time, _data) {}
  destroy() {}
}

// -----------------------------------------------------------------------------
// Shader (GLSL)
// -----------------------------------------------------------------------------

class ShaderExecutor extends BaseExecutor {
  async init() {
    const gl = this.hub.gl;
    this.outputKind = this.outputKind || 'render';

    // Allocate output render target
    this.outputTexture = this.hub.allocate(this.id, this.zLayer, { kind: 'render', visible: true });

    // Build shader
    const vs = `#version 300 es
      in vec2 a_position;
      out vec2 v_uv;
      void main() {
        v_uv = a_position * 0.5 + 0.5;
        gl_Position = vec4(a_position, 0.0, 1.0);
      }`;

    const sideSamplers = this._buildSideSamplerDecls();
    const paramUniforms = this._buildParamUniformDecls();

    let snippet = sanitizeLLMArtifacts(this.codeSnippet);
    snippet = this._cleanSnippet(snippet);

    // Fallback to passthrough shader if snippet is empty or has no main function
    if (!snippet || !snippet.includes('main')) {
      console.warn(`[ShaderExecutor] Node ${this.id} has no valid GLSL code, using passthrough shader`);
      snippet = `void main() {
        vec2 uv = v_uv;
        vec4 col = texture(u_input0, uv);
        fragColor = col;
      }`;
    }

    const fs = `#version 300 es
      precision highp float;

      in vec2 v_uv;
      out vec4 fragColor;

      uniform float u_time;
      uniform vec2 u_resolution;

      // Main input (prev layer composite)
      uniform sampler2D u_input0;

      // Feedback: this node's own output from the previous frame
      // Use for GPU particle simulation (TouchDesigner-style ping-pong)
      uniform sampler2D u_feedback;

      // Side inputs (by node id)
${sideSamplers}

      // Node parameters (from JSON)
${paramUniforms}

      // Auto-declared missing uniforms (safety net for aliased/predefined templates)
${this._autoDeclareMissingUniforms(snippet, sideSamplers + paramUniforms)}

      // Common alias macros for LLM friendliness
#define vUv v_uv
#define u_texture u_input0
#define uTexture u_input0
#define iTime u_time
#define iResolution vec3(u_resolution, 1.0)
#define iChannel0 u_input0

${GLSL_UTILS}

${snippet}
    `;

    this.program = this._createProgram(gl, vs, fs);

    // Create a 1x1 transparent black texture as fallback for u_input0
    // Prevents feedback loop when prevTex is null (would bind outputTexture as both input and output)
    this._blankTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this._blankTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([0, 0, 0, 0]));
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.bindTexture(gl.TEXTURE_2D, null);

    // Setup quad
    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);

    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
      -1, -1,
       1, -1,
      -1,  1,
      -1,  1,
       1, -1,
       1,  1
    ]), gl.STATIC_DRAW);

    const posLoc = gl.getAttribLocation(this.program, 'a_position');
    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);

    gl.bindVertexArray(null);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    // Feedback texture: only allocate if the shader actually samples u_feedback.
    // Allocating for every node regardless wastes one full-res RGBA8 texture + framebuffer
    // per shader node (~3.7 MB each at 1280×720), causing VRAM pressure and GPU TDR.
    const _usesFeedback = /\bu_feedback\b/.test(snippet);
    if (_usesFeedback) {
      const w = this.hub.width, h = this.hub.height;
      this._feedbackTex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, this._feedbackTex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.bindTexture(gl.TEXTURE_2D, null);
      this._feedbackFb = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, this._feedbackFb);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this._feedbackTex, 0);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    } else {
      this._feedbackTex = null;
      this._feedbackFb = null;
    }

    this.initialized = true;
  }

  _buildParamUniformDecls() {
    let decl = '';
    for (const [k, v] of Object.entries(this.params || {})) {
      if (typeof v === 'number' || typeof v === 'boolean') {
        decl += `      uniform float u_${k};\n`;
      }
    }
    return decl;
  }

  _autoDeclareMissingUniforms(snippet, existingDecls) {
    // Find all u_* identifiers in the GLSL body that aren't already declared
    const used = new Set((snippet.match(/\bu_[a-zA-Z_][a-zA-Z0-9_]*\b/g) || []));
    const declared = new Set((existingDecls.match(/\bu_[a-zA-Z_][a-zA-Z0-9_]*\b/g) || []));
    // Always declared by the wrapper
    declared.add('u_time');
    declared.add('u_resolution');
    declared.add('u_input0');
    declared.add('u_feedback');

    let extra = '';
    for (const u of used) {
      if (!declared.has(u)) {
        // Heuristic type inference from naming
        const lower = u.toLowerCase();
        if (/resolution|size|offset|pos|coord|mouse/.test(lower)) {
          extra += `      uniform vec2 ${u};\n`;
        } else if (/color|col|rgb|tint/.test(lower)) {
          extra += `      uniform vec3 ${u};\n`;
        } else if (/tex|texture|sampler|map|input\d/.test(lower)) {
          // 'input\d' catches u_input1, u_input2, etc. — sequential input aliases
          extra += `      uniform sampler2D ${u};\n`;
        } else {
          extra += `      uniform float ${u};\n`;
        }
      }
    }
    return extra;
  }

  _buildSideSamplerDecls() {
    let decl = '';
    for (let i = 0; i < (this.inputNodes || []).length; i++) {
      const nodeId = this.inputNodes[i];
      if (!nodeId) continue;
      decl += `      uniform sampler2D u_${nodeId};\n`;
      // Provide sequential index alias: u_input1, u_input2, ... for LLM convenience
      // (u_input0 is already the primary/prev_layer_texture alias)
      if (i >= 1) {
        decl += `#define u_input${i} u_${nodeId}\n`;
      }
    }
    return decl;
  }

  _cleanSnippet(snippet) {
    // Strip markdown fences (keep contents)
    snippet = snippet.replace(/```[\s\S]*?```/g, (m) => {
      const inner = m.replace(/```[\w]*\n?/g, '').replace(/```/g, '');
      return inner;
    });

    // Remove forbidden header lines if present
    snippet = snippet.replace(/#version\s+\d+\s*(es)?/g, '');
    snippet = snippet.replace(/precision\s+(highp|mediump|lowp)\s+float\s*;/g, '');
    snippet = snippet.replace(/\bout\s+vec4\s+fragColor\s*;/g, '');
    snippet = snippet.replace(/\bin\s+vec2\s+v_uv\s*;/g, '');
    snippet = snippet.replace(/\buniform\s+float\s+u_time\s*;/g, '');
    snippet = snippet.replace(/\buniform\s+vec2\s+u_resolution\s*;/g, '');
    snippet = snippet.replace(/\buniform\s+sampler2D\s+u_input0\s*;/g, '');

    // Fix WebGL1 patterns
    snippet = snippet.replace(/gl_FragColor/g, 'fragColor');
    snippet = snippet.replace(/texture2D\s*\(/g, 'texture(');

    if (!/void\s+main\s*\(\s*\)/.test(snippet)) {
      // If the model failed to emit a shader, default to passthrough
      snippet = `void main() {\n  vec2 uv = v_uv;\n  fragColor = texture(u_input0, uv);\n}\n`;
    }

    return snippet.trim();
  }

  async execute(textures, time, data) {
    if (!this.initialized) return;
    const gl = this.hub.gl;

    const inputs = textures || {};
    const prevTex = inputs.prev_layer_texture || null;

    const fb = this.hub.nodeTextures.get(this.id)?.framebuffer;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.viewport(0, 0, this.hub.width, this.hub.height);

    gl.useProgram(this.program);
    gl.bindVertexArray(this.vao);

    gl.uniform1f(gl.getUniformLocation(this.program, 'u_time'), time);
    gl.uniform2f(gl.getUniformLocation(this.program, 'u_resolution'), this.hub.width, this.hub.height);

    // Params
    for (const [k, v] of Object.entries(this.params || {})) {
      if (typeof v === 'number') {
        const loc = gl.getUniformLocation(this.program, `u_${k}`);
        if (loc) gl.uniform1f(loc, v);
      } else if (typeof v === 'boolean') {
        const loc = gl.getUniformLocation(this.program, `u_${k}`);
        if (loc) gl.uniform1f(loc, v ? 1.0 : 0.0);
      }
    }

    // Upstream structured data -> uniforms (audio level, tracking, etc.)
    // Merge: globalData (any source node in project) + directly connected inputNode data.
    // Direct connections take priority; globalData fills in when not directly wired.
    {
      let trackSet = false;
      const mergedData = {};
      if (data && data.globalData) Object.assign(mergedData, data.globalData);
      if (data && data.inputs) {
        for (const inId of this.inputNodes) {
          const upstream = data.inputs[inId];
          if (upstream && upstream.data && upstream.data.trackingData) {
            mergedData[inId] = upstream.data;
          }
        }
      }
      for (const nd of Object.values(mergedData)) {
        if (!trackSet && nd.trackingData && nd.trackingData.length > 0) {
          const det = nd.trackingData[0];
          const kps = det.keypoints || [];
          if (kps.length > 0) {
            const primary = kps[0];
            const tx = gl.getUniformLocation(this.program, 'u_track_x');
            if (tx) gl.uniform1f(tx, primary.x || 0);
            const ty = gl.getUniformLocation(this.program, 'u_track_y');
            if (ty) gl.uniform1f(ty, primary.y || 0);
            const tc = gl.getUniformLocation(this.program, 'u_track_count');
            if (tc) gl.uniform1f(tc, kps.length);
          }
          trackSet = true;
        }
        if (trackSet) break;
      }
    }

    // Main input texture (use blank fallback to avoid feedback loop when no prev layer)
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, prevTex || this._blankTex);
    gl.uniform1i(gl.getUniformLocation(this.program, 'u_input0'), 0);

    // Feedback texture — only bound when the shader uses u_feedback (lazy allocation).
    // When not used, bind blank so the sampler unit is defined but costs nothing.
    let texUnit = 1;
    gl.activeTexture(gl.TEXTURE0 + texUnit);
    gl.bindTexture(gl.TEXTURE_2D, this._feedbackTex || this._blankTex);
    gl.uniform1i(gl.getUniformLocation(this.program, 'u_feedback'), texUnit);
    texUnit += 1;

    // Side inputs
    for (const [inId, tex] of Object.entries(inputs)) {
      if (inId === 'prev_layer_texture') continue;
      if (!tex) continue;

      const loc = gl.getUniformLocation(this.program, `u_${inId}`);
      if (!loc) continue;

      gl.activeTexture(gl.TEXTURE0 + texUnit);
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.uniform1i(loc, texUnit);
      texUnit += 1;
    }

    gl.drawArrays(gl.TRIANGLES, 0, 6);

    gl.bindVertexArray(null);

    // Blit this frame's output → feedback buffer for next frame (zero-copy GPU ping-pong)
    // (fb is already declared above at the top of execute())
    if (fb && this._feedbackFb) {
      gl.bindFramebuffer(gl.READ_FRAMEBUFFER, fb);
      gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, this._feedbackFb);
      gl.blitFramebuffer(0, 0, this.hub.width, this.hub.height,
                         0, 0, this.hub.width, this.hub.height,
                         gl.COLOR_BUFFER_BIT, gl.NEAREST);
      gl.bindFramebuffer(gl.READ_FRAMEBUFFER, null);
      gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, null);
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    this.hub.setNodeData(this.id, {
      type: 'shader',
      time,
      params: this.params
    });
  }

  _compileShader(gl, type, source) {
    // A lost WebGL context makes ALL GL calls no-ops returning null/false.
    // Detect this first so we don't misreport a valid shader as "compile failed"
    // and accidentally trigger Mason auto-fix on working code.
    if (gl.isContextLost()) {
      throw new Error(`[ShaderExecutor:${this.id}] WebGL context is lost — skipping compile`);
    }

    const sh = gl.createShader(type);
    gl.shaderSource(sh, source);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      // Check again: context may have been lost between createShader and compile
      if (gl.isContextLost()) {
        throw new Error(`[ShaderExecutor:${this.id}] WebGL context lost during compile — skipping`);
      }
      const rawInfo = gl.getShaderInfoLog(sh);
      gl.deleteShader(sh);
      // Some GPU drivers return null instead of an error message.
      // In that case, log the full source so we have something to debug with.
      let info;
      if (rawInfo == null || rawInfo.trim() === '') {
        const typeName = type === gl.FRAGMENT_SHADER ? 'FRAGMENT' : 'VERTEX';
        console.warn(`[ShaderExecutor:${this.id}] ${typeName} shader compile failed — driver returned null info log.\nFull source:\n${source}`);
        info = `(GPU driver returned no error details — check browser console for shader source)\nCommon causes: scalar swizzle on float, function redeclaration, gl_FragColor instead of fragColor, texture2D instead of texture().`;
      } else {
        info = rawInfo;
      }
      // Notify hub for RuntimeInspector (compile errors are the most common)
      if (this.hub.onShaderError) {
        this.hub.onShaderError(this.id, info);
      }
      throw new Error(`[ShaderExecutor:${this.id}] Shader compile failed:\n${info}`);
    }
    return sh;
  }

  _createProgram(gl, vsSource, fsSource) {
    const vs = this._compileShader(gl, gl.VERTEX_SHADER, vsSource);
    const fs = this._compileShader(gl, gl.FRAGMENT_SHADER, fsSource);

    const prog = gl.createProgram();
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);

    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      const info = gl.getProgramInfoLog(prog);
      gl.deleteProgram(prog);
      const err = new Error(`[ShaderExecutor:${this.id}] Program link failed:\n${info}`);
      // Notify hub if error callback is registered (for RuntimeInspector)
      if (this.hub.onShaderError) {
        this.hub.onShaderError(this.id, info);
      }
      throw err;
    }

    gl.deleteShader(vs);
    gl.deleteShader(fs);
    return prog;
  }

  destroy() {
    const gl = this.hub.gl;
    if (!gl.isContextLost()) {
      if (this.program) gl.deleteProgram(this.program);
      if (this.vao) gl.deleteVertexArray(this.vao);
      if (this._blankTex) gl.deleteTexture(this._blankTex);
      if (this._feedbackTex) gl.deleteTexture(this._feedbackTex);
      if (this._feedbackFb) gl.deleteFramebuffer(this._feedbackFb);
    }
  }
}

// -----------------------------------------------------------------------------
// Three.js (Geometry / 3D)
// -----------------------------------------------------------------------------

class ThreeExecutor extends BaseExecutor {
  // ── Shared singleton renderer ──────────────────────────────────────────────
  // One WebGL context for ALL ThreeExecutor instances.
  // Per-node offscreen canvases created N separate GL contexts → GPU OOM → "Render process gone".
  static _shared = null;

  static _acquireRenderer(THREE, W, H) {
    if (!ThreeExecutor._shared) {
      const canvas = document.createElement('canvas');
      canvas.width = W || 512; canvas.height = H || 512;
      const renderer = new THREE.WebGLRenderer({
        canvas, antialias: false, alpha: true, preserveDrawingBuffer: false,
      });
      renderer.setPixelRatio(1);
      ThreeExecutor._shared = { renderer, refCount: 0 };
    }
    ThreeExecutor._shared.refCount++;
    return ThreeExecutor._shared.renderer;
  }

  static _releaseRenderer() {
    if (!ThreeExecutor._shared) return;
    if (--ThreeExecutor._shared.refCount <= 0) {
      ThreeExecutor._shared.renderer.dispose();
      ThreeExecutor._shared = null;
    }
  }
  // ──────────────────────────────────────────────────────────────────────────

  async init() {
    this.outputKind = this.outputKind || 'render';
    this.outputTexture = this.hub.allocate(this.id, this.zLayer, { kind: 'render', visible: true });

    // Prefer global THREE if available; otherwise user can use js_module to import.
    const THREE = globalThis.THREE;
    if (!THREE) {
      console.warn(`[ThreeExecutor:${this.id}] globalThis.THREE is not available. Use engine='js_module' if you need imports.`);
      this.initialized = true;
      return;
    }

    // Acquire the shared renderer and create a per-node RenderTarget
    this.renderer = ThreeExecutor._acquireRenderer(THREE, this.hub.width, this.hub.height);
    const W = this.hub.width, H = this.hub.height;
    this.renderTarget = new THREE.WebGLRenderTarget(W, H, {
      minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter,
      depthBuffer: true, stencilBuffer: false,
    });
    // Pre-allocate pixel readback buffer (avoids per-frame allocation)
    this._pixels = new Uint8Array(W * H * 4);
    this._lastW = W; this._lastH = H;

    // Intercept console.error to capture Three.js WebGL shader errors (which are logged,
    // not thrown, so they're invisible to try-catch). Route them to hub.onShaderError.
    const _origConsoleError = console.error;
    const capturedThreeErrors = [];
    console.error = (...args) => {
      const msg = args.map(a => (typeof a === 'object' ? String(a) : a)).join(' ');
      if (msg.includes('THREE.WebGLProgram') || msg.includes('THREE.WebGL') || msg.includes('VALIDATE_STATUS')) {
        capturedThreeErrors.push(msg);
      }
      _origConsoleError.apply(console, args);
    };

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(45, this.hub.width / this.hub.height, 0.1, 1000);
    this.camera.position.z = 5;

    // Compile user code body (no imports)
    let body = sanitizeLLMArtifacts(this.codeSnippet);
    body = this._cleanBody(body);

    this.userFactory = null;
    try {
      const wrappedBody =
        '"use strict";\n' + body + '\n\n' +
        'if (typeof update === "function") return { update };\n' +
        'return {};';
      this.userFactory = new Function('THREE', 'scene', 'camera', 'params', 'inputs', wrappedBody);
    } catch (e) {
      _origConsoleError(`[ThreeExecutor:${this.id}] Failed to compile code body:`, e);
      if (this.hub.onShaderError) this.hub.onShaderError(this.id, `Compile error: ${e.message}`);
    }

    // Patch TextureLoader to block hallucinated local file paths (e.g. 'path/to/texture.jpg').
    // LLMs often generate fake asset paths — intercept them and return a blank 1x1 canvas texture.
    const _origLoad = THREE.TextureLoader?.prototype?.load;
    if (_origLoad) {
      THREE.TextureLoader.prototype.load = function(url, ...rest) {
        const isFake = !url.startsWith('http') && !url.startsWith('blob') && !url.startsWith('data:');
        if (isFake) {
          console.warn(`[ThreeExecutor] Blocked hallucinated texture path: ${url} — returning blank texture`);
          const offscreen = new OffscreenCanvas(1, 1);
          return new THREE.CanvasTexture(offscreen);
        }
        return _origLoad.call(this, url, ...rest);
      };
    }

    this.nodeInstance = null;
    if (this.userFactory) {
      try {
        const initInputs = { textures: {}, data: {} };
        this.nodeInstance = this.userFactory(THREE, this.scene, this.camera, this.params, initInputs);
      } catch (e) {
        _origConsoleError(`[ThreeExecutor:${this.id}] Error executing init body:`, e);
        if (this.hub.onShaderError) this.hub.onShaderError(this.id, `Init error: ${e.message}`);
      }
    }

    // Restore TextureLoader after init (don't pollute global THREE for other nodes)
    if (_origLoad) THREE.TextureLoader.prototype.load = _origLoad;

    // Restore console.error and report any captured Three.js WebGL errors
    console.error = _origConsoleError;
    if (capturedThreeErrors.length > 0 && this.hub.onShaderError) {
      this.hub.onShaderError(this.id, capturedThreeErrors.join('\n'));
    }

    this.initialized = true;
  }

  _cleanBody(body) {
    body = body.replace(/```[\s\S]*?```/g, (m) => {
      const inner = m.replace(/```[\w]*\n?/g, '').replace(/```/g, '');
      return inner;
    });

    const lines = body.split('\n');
    const out = [];
    for (const line of lines) {
      const t = line.trim();

      // Strip drift / whole-file attempts
      if (t.startsWith('<') || t.includes('</') || t.includes('<script')) continue;
      if (/^\s*import\s+/.test(t)) continue;
      if (/^\s*export\s+/.test(t)) continue;
      if (t.includes('require(')) continue;

      // Prevent redeclarations of provided vars
      if (/\b(const|let|var)\s+THREE\b/.test(t)) continue;
      if (/\b(const|let|var)\s+scene\b/.test(t)) continue;
      if (/\b(const|let|var)\s+camera\b/.test(t)) continue;
      if (/\b(const|let|var)\s+params\b/.test(t)) continue;

      out.push(line);
    }
    return out.join('\n').trim();
  }

  async execute(textures, time, data) {
    if (!this.initialized || !this.renderer || !this.renderTarget) return;

    const inputs = { textures: textures || {}, data: data || {} };

    if (this.nodeInstance && typeof this.nodeInstance.update === 'function') {
      try {
        this.nodeInstance.update(time, inputs);
      } catch (e) {
        console.error(`[ThreeExecutor:${this.id}] update() threw:`, e);
        if (this.hub.onShaderError) this.hub.onShaderError(this.id, `Runtime error: ${e.message || e}`);
      }
    }

    const w = this.hub.width, h = this.hub.height;
    // Resize render target and pixel buffer only when needed
    if (this._lastW !== w || this._lastH !== h) {
      this.renderTarget.setSize(w, h);
      this._pixels = new Uint8Array(w * h * 4);
      this._lastW = w; this._lastH = h;
    }

    // Render to per-node RenderTarget (no canvas swap, no separate GL context upload)
    this.renderer.setSize(w, h, false);
    this.renderer.setRenderTarget(this.renderTarget);
    this.renderer.render(this.scene, this.camera);
    this.renderer.setRenderTarget(null);

    // Readback from RenderTarget → pre-allocated buffer → hub texture
    this.renderer.readRenderTargetPixels(this.renderTarget, 0, 0, w, h, this._pixels);
    const gl = this.hub.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.outputTexture);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, this._pixels);
    gl.bindTexture(gl.TEXTURE_2D, null);

    this.hub.setNodeData(this.id, { type: 'three', time, params: this.params });
  }

  destroy() {
    if (this.renderTarget) { this.renderTarget.dispose(); this.renderTarget = null; }
    ThreeExecutor._releaseRenderer();
    this.renderer = null;
  }
}

// -----------------------------------------------------------------------------
// HTML Video / Webcam (Media)
// -----------------------------------------------------------------------------

class VideoExecutor extends BaseExecutor {
  async init() {
    const gl = this.hub.gl;
    this.outputKind = 'render';

    // _rawTex: receives raw video pixels; this.texture: post-flip output
    this._rawTex = gl.createTexture();
    const initTex = (tex) => {
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, this.hub.width, this.hub.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
      gl.bindTexture(gl.TEXTURE_2D, null);
    };
    initTex(this._rawTex);
    this.texture = gl.createTexture();
    initTex(this.texture);

    // Framebuffer for flip blit output
    this._flipFb = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, this._flipFb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.texture, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    // Register output texture in hub
    this.hub.nodeTextures.set(this.id, {
      texture: this.texture,
      framebuffer: this._flipFb,
      z_layer: this.zLayer,
      kind: 'render',
      width: this.hub.width,
      height: this.hub.height,
      internalFormat: gl.RGBA8,
      format: gl.RGBA,
      type: gl.UNSIGNED_BYTE,
      visible: true,
    });

    // Compile flip blit shader (runs every frame to copy _rawTex → texture with optional X flip)
    const vs = `#version 300 es
      in vec2 a_position; out vec2 v_uv;
      void main() { v_uv = a_position * 0.5 + 0.5; gl_Position = vec4(a_position, 0.0, 1.0); }`;
    const fs = `#version 300 es
      precision mediump float;
      in vec2 v_uv; uniform sampler2D u_raw; uniform float u_flip_x; out vec4 fragColor;
      void main() { vec2 uv = v_uv; if (u_flip_x > 0.5) uv.x = 1.0 - uv.x; fragColor = texture(u_raw, uv); }`;
    const compSh = (type, src) => {
      const sh = gl.createShader(type);
      gl.shaderSource(sh, src); gl.compileShader(sh);
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
        const info = gl.getShaderInfoLog(sh); gl.deleteShader(sh);
        throw new Error(`[VideoExecutor:${this.id}] Shader error: ${info}`);
      }
      return sh;
    };
    const vsh = compSh(gl.VERTEX_SHADER, vs), fsh = compSh(gl.FRAGMENT_SHADER, fs);
    this._flipProg = gl.createProgram();
    gl.attachShader(this._flipProg, vsh); gl.attachShader(this._flipProg, fsh);
    gl.linkProgram(this._flipProg);
    gl.deleteShader(vsh); gl.deleteShader(fsh);
    this._flipRawLoc = gl.getUniformLocation(this._flipProg, 'u_raw');
    this._flipXLoc   = gl.getUniformLocation(this._flipProg, 'u_flip_x');

    // Fullscreen quad VAO for blit
    this._flipVao = gl.createVertexArray();
    gl.bindVertexArray(this._flipVao);
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, -1,1, 1,-1, 1,1]), gl.STATIC_DRAW);
    const posLoc = gl.getAttribLocation(this._flipProg, 'a_position');
    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null); gl.bindBuffer(gl.ARRAY_BUFFER, null);
    this._flipQuadBuf = buf;

    // Source state — deferred until user picks source_type (stays -1 until then)
    this._sourceReady = false;
    this._initPending = false;
    this._currentStream = null;
    this._lastSourceType = undefined;
    this._lastSrc = undefined;
    this._assetType = 'video';

    this.initialized = true;
  }

  async _initSource(sourceType, src) {
    // Stop previous stream
    if (this._currentStream) {
      this._currentStream.getTracks().forEach(t => t.stop());
      this._currentStream = null;
    }
    if (this.video && this.video.srcObject && this.video.srcObject !== this._currentStream) {
      const tracks = this.video.srcObject.getTracks ? this.video.srcObject.getTracks() : [];
      tracks.forEach(t => t.stop());
    }

    // Detect asset type from extension
    const ext = src ? src.split('.').pop().split('?')[0].toLowerCase() : '';
    const imageExts = ['png','jpg','jpeg','gif','bmp','webp','svg'];
    const audioExts = ['mp3','wav','ogg','aac','flac','m4a'];
    this._assetType = imageExts.includes(ext) ? 'image' : (audioExts.includes(ext) ? 'audio' : 'video');

    if (this._assetType === 'image' && src) {
      this._imageEl = new Image();
      this._imageEl.crossOrigin = 'anonymous';
      this._imageEl.src = src;
      await new Promise((res, rej) => { this._imageEl.onload = res; this._imageEl.onerror = rej; });
      const gl = this.hub.gl;
      gl.bindTexture(gl.TEXTURE_2D, this._rawTex);
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, this._imageEl);
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
      gl.bindTexture(gl.TEXTURE_2D, null);
      this._runFlipBlit();
      this._lastSourceType = sourceType; this._lastSrc = src; this._sourceReady = true;
      return;
    }

    if (!this.video) {
      this.video = document.createElement('video');
      this.video.autoplay = true; this.video.muted = true; this.video.playsInline = true;
    }

    const isWebcam = (sourceType === 0 || sourceType === 'webcam' || !src || src === 'webcam');
    if (isWebcam) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        this._currentStream = stream;
        this.video.srcObject = stream;
        this.video.src = '';
      } catch (e) {
        console.warn(`[VideoExecutor:${this.id}] getUserMedia failed:`, e.message);
        this._lastSourceType = sourceType; this._lastSrc = src;
        return; // stay black
      }
    } else {
      this.video.srcObject = null;
      this.video.crossOrigin = 'anonymous';
      this.video.src = src;
      this.video.loop = true;
    }
    try { await this.video.play(); } catch (_) {}
    this._lastSourceType = sourceType; this._lastSrc = src; this._sourceReady = true;
  }

  _runFlipBlit() {
    const gl = this.hub.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this._flipFb);
    gl.viewport(0, 0, this.hub.width, this.hub.height);
    gl.useProgram(this._flipProg);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this._rawTex);
    gl.uniform1i(this._flipRawLoc, 0);
    gl.uniform1f(this._flipXLoc, this.params.flip_horizontal ? 1.0 : 0.0);
    gl.bindVertexArray(this._flipVao);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    gl.bindVertexArray(null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  // Called from updateNodeParam hook when source_type or src changes
  reinitSource() {
    const srcType = this.params.source_type;
    if (srcType === undefined || srcType === null || srcType < 0) return; // still on "Select source..."
    this._sourceReady = false;
    this._lastSourceType = undefined;
    this._lastSrc = undefined;
    const src = this.params.src || '';
    if (!this._initPending) {
      this._initPending = true;
      this._initSource(srcType, src).finally(() => { this._initPending = false; });
    }
  }

  async execute(_textures, _time, _data) {
    if (!this.initialized) return;

    // source_type -1 means "Select source..." — stay black until user picks one
    const srcType = this.params.source_type;
    if (srcType === undefined || srcType === null || srcType < 0) return;

    const src = this.params.src || '';

    // Check if source needs (re)init due to param change
    if (!this._sourceReady || srcType !== this._lastSourceType || src !== this._lastSrc) {
      if (!this._initPending) {
        this._initPending = true;
        this._initSource(srcType, src).finally(() => { this._initPending = false; });
      }
      return;
    }

    if (this._assetType === 'image') {
      this.hub.setNodeData(this.id, { type: 'image', ready: true });
      return;
    }

    const gl = this.hub.gl;
    if (this.video && this.video.readyState >= 2) {
      if (typeof this.params.playback_rate === 'number') {
        this.video.playbackRate = this.params.playback_rate;
      }
      // Upload raw frame → _rawTex, then blit with optional flip → this.texture
      gl.bindTexture(gl.TEXTURE_2D, this._rawTex);
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, this.video);
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
      gl.bindTexture(gl.TEXTURE_2D, null);
      this._runFlipBlit();
    }
    this.hub.setNodeData(this.id, { type: this._assetType || 'video', ready: this.video ? this.video.readyState : 0 });
  }

  destroy() {
    if (this._currentStream) this._currentStream.getTracks().forEach(t => t.stop());
    if (this.video && this.video.srcObject) {
      const tracks = this.video.srcObject.getTracks ? this.video.srcObject.getTracks() : [];
      tracks.forEach(t => t.stop());
    }
    const gl = this.hub.gl;
    if (gl && !gl.isContextLost()) {
      if (this._flipProg) gl.deleteProgram(this._flipProg);
      if (this._flipVao) gl.deleteVertexArray(this._flipVao);
      if (this._flipQuadBuf) gl.deleteBuffer(this._flipQuadBuf);
      if (this._rawTex) gl.deleteTexture(this._rawTex);
      if (this._flipFb) gl.deleteFramebuffer(this._flipFb);
      // this.texture owned by hub's nodeTextures — not deleted here
    }
  }
}

// -----------------------------------------------------------------------------
// Canvas2D (CPU draw -> texture)
// -----------------------------------------------------------------------------

class Canvas2DExecutor extends BaseExecutor {
  async init() {
    this.outputKind = this.outputKind || 'render';
    // flipY:true — canvas pixels are uploaded without Y-flip; the blit shader corrects orientation.
    this.outputTexture = this.hub.allocate(this.id, this.zLayer, { kind: 'render', visible: true, flipY: true });

    // Use OffscreenCanvas if available, otherwise fall back to hidden canvas
    const W = this.hub.width;
    const H = this.hub.height;

    this.canvas = (typeof OffscreenCanvas !== 'undefined')
      ? new OffscreenCanvas(W, H)
      : (() => { const c = document.createElement('canvas'); c.width = W; c.height = H; return c; })();

    this.ctx2d = this.canvas.getContext('2d', { willReadFrequently: true });

    let body = sanitizeLLMArtifacts(this.codeSnippet);
    body = this._cleanBody(body);

    // Two supported authoring modes:
    // 1) Factory style: defines draw()/update() or returns an object
    // 2) Per-frame draw BODY: just statements that draw one frame
    const isFactoryStyle = this._looksLikeFactory(body);

    this.userFactory = null;
    this.userDrawFn = null;
    this.nodeInstance = null;

    if (isFactoryStyle) {
      // Auto-bridge: if the snippet defines draw() or update() but forgets to return it,
      // we still provide a runtime update() so the node animates.
      const wrappedBody =
        '"use strict";\n' + body + '\n\n' +
        'if (typeof draw === "function") return { update: (t, inputs) => draw(ctx, width, height, t, inputs, params) };\n' +
        'if (typeof update === "function") return { update };\n' +
        'return {};';

      try {
        this.userFactory = new Function('ctx2d', 'canvas', 'width', 'height', 'ctx', 'params', 'inputs', wrappedBody);
      } catch (e) {
        console.error(`[Canvas2DExecutor:${this.id}] Failed to compile factory body:`, e);
        if (this.hub.onShaderError) this.hub.onShaderError(this.id, `Compile error: ${e.message}`);
      }

      if (this.userFactory) {
        try {
          const initInputs = { textures: {}, data: {} };
          this.nodeInstance = this.userFactory(this.ctx2d, this.canvas, W, H, this.ctx2d, this.params, initInputs);
        } catch (e) {
          console.error(`[Canvas2DExecutor:${this.id}] Error executing init body:`, e);
          if (this.hub.onShaderError) this.hub.onShaderError(this.id, `Init error: ${e.message}`);
          this.nodeInstance = null;
        }
      }
    } else {
      // Per-frame draw BODY mode: run the snippet every frame.
      const perFrameBody = '"use strict";\n' + body;

      try {
        this.userDrawFn = new Function('ctx2d', 'canvas', 'width', 'height', 'ctx', 'params', 'inputs', 't', perFrameBody);
      } catch (e) {
        console.error(`[Canvas2DExecutor:${this.id}] Failed to compile per-frame draw body:`, e);
        if (this.hub.onShaderError) this.hub.onShaderError(this.id, `Compile error: ${e.message}`);
        this.userDrawFn = null;
      }
    }

    this.initialized = true;
  }

  _looksLikeFactory(body) {
    // Heuristic: if they return something or declare draw/update, treat it as factory style.
    if (!body) return false;
    if (/\breturn\b/.test(body)) return true;
    if (/\bfunction\s+draw\b/.test(body)) return true;
    if (/\bfunction\s+update\b/.test(body)) return true;
    if (/\b(draw|update)\s*=\s*\(/.test(body)) return true;          // arrow fn assignment
    if (/\b(draw|update)\s*=\s*function\b/.test(body)) return true;  // function assignment
    return false;
  }

  _cleanBody(body) {
    body = body.replace(/```[\s\S]*?```/g, (m) => {
      const inner = m.replace(/```[\w]*\n?/g, '').replace(/```/g, '');
      return inner;
    });

    const lines = body.split('\n');
    const out = [];
    for (const line of lines) {
      const t = line.trim();

      if (t.startsWith('<') || t.includes('</') || t.includes('<script')) continue;
      if (/^\s*import\s+/.test(t)) continue;
      if (/^\s*export\s+/.test(t)) continue;
      if (t.includes('require(')) continue;

      // Prevent redeclarations of provided vars (they are function args)
      if (/\b(const|let|var)\s+ctx2d\b/.test(t)) continue;
      if (/\b(const|let|var)\s+canvas\b/.test(t)) continue;
      if (/\b(const|let|var)\s+width\b/.test(t)) continue;
      if (/\b(const|let|var)\s+height\b/.test(t)) continue;
      if (/\b(const|let|var)\s+ctx\b/.test(t)) continue;
      if (/\b(const|let|var)\s+params\b/.test(t)) continue;
      if (/\b(const|let|var)\s+inputs\b/.test(t)) continue;

      out.push(line);
    }
    return out.join('\n').trim();
  }

  /**
   * Read a WebGL texture into an ImageData that Canvas2D can draw.
   * Enables Canvas2D nodes to composite upstream textures.
   */
  _readTexture(texId) {
    const gl = this.hub.gl;
    const entry = this.hub.nodeTextures.get(texId);
    if (!entry || !entry.texture) return null;

    const w = entry.width || this.hub.width;
    const h = entry.height || this.hub.height;

    // Create temporary framebuffer to read from the texture
    if (!this._readFb) {
      this._readFb = gl.createFramebuffer();
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, this._readFb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, entry.texture, 0);

    const size = w * h * 4;
    // Reuse pre-allocated buffers — avoids two large TypedArray allocations per call
    if (!this._readPixelsBuf || this._readPixelsBuf.length !== size) {
      this._readPixelsBuf = new Uint8Array(size);
      this._readFlippedBuf = new Uint8ClampedArray(size);
    }
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, this._readPixelsBuf);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    // WebGL reads bottom-up, flip vertically for Canvas2D
    const rowSize = w * 4;
    for (let y = 0; y < h; y++) {
      const srcRow = (h - 1 - y) * rowSize;
      const dstRow = y * rowSize;
      this._readFlippedBuf.set(this._readPixelsBuf.subarray(srcRow, srcRow + rowSize), dstRow);
    }

    return new ImageData(this._readFlippedBuf, w, h);
  }

  async execute(_textures, time, data) {
    if (!this.initialized || !this.ctx2d) return;

    // Build inputs with readTexture helper so Canvas2D nodes can sample upstream textures.
    // Merge globalData (all audio/tracking from any project node) + direct connection data.
    // Direct connections take priority; globalData fills in the rest.
    const self = this;
    const upstreamData = {};
    if (data && data.globalData) Object.assign(upstreamData, data.globalData);
    if (data && data.inputs) Object.assign(upstreamData, data.inputs);
    const inputs = {
      textures: _textures || {},
      data: upstreamData,
      frame: data || {},
      readTexture: (texId) => self._readTexture(texId),
    };

    if (this.nodeInstance && typeof this.nodeInstance.update === 'function') {
      try {
        this.nodeInstance.update(time, inputs);
      } catch (e) {
        console.error(`[Canvas2DExecutor:${this.id}] update() threw:`, e);
        if (this.hub.onShaderError) this.hub.onShaderError(this.id, `Runtime error: ${e.message || e}`);
      }
    } else if (this.userDrawFn) {
      try {
        this.userDrawFn(this.ctx2d, this.canvas, this.canvas.width, this.canvas.height, this.ctx2d, this.params, inputs, time);
      } catch (e) {
        console.error(`[Canvas2DExecutor:${this.id}] per-frame draw threw:`, e);
        if (this.hub.onShaderError) this.hub.onShaderError(this.id, `Runtime error: ${e.message || e}`);
      }
    }

    // Upload canvas to output texture — synchronous, no extra GPU objects.
    // Y-flip is handled by the blit shader (v_uv.y inversion) so we avoid
    // UNPACK_FLIP_Y_WEBGL (CPU pixel copy) and createImageBitmap (async GPU pressure).
    const gl = this.hub.gl;
    if (gl.isContextLost()) return;
    gl.bindTexture(gl.TEXTURE_2D, this.outputTexture);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, gl.RGBA, gl.UNSIGNED_BYTE, this.canvas);
    gl.bindTexture(gl.TEXTURE_2D, null);

    // Pass through tracking data if the canvas code stored it
    const nodeData = { type: 'canvas2d', time, params: this.params };
    if (this.canvas._trackingData) {
      nodeData.trackingData = this.canvas._trackingData;
    }
    this.hub.setNodeData(this.id, nodeData);
  }

  destroy() {
    const gl = this.hub.gl;
    if (this._readFb && !gl.isContextLost()) {
      gl.deleteFramebuffer(this._readFb);
    }
    if (this.nodeInstance?.cleanup) {
      try { this.nodeInstance.cleanup(); } catch (_) {}
    }
  }
}

// -----------------------------------------------------------------------------
// JS Module (ESM with imports) -> draws to canvas -> texture
//
// This is the "point 6" engine: snippets can include imports.
// Contract for js_module code:
//
//   export default async function create({ canvas, ctx2d, params, inputs, hub }) {
//     // setup...
//     return {
//       update(t, inputs) { ...draw to ctx2d/canvas... }
//     };
//   }
//
// - The runtime rewrites bare imports using globalThis.__MASON_IMPORTS__
//   or a CDN fallback (esm.sh).
// -----------------------------------------------------------------------------

class JSModuleExecutor extends BaseExecutor {
  async init() {
    this.outputKind = this.outputKind || 'render';
    // flipY:true — canvas pixels are uploaded without Y-flip; the blit shader corrects orientation.
    this.outputTexture = this.hub.allocate(this.id, this.zLayer, { kind: 'render', visible: true, flipY: true });

    const W = this.hub.width;
    const H = this.hub.height;

    // Create a dedicated canvas for this node — the library (Three.js, P5, etc.) owns it
    this.canvas = document.createElement('canvas');
    this.canvas.width = W;
    this.canvas.height = H;

    // Do NOT pre-acquire the 2D context here. Let p5 (or whatever module) create
    // its own GPU-accelerated context. Forcing willReadFrequently:true here would
    // lock the canvas into software (CPU) rasterization, killing draw performance.
    // Canvas2DExecutor (tracking/data nodes) keeps willReadFrequently separately.

    // Register canvas so downstream p5 nodes can access it via inputs[]
    this.hub.setNodeCanvas(this.id, this.canvas);

    let code = sanitizeLLMArtifacts(this.codeSnippet);
    code = rewriteBareImports(code);

    // Auto-wrap non-ESM code as an ESM module with init(canvas) contract
    if (!/\bexport\b/.test(code)) {
      console.warn(`[JSModuleExecutor:${this.id}] No 'export' found, auto-wrapping as ESM init(canvas) module`);
      code = `
export default function init(canvas, width, height, params) {
  const ctx = canvas.getContext('2d');
  ${code}
  if (typeof draw === 'function') return (t, inp) => draw(ctx, width, height, t, inp, params);
  if (typeof update === 'function') return (t, inp) => update(t, inp);
  return () => {};
}`;
    }

    // Detect nested pixel loops before the module even runs — these cause 15-second browser hangs.
    // Only block increment-by-1 loops (x++ or x+=1) over canvas dimensions; stride loops are fine.
    // Throw so the loadProject() catch block handles it: red-tint fallback + batch fix report.
    const pixelLoopFwd = /for\s*\([^)]*\bwidth\b[^;]*;\s*(?:\w+\+\+|\w+\s*\+=\s*1\b)[^)]*\)[\s\S]{0,400}for\s*\([^)]*\bheight\b[^;]*;\s*(?:\w+\+\+|\w+\s*\+=\s*1\b)[^)]*\)/;
    const pixelLoopRev = /for\s*\([^)]*\bheight\b[^;]*;\s*(?:\w+\+\+|\w+\s*\+=\s*1\b)[^)]*\)[\s\S]{0,400}for\s*\([^)]*\bwidth\b[^;]*;\s*(?:\w+\+\+|\w+\s*\+=\s*1\b)[^)]*\)/;
    if (pixelLoopFwd.test(code) || pixelLoopRev.test(code)) {
      throw new Error(
        'Performance: nested pixel loop detected (for x<width; x++ inside for y<height; y++). ' +
        'This creates 262,144+ draw calls per frame and freezes the browser. ' +
        'Use s.background() for fills, or ≤500 random points for starfields/particles.'
      );
    }

    const blob = new Blob([code], { type: 'text/javascript' });
    const url = URL.createObjectURL(blob);
    this._moduleUrl = url;

    this.updateFunc = null;
    this.cleanupFunc = null;

    try {
      const mod = await import(/* @vite-ignore */ url);
      // Revoke blob URL immediately after import — browser has compiled the module, URL no longer needed
      URL.revokeObjectURL(url);
      this._moduleUrl = null;

      const initFn = mod.default || mod.init || mod.create || mod.createNode;

      if (typeof initFn !== 'function') {
        throw new Error(`js_module must export default function init(canvas, w, h, params)`);
      }

      // Call init(canvas, width, height, params) — the library takes ownership of the canvas
      const result = await initFn(this.canvas, W, H, this.params);

      if (typeof result === 'function') {
        // Three.js style: returns (time, inputs) => { ... }
        this.updateFunc = result;
      } else if (result && typeof result === 'object') {
        // Object style: { update, dispose }
        if (typeof result.update === 'function') this.updateFunc = result.update;
        if (typeof result.dispose === 'function') this.cleanupFunc = result.dispose;
      }

      // p5.js defers s.setup() (which binds to our canvas) to the next animation frame.
      // Wait one RAF so setup() runs before the first execute() call; otherwise
      // sketch.redraw() draws on p5's internal default canvas → first frame is black.
      await new Promise(r => requestAnimationFrame(r));
    } catch (e) {
      URL.revokeObjectURL(url);
      this._moduleUrl = null;
      console.error(`[JSModuleExecutor:${this.id}] Failed to load module:`, e);
      // Route to error pipeline so Mason can fix the code (same path as GLSL shader errors)
      if (this.hub.onShaderError) this.hub.onShaderError(this.id, e.message || String(e));
    }

    this.initialized = true;
  }

  async execute(_textures, time, _data) {
    if (!this.initialized) return;

    // Build ordered array of input canvases for p5 nodes (inputs[0], inputs[1], etc.)
    const inputCanvases = this.inputNodes.map(id => this.hub.getNodeCanvas(id));

    // Build rich data map: global audio/tracking (any source node) + directly connected data.
    // Direct connections override global data so canvas/texture inputs aren't shadowed.
    const inputDataMap = {};
    if (_data && _data.globalData) Object.assign(inputDataMap, _data.globalData);
    for (const id of this.inputNodes) {
      const d = this.hub.getNodeData(id);
      if (d) inputDataMap[id] = d;
    }

    if (this.updateFunc) {
      try {
        await this.updateFunc(time, inputCanvases, inputDataMap);
      } catch (e) {
        console.warn(`[JSModuleExecutor:${this.id}] update() threw:`, e);
        // Runtime execution errors also go to the fix pipeline
        if (this.hub.onShaderError) this.hub.onShaderError(this.id, `Runtime error: ${e.message || e}`);
      }
    }

    // Upload canvas directly — synchronous, no macrotask gap that lets p5's RAF accumulate.
    const gl = this.hub.gl;
    if (gl.isContextLost()) return;
    gl.bindTexture(gl.TEXTURE_2D, this.outputTexture);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, gl.RGBA, gl.UNSIGNED_BYTE, this.canvas);
    gl.bindTexture(gl.TEXTURE_2D, null);

    this.hub.setNodeData(this.id, { type: 'js_module', time });
  }

  destroy() {
    if (this.cleanupFunc) {
      try { this.cleanupFunc(); } catch (_) {}
    }
    if (this._moduleUrl) URL.revokeObjectURL(this._moduleUrl);
  }
}

// -----------------------------------------------------------------------------
// WebAudio (Audio -> DataTexture)
// -----------------------------------------------------------------------------

class WebAudioExecutor extends BaseExecutor {
  async init() {
    this.outputKind = this.outputKind || 'data';

    const AudioCtx = globalThis.AudioContext || globalThis.webkitAudioContext;
    if (!AudioCtx) {
      console.warn(`[WebAudioExecutor:${this.id}] AudioContext not supported`);
      this.initialized = true;
      return;
    }

    this.audioCtx = new AudioCtx();

    // Compile factory but do NOT run it yet (running triggers getUserMedia for mic)
    let body = sanitizeLLMArtifacts(this.codeSnippet);
    body = this._cleanBody(body);
    this._factory = null;
    try {
      this._factory = new Function('audioCtx', 'params', 'inputs', body);
    } catch (e) {
      console.error(`[WebAudioExecutor:${this.id}] Failed to compile code body:`, e);
      if (this.hub.onShaderError) this.hub.onShaderError(this.id, `Compile error: ${e.message || e}`);
    }

    // Allocate a minimal data texture (1 bin, silent) — resized when source inits
    this.dataWidth = 1; this.dataHeight = 1;
    this.hub.allocateDataTexture(this.id, this.zLayer, 1, 1, {
      internalFormat: this.hub.gl.R8,
      format: this.hub.gl.RED,
      type: this.hub.gl.UNSIGNED_BYTE,
      visible: false,
    });

    // Source deferred until user picks source_type (empty string = unselected)
    this._sourceReady = false;
    this._initPending = false;
    this.audioNode = null;
    this._lastFftSize = null;
    this._lastSourceType = undefined;
    this._lastSrc = undefined;

    this.initialized = true;
  }

  async _initAudioSource() {
    if (!this._factory || !this.audioCtx) return;

    // Stop any previous audio nodes
    if (this.audioNode && typeof this.audioNode.dispose === 'function') {
      try { this.audioNode.dispose(); } catch (_) {}
    }
    this.audioNode = null;

    // Resume AudioContext (may be suspended until user gesture)
    if (this.audioCtx.state === 'suspended') {
      await this.audioCtx.resume().catch(() => {});
    }

    try {
      const initInputs = { textures: {}, data: {} };
      this.audioNode = this._factory(this.audioCtx, this.params, initInputs);
    } catch (e) {
      console.error(`[WebAudioExecutor:${this.id}] Error executing init body:`, e);
    }

    // Ensure analyser/dataArray/update exist
    if (!this.audioNode || !this.audioNode.analyser) {
      const analyser = this.audioCtx.createAnalyser();
      analyser.fftSize = this.params.fft_size || this.params.fftSize || 256;
      this.audioNode = this.audioNode || {};
      this.audioNode.analyser = analyser;
    }
    if (!this.audioNode.dataArray) {
      this.audioNode.dataArray = new Uint8Array(this.audioNode.analyser.frequencyBinCount);
    }
    if (typeof this.audioNode.update !== 'function') {
      this.audioNode.update = () => {
        this.audioNode.analyser.getByteFrequencyData(this.audioNode.dataArray);
        return this.audioNode.dataArray;
      };
    }

    // Reallocate data texture with correct FFT size
    this.dataWidth = this.audioNode.dataArray.length;
    this.dataHeight = 1;
    this.hub.allocateDataTexture(this.id, this.zLayer, this.dataWidth, this.dataHeight, {
      internalFormat: this.hub.gl.R8,
      format: this.hub.gl.RED,
      type: this.hub.gl.UNSIGNED_BYTE,
      visible: false,
    });

    this._lastFftSize = this.audioNode.analyser.fftSize;
    this._lastSourceType = this.params.source_type;
    this._lastSrc = this.params.src;
    this._sourceReady = true;
  }

  // Called from updateNodeParam hook when source_type or src changes
  reinitSource() {
    if (!this.params.source_type) return; // still on "Select source..."
    this._sourceReady = false;
    if (!this._initPending) {
      this._initPending = true;
      this._initAudioSource().finally(() => { this._initPending = false; });
    }
  }

  _cleanBody(body) {
    body = body.replace(/```[\s\S]*?```/g, (m) => {
      const inner = m.replace(/```[\w]*\n?/g, '').replace(/```/g, '');
      return inner;
    });

    const lines = body.split('\n');
    const out = [];
    for (const line of lines) {
      const t = line.trim();

      if (t.startsWith('<') || t.includes('</') || t.includes('<script')) continue;
      if (/^\s*import\s+/.test(t)) continue;
      if (/^\s*export\s+/.test(t)) continue;
      if (t.includes('require(')) continue;

      if (/\b(const|let|var)\s+audioCtx\b/.test(t)) continue;
      if (/\b(const|let|var)\s+params\b/.test(t)) continue;

      // Avoid browser-only globals, but allow navigator.mediaDevices (needed for mic)
      if (/\bwindow\b|\bdocument\b/.test(t)) continue;
      if (/\bnavigator\b/.test(t) && !/mediaDevices|getUserMedia/.test(t)) continue;

      out.push(line);
    }
    return out.join('\n').trim();
  }

  async execute(_textures, time, data) {
    if (!this.initialized) return;

    // source_type empty = "Select source..." — stay silent until user picks one
    if (!this.params.source_type) return;

    // Init source on first call or when source params changed
    if (!this._sourceReady || !this.audioNode) {
      if (!this._initPending) {
        this._initPending = true;
        this._initAudioSource().finally(() => { this._initPending = false; });
      }
      return;
    }

    // Live param updates (no reinit needed for these)
    if (this.audioNode.analyser) {
      this.audioNode.analyser.smoothingTimeConstant = this.params.smoothing ?? 0.8;
    }
    if (this.audioNode.gainNode) {
      this.audioNode.gainNode.gain.value = this.params.gain ?? 1.0;
    }
    // fft_size change requires full reinit
    const curFftSize = this.params.fft_size || this.params.fftSize || 256;
    if (this._lastFftSize && curFftSize !== this._lastFftSize) {
      this._initPending = true;
      this._initAudioSource().finally(() => { this._initPending = false; });
      return;
    }

    let arr = null;
    try {
      const inputs = { textures: _textures || {}, data: data || {} };
      arr = this.audioNode.update(time, inputs) || this.audioNode.dataArray;
    } catch (e) {
      console.warn(`[WebAudioExecutor:${this.id}] update() threw:`, e);
      arr = this.audioNode.dataArray;
    }

    let bytes = null;
    if (arr instanceof Uint8Array) bytes = arr;
    else if (arr instanceof Float32Array || Array.isArray(arr)) {
      // Reuse pre-allocated conversion buffer — avoids new Uint8Array every frame
      if (!this._convBuf || this._convBuf.length !== arr.length) {
        this._convBuf = new Uint8Array(arr.length);
      }
      for (let i = 0; i < arr.length; i++) this._convBuf[i] = Math.round(clamp01(arr[i]) * 255);
      bytes = this._convBuf;
    } else bytes = this.audioNode.dataArray;

    if (bytes.length !== this.dataWidth) {
      this.dataWidth = bytes.length;
      this.hub.allocateDataTexture(this.id, this.zLayer, this.dataWidth, 1, {
        internalFormat: this.hub.gl.R8,
        format: this.hub.gl.RED,
        type: this.hub.gl.UNSIGNED_BYTE,
        visible: false,
      });
    }

    this.hub.updateDataTexture(this.id, this.zLayer, this.dataWidth, 1, bytes, {
      internalFormat: this.hub.gl.R8,
      format: this.hub.gl.RED,
      type: this.hub.gl.UNSIGNED_BYTE,
    });

    // Compute frequency bands: bass (0-20%), mid (20-60%), treble (60-100%)
    const len = bytes.length;
    let avg = 0, bassSum = 0, midSum = 0, trebSum = 0;
    const bassEnd = Math.floor(len * 0.2);
    const midEnd = Math.floor(len * 0.6);
    for (let i = 0; i < len; i++) {
      const v = bytes[i];
      avg += v;
      if (i < bassEnd) bassSum += v;
      else if (i < midEnd) midSum += v;
      else trebSum += v;
    }
    avg = len ? avg / (len * 255) : 0;
    const bass = bassEnd ? bassSum / (bassEnd * 255) : 0;
    const mid = (midEnd - bassEnd) ? midSum / ((midEnd - bassEnd) * 255) : 0;
    const treble = (len - midEnd) ? trebSum / ((len - midEnd) * 255) : 0;

    this.hub.setNodeData(this.id, {
      type: 'audio', time, level: avg, bass, mid, treble, bins: len,
      audioData: { level: avg, bass, mid, treble }
    });
  }

  destroy() {
    if (this.audioCtx) this.audioCtx.close();
  }
}

// -----------------------------------------------------------------------------
// Events / Control (State -> DataTexture + JSON)
// -----------------------------------------------------------------------------

class EventExecutor extends BaseExecutor {
  async init() {
    this.outputKind = this.outputKind || 'data';

    let body = sanitizeLLMArtifacts(this.codeSnippet);
    body = this._cleanBody(body);

    let factory = null;
    try {
      factory = new Function('params', 'inputs', body);
    } catch (e) {
      console.error(`[EventExecutor:${this.id}] Failed to compile code body:`, e);
      if (this.hub.onShaderError) this.hub.onShaderError(this.id, `Compile error: ${e.message || e}`);
    }

    this.nodeInstance = null;
    if (factory) {
      try {
        const initInputs = { textures: {}, data: {} };
        this.nodeInstance = factory(this.params, initInputs);
      } catch (e) {
        console.error(`[EventExecutor:${this.id}] Error executing init body:`, e);
        if (this.hub.onShaderError) this.hub.onShaderError(this.id, `Init error: ${e.message || e}`);
      }
    }

    if (!this.nodeInstance || typeof this.nodeInstance.get !== 'function') {
      const state = { value: this.params.default ?? 0.0 };
      this.nodeInstance = {
        state,
        get: () => state.value,
        set: (v) => { state.value = v; },
      };
    }

    this.hub.allocate(this.id, this.zLayer, {
      kind: 'data',
      width: 1,
      height: 1,
      internalFormat: this.hub.gl.RGBA8,
      format: this.hub.gl.RGBA,
      type: this.hub.gl.UNSIGNED_BYTE,
      visible: false,
    });

    // Pre-allocate reusable RGBA buffer — avoids new Uint8Array(4) every frame
    this._rgba = new Uint8Array(4);

    this.initialized = true;
  }

  _cleanBody(body) {
    body = body.replace(/```[\s\S]*?```/g, (m) => {
      const inner = m.replace(/```[\w]*\n?/g, '').replace(/```/g, '');
      return inner;
    });

    const lines = body.split('\n');
    const out = [];
    for (const line of lines) {
      const t = line.trim();
      if (t.startsWith('<') || t.includes('</') || t.includes('<script')) continue;
      if (/^\s*import\s+/.test(t)) continue;
      if (/^\s*export\s+/.test(t)) continue;
      if (t.includes('require(')) continue;
      if (/\b(const|let|var)\s+params\b/.test(t)) continue;
      out.push(line);
    }
    return out.join('\n').trim();
  }

  _encodeStateToRGBA(state) {
    const vals = [];
    if (typeof state === 'number') vals.push(state);
    else if (state && typeof state === 'object') {
      for (const v of Object.values(state)) {
        if (typeof v === 'number') vals.push(v);
        if (vals.length >= 4) break;
      }
    }
    while (vals.length < 4) vals.push(0);

    // Reuse pre-allocated buffer instead of allocating new Uint8Array every frame
    const bytes = this._rgba || (this._rgba = new Uint8Array(4));
    for (let i = 0; i < 4; i++) {
      const x = vals[i];
      let y = 0;
      if (Number.isFinite(x)) {
        if (x >= 0 && x <= 1) y = x;
        else y = (Math.tanh(x) + 1) * 0.5;
      }
      bytes[i] = Math.round(clamp01(y) * 255);
    }
    return bytes;
  }

  async execute(_textures, time, data) {
    if (!this.initialized || !this.nodeInstance) return;

    if (typeof this.nodeInstance.update === 'function') {
      try {
        const inputs = { textures: _textures || {}, data: data || {} };
        await this.nodeInstance.update(time, inputs);
      } catch (e) {
        console.warn(`[EventExecutor:${this.id}] update() threw:`, e);
      }
    }

    const state = this.nodeInstance.state ?? { value: this.nodeInstance.get() };
    const rgba = this._encodeStateToRGBA(state);

    this.hub.updateDataTexture(this.id, this.zLayer, 1, 1, rgba, {
      internalFormat: this.hub.gl.RGBA8,
      format: this.hub.gl.RGBA,
      type: this.hub.gl.UNSIGNED_BYTE,
    });

    this.hub.setNodeData(this.id, { type: 'event', time, state });
  }
}

// -----------------------------------------------------------------------------
// Registry
// -----------------------------------------------------------------------------

function createExecutor(spec, hub) {
  const engine = spec.engine;

  // Normalize spec shape
  spec = {
    id: spec.id,
    engine: engine,
    code_snippet: spec.code_snippet || '',
    parameters: spec.parameters || {},
    z_layer: spec.z_layer ?? spec.grid_position?.[2] ?? 0,
    input_nodes: spec.input_nodes || [],
    output_kind: spec.output_kind || null,
  };

  switch (engine) {
    case 'glsl':
    case 'regl':
      return new ShaderExecutor(spec, hub);

    case 'three_js':
      // Three.js nodes now use ESM with init(canvas) contract
      // JSModuleExecutor handles the import() and canvas ownership
      return new JSModuleExecutor(spec, hub);

    case 'webaudio':
      return new WebAudioExecutor(spec, hub);

    case 'events':
      return new EventExecutor(spec, hub);

    case 'html_video':
      return new VideoExecutor(spec, hub);

    case 'canvas2d':
      return new Canvas2DExecutor(spec, hub);

    case 'js_module':
      return new JSModuleExecutor(spec, hub);

    // P5.js uses ESM import — needs JSModuleExecutor (NOT Canvas2DExecutor)
    case 'p5':
    case 'p5js':
    case 'p5.js':
      return new JSModuleExecutor(spec, hub);

    // If you add taichi/tfjs/mediapipe engines in the planner, map them to js_module for now.
    case 'taichi':
    case 'tensorflow_js':
    case 'mediapipe':
      return new JSModuleExecutor(spec, hub);

    default:
      console.warn(`[createExecutor] Unknown engine '${engine}', using passthrough GLSL shader.`);
      return new ShaderExecutor({
        ...spec,
        engine: 'glsl',
        code_snippet: `void main(){ vec2 uv=v_uv; fragColor = texture(u_input0, uv); }`
      }, hub);
  }
}
