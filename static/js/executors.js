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
// Mason prompts reference snoise(), fbm(), and hash().
// We provide a compatible snoise alias (not true simplex but sufficient).
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

function rewriteBareImports(code) {
  let out = String(code);

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

    let extra = '';
    for (const u of used) {
      if (!declared.has(u)) {
        // Heuristic type inference from naming
        const lower = u.toLowerCase();
        if (/resolution|size|offset|pos|coord|mouse/.test(lower)) {
          extra += `      uniform vec2 ${u};\n`;
        } else if (/color|col|rgb|tint/.test(lower)) {
          extra += `      uniform vec3 ${u};\n`;
        } else if (/tex|texture|sampler|map/.test(lower)) {
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
    for (const nodeId of (this.inputNodes || [])) {
      if (!nodeId) continue;
      decl += `      uniform sampler2D u_${nodeId};\n`;
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

    // Main input texture (use blank fallback to avoid feedback loop when no prev layer)
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, prevTex || this._blankTex);
    gl.uniform1i(gl.getUniformLocation(this.program, 'u_input0'), 0);

    // Side inputs
    let texUnit = 1;
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
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    this.hub.setNodeData(this.id, {
      type: 'shader',
      time,
      params: this.params
    });
  }

  _compileShader(gl, type, source) {
    const sh = gl.createShader(type);
    gl.shaderSource(sh, source);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      const info = gl.getShaderInfoLog(sh);
      gl.deleteShader(sh);
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
    if (this.program) gl.deleteProgram(this.program);
    if (this.vao) gl.deleteVertexArray(this.vao);
    if (this._blankTex) gl.deleteTexture(this._blankTex);
  }
}

// -----------------------------------------------------------------------------
// Three.js (Geometry / 3D)
// -----------------------------------------------------------------------------

class ThreeExecutor extends BaseExecutor {
  async init() {
    this.outputKind = this.outputKind || 'render';
    this.outputTexture = this.hub.allocate(this.id, this.zLayer, { kind: 'render', visible: true });

    // Offscreen canvas for THREE renderer
    this.offCanvas = document.createElement('canvas');
    this.offCanvas.width = this.hub.width;
    this.offCanvas.height = this.hub.height;

    // Prefer global THREE if available; otherwise user can use js_module to import.
    const THREE = globalThis.THREE;
    if (!THREE) {
      console.warn(`[ThreeExecutor:${this.id}] globalThis.THREE is not available. Use engine='js_module' if you need imports.`);
      this.initialized = true;
      return;
    }

    this.renderer = new THREE.WebGLRenderer({
      canvas: this.offCanvas,
      antialias: true,
      alpha: true,
      preserveDrawingBuffer: true
    });

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
      console.error(`[ThreeExecutor:${this.id}] Failed to compile code body:`, e);
    }

    this.nodeInstance = null;
    if (this.userFactory) {
      try {
        const initInputs = { textures: {}, data: {} };
        this.nodeInstance = this.userFactory(THREE, this.scene, this.camera, this.params, initInputs);
      } catch (e) {
        console.error(`[ThreeExecutor:${this.id}] Error executing init body:`, e);
      }
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
    if (!this.initialized || !this.renderer) return;

    const inputs = { textures: textures || {}, data: data || {} };

    if (this.nodeInstance && typeof this.nodeInstance.update === 'function') {
      try {
        this.nodeInstance.update(time, inputs);
      } catch (e) {
        console.warn(`[ThreeExecutor:${this.id}] update() threw:`, e);
      }
    }

    this.renderer.setSize(this.hub.width, this.hub.height, false);
    this.renderer.render(this.scene, this.camera);

    const gl = this.hub.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.outputTexture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, this.offCanvas);
    gl.bindTexture(gl.TEXTURE_2D, null);

    this.hub.setNodeData(this.id, { type: 'three', time, params: this.params });
  }

  destroy() {
    if (this.renderer) this.renderer.dispose();
  }
}

// -----------------------------------------------------------------------------
// HTML Video / Webcam (Media)
// -----------------------------------------------------------------------------

class VideoExecutor extends BaseExecutor {
  async init() {
    const gl = this.hub.gl;
    this.outputKind = this.outputKind || 'render';

    // Create output texture (updated from video each frame)
    this.texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, this.hub.width, this.hub.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.bindTexture(gl.TEXTURE_2D, null);

    // Register in hub
    this.hub.nodeTextures.set(this.id, {
      texture: this.texture,
      framebuffer: null,
      z_layer: this.zLayer,
      kind: 'render',
      width: this.hub.width,
      height: this.hub.height,
      internalFormat: gl.RGBA8,
      format: gl.RGBA,
      type: gl.UNSIGNED_BYTE,
      visible: true,
    });

    // Create media element
    this.video = document.createElement('video');
    this.video.autoplay = true;
    this.video.muted = true;
    this.video.playsInline = true;

    const src = (this.codeSnippet || '').trim();
    if (src && src !== 'webcam') {
      this.video.src = src;
      try { await this.video.play(); } catch (_) {}
    } else {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      this.video.srcObject = stream;
      await this.video.play();
    }

    this.initialized = true;
  }

  async execute(_textures, _time, _data) {
    if (!this.initialized) return;

    const gl = this.hub.gl;
    if (this.video.readyState >= 2) {
      gl.bindTexture(gl.TEXTURE_2D, this.texture);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, this.video);
      gl.bindTexture(gl.TEXTURE_2D, null);
    }

    this.hub.setNodeData(this.id, { type: 'video', ready: this.video.readyState });
  }

  destroy() {
    if (this.video && this.video.srcObject) {
      const tracks = this.video.srcObject.getTracks ? this.video.srcObject.getTracks() : [];
      tracks.forEach(t => t.stop());
    }
  }
}

// -----------------------------------------------------------------------------
// Canvas2D (CPU draw -> texture)
// -----------------------------------------------------------------------------

class Canvas2DExecutor extends BaseExecutor {
  async init() {
    this.outputKind = this.outputKind || 'render';
    this.outputTexture = this.hub.allocate(this.id, this.zLayer, { kind: 'render', visible: true });

    // Use OffscreenCanvas if available, otherwise fall back to hidden canvas
    const W = this.hub.width;
    const H = this.hub.height;

    this.canvas = (typeof OffscreenCanvas !== 'undefined')
      ? new OffscreenCanvas(W, H)
      : (() => { const c = document.createElement('canvas'); c.width = W; c.height = H; return c; })();

    this.ctx2d = this.canvas.getContext('2d');

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
      }

      if (this.userFactory) {
        try {
          const initInputs = { textures: {}, data: {} };
          this.nodeInstance = this.userFactory(this.ctx2d, this.canvas, W, H, this.ctx2d, this.params, initInputs);
        } catch (e) {
          console.error(`[Canvas2DExecutor:${this.id}] Error executing init body:`, e);
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

    const pixels = new Uint8Array(w * h * 4);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    // WebGL reads bottom-up, flip vertically for Canvas2D
    const flipped = new Uint8ClampedArray(w * h * 4);
    const rowSize = w * 4;
    for (let y = 0; y < h; y++) {
      const srcRow = (h - 1 - y) * rowSize;
      const dstRow = y * rowSize;
      flipped.set(pixels.subarray(srcRow, srcRow + rowSize), dstRow);
    }

    return new ImageData(flipped, w, h);
  }

  async execute(_textures, time, data) {
    if (!this.initialized || !this.ctx2d) return;

    // Build inputs with readTexture helper so Canvas2D nodes can sample upstream textures
    const self = this;
    const inputs = {
      textures: _textures || {},
      data: data || {},
      readTexture: (texId) => self._readTexture(texId),
    };

    if (this.nodeInstance && typeof this.nodeInstance.update === 'function') {
      try {
        this.nodeInstance.update(time, inputs);
      } catch (e) {
        console.warn(`[Canvas2DExecutor:${this.id}] update() threw:`, e);
      }
    } else if (this.userDrawFn) {
      try {
        this.userDrawFn(this.ctx2d, this.canvas, this.canvas.width, this.canvas.height, this.ctx2d, this.params, inputs, time);
      } catch (e) {
        console.warn(`[Canvas2DExecutor:${this.id}] per-frame draw threw:`, e);
      }
    }

    // Upload canvas to output texture
    const gl = this.hub.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.outputTexture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, this.canvas);
    gl.bindTexture(gl.TEXTURE_2D, null);

    this.hub.setNodeData(this.id, { type: 'canvas2d', time, params: this.params });
  }

  destroy() {
    if (this._readFb) {
      this.hub.gl.deleteFramebuffer(this._readFb);
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
    this.outputTexture = this.hub.allocate(this.id, this.zLayer, { kind: 'render', visible: true });

    const W = this.hub.width;
    const H = this.hub.height;

    // Create a dedicated canvas for this node — the library (Three.js, P5, etc.) owns it
    this.canvas = document.createElement('canvas');
    this.canvas.width = W;
    this.canvas.height = H;

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

    const blob = new Blob([code], { type: 'text/javascript' });
    const url = URL.createObjectURL(blob);
    this._moduleUrl = url;

    this.updateFunc = null;
    this.cleanupFunc = null;

    try {
      const mod = await import(/* @vite-ignore */ url);
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
    } catch (e) {
      console.error(`[JSModuleExecutor:${this.id}] Failed to load module:`, e);
    }

    this.initialized = true;
  }

  async execute(textures, time, data) {
    if (!this.initialized) return;

    const inputs = { textures: textures || {}, data: data || {} };

    if (this.updateFunc) {
      try {
        await this.updateFunc(time, inputs);
      } catch (e) {
        console.warn(`[JSModuleExecutor:${this.id}] update() threw:`, e);
      }
    }

    // Upload the node's canvas to the output texture
    const gl = this.hub.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.outputTexture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, this.canvas);
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

    let body = sanitizeLLMArtifacts(this.codeSnippet);
    body = this._cleanBody(body);

    let factory = null;
    try {
      factory = new Function('audioCtx', 'params', 'inputs', body);
    } catch (e) {
      console.error(`[WebAudioExecutor:${this.id}] Failed to compile code body:`, e);
    }

    this.audioNode = null;
    if (factory) {
      try {
        const initInputs = { textures: {}, data: {} };
        this.audioNode = factory(this.audioCtx, this.params, initInputs);
      } catch (e) {
        console.error(`[WebAudioExecutor:${this.id}] Error executing init body:`, e);
      }
    }

    // Ensure analyser/dataArray/update exist
    if (!this.audioNode || !this.audioNode.analyser) {
      const analyser = this.audioCtx.createAnalyser();
      analyser.fftSize = this.params.fftSize || 256;
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

    this.dataWidth = this.audioNode.dataArray.length;
    this.dataHeight = 1;

    this.hub.allocateDataTexture(this.id, this.zLayer, this.dataWidth, this.dataHeight, {
      internalFormat: this.hub.gl.R8,
      format: this.hub.gl.RED,
      type: this.hub.gl.UNSIGNED_BYTE,
      visible: false,
    });

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

      if (/\b(const|let|var)\s+audioCtx\b/.test(t)) continue;
      if (/\b(const|let|var)\s+params\b/.test(t)) continue;

      // Avoid browser-only globals at init-time
      if (/\bwindow\b|\bdocument\b|\bnavigator\b/.test(t)) continue;

      out.push(line);
    }
    return out.join('\n').trim();
  }

  async execute(_textures, time, data) {
    if (!this.initialized || !this.audioNode) return;

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
      const tmp = new Uint8Array(arr.length);
      for (let i = 0; i < arr.length; i++) tmp[i] = Math.round(clamp01(arr[i]) * 255);
      bytes = tmp;
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

    let avg = 0;
    for (let i = 0; i < bytes.length; i++) avg += bytes[i];
    avg = bytes.length ? avg / (bytes.length * 255) : 0;

    this.hub.setNodeData(this.id, { type: 'audio', time, avg, bins: bytes.length });
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
    }

    this.nodeInstance = null;
    if (factory) {
      try {
        const initInputs = { textures: {}, data: {} };
        this.nodeInstance = factory(this.params, initInputs);
      } catch (e) {
        console.error(`[EventExecutor:${this.id}] Error executing init body:`, e);
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

    const bytes = new Uint8Array(4);
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

    // P5.js nodes for tracking (face, hand, body) - use JSModule executor
    case 'p5':
    case 'p5js':
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
