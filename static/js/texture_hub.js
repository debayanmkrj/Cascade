// TextureHub: Universal texture bus for the polyglot volumetric grid.
// - Render nodes write RGBA render targets (visual textures).
// - Data nodes (audio/control) write small "data textures" (e.g., 1xN or 1x1) that can still be sampled by GLSL.
// - Layer composites are always produced (even for empty layers) so Z propagation is continuous.

class TextureHub {
  constructor(gl, width, height) {
    this.gl = gl;
    this.width = width;
    this.height = height;

    this.nodeTextures = new Map();
    this.layerComposites = new Map();
    this.nodeData = new Map();

    this.blitProgram = this._createBlitProgram();
  }

  setNodeData(nodeId, data) {
    this.nodeData.set(nodeId, data);
  }

  getNodeData(nodeId) {
    return this.nodeData.get(nodeId) ?? null;
  }

  allocate(nodeId, zLayer, opts = {}) {
    const kind = opts.kind || 'render';

    if (kind === 'data') {
      const w = opts.width ?? 1;
      const h = opts.height ?? 1;
      return this.allocateDataTexture(nodeId, zLayer, w, h, opts);
    }

    const w = opts.width ?? this.width;
    const h = opts.height ?? this.height;
    const internalFormat = opts.internalFormat ?? this.gl.RGBA8;
    const format = opts.format ?? this.gl.RGBA;
    const type = opts.type ?? this.gl.UNSIGNED_BYTE;

    const texture = this._createTexture(w, h, internalFormat, format, type);
    const framebuffer = this._createFramebuffer(texture);

    this.nodeTextures.set(nodeId, {
      texture, framebuffer, z_layer: zLayer, kind: 'render',
      width: w, height: h, internalFormat, format, type,
      visible: opts.visible ?? true,
    });

    return texture;
  }

  allocateDataTexture(nodeId, zLayer, width, height, opts = {}) {
    const gl = this.gl;
    const internalFormat = opts.internalFormat ?? gl.R8;
    const format = opts.format ?? gl.RED;
    const type = opts.type ?? gl.UNSIGNED_BYTE;

    const texture = this._createTexture(width, height, internalFormat, format, type);

    this.nodeTextures.set(nodeId, {
      texture, framebuffer: null, z_layer: zLayer, kind: 'data',
      width, height, internalFormat, format, type,
      visible: opts.visible ?? false,
    });

    return texture;
  }

  updateDataTexture(nodeId, zLayer, width, height, data, opts = {}) {
    const gl = this.gl;
    let entry = this.nodeTextures.get(nodeId);
    if (!entry || entry.kind !== 'data' || entry.width !== width || entry.height !== height) {
      this.allocateDataTexture(nodeId, zLayer, width, height, opts);
      entry = this.nodeTextures.get(nodeId);
    }
    gl.bindTexture(gl.TEXTURE_2D, entry.texture);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, width, height, entry.format, entry.type, data);
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  getNodeTexture(nodeId) {
    const entry = this.nodeTextures.get(nodeId);
    return entry ? entry.texture : null;
  }

  _ensureLayerComposite(zLayer) {
    if (this.layerComposites.has(zLayer)) return this.layerComposites.get(zLayer);
    const gl = this.gl;
    const texture = this._createTexture(this.width, this.height, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE);
    const framebuffer = this._createFramebuffer(texture);
    const entry = { texture, framebuffer, width: this.width, height: this.height };
    this.layerComposites.set(zLayer, entry);
    return entry;
  }

  publishLayerComposite(zLayer) {
    const gl = this.gl;
    const composite = this._ensureLayerComposite(zLayer);
    gl.bindFramebuffer(gl.FRAMEBUFFER, composite.framebuffer);
    gl.viewport(0, 0, composite.width, composite.height);

    if (zLayer === 0) {
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
    } else {
      const prev = this.layerComposites.get(zLayer - 1);
      if (prev) {
        gl.disable(gl.BLEND);
        this._blit(prev.texture);
      } else {
        gl.clearColor(0, 0, 0, 0);
        gl.clear(gl.COLOR_BUFFER_BIT);
      }
    }

    const nodesAtZ = Array.from(this.nodeTextures.entries())
      .filter(([_id, e]) => e.z_layer === zLayer && e.kind === 'render' && (e.visible ?? true))
      .map(([_id, e]) => e);

    if (nodesAtZ.length > 0) {
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      for (const nodeEntry of nodesAtZ) {
        this._blit(nodeEntry.texture);
      }
      gl.disable(gl.BLEND);
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  /**
   * Resolve input textures for a node using Lego 1:1 snap.
   * prev_layer_texture = the connected source node's render output (NOT broadcast composite).
   * Falls back to layer composite only if no 1:1 connection exists.
   */
  resolveInputs(prevZ, inputNodeIds) {
    const inputs = {};

    // Lego 1:1 snap: use the connected node's texture as primary input
    if (inputNodeIds && inputNodeIds.length > 0) {
      const srcId = inputNodeIds[0];  // 1:1 = at most 1 source
      const tex = this.getNodeTexture(srcId);
      if (tex) {
        inputs.prev_layer_texture = tex;
        inputs[srcId] = tex;
      }
      // Include any additional connected nodes (future: multi-input)
      for (let i = 1; i < inputNodeIds.length; i++) {
        const id = inputNodeIds[i];
        const t = this.getNodeTexture(id);
        if (t) inputs[id] = t;
      }
    }

    // Fallback: no 1:1 connection → use layer composite (e.g. Z=0 generators)
    if (!inputs.prev_layer_texture && prevZ >= 0) {
      const prevComposite = this.layerComposites.get(prevZ);
      if (prevComposite) inputs.prev_layer_texture = prevComposite.texture;
    }

    return inputs;
  }

  /**
   * Get a node's full texture entry (for rich tensor auto-conversion).
   * Returns { texture, framebuffer, z_layer, kind, width, height, ... } or null.
   */
  getNodeEntry(nodeId) {
    return this.nodeTextures.get(nodeId) ?? null;
  }

  blitToCanvas(zLayer) {
    const gl = this.gl;
    let entry = this.layerComposites.get(zLayer);
    if (!entry) {
      for (let z = zLayer - 1; z >= 0; z--) {
        entry = this.layerComposites.get(z);
        if (entry) break;
      }
    }
    if (!entry) return;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.width, this.height);
    gl.disable(gl.BLEND);
    this._blit(entry.texture);
  }

  _createTexture(width, height, internalFormat, format, type) {
    const gl = this.gl;
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, width, height, 0, format, type, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.bindTexture(gl.TEXTURE_2D, null);
    return tex;
  }

  _createFramebuffer(texture) {
    const gl = this.gl;
    const fb = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return fb;
  }

  _createBlitProgram() {
    const gl = this.gl;
    const vs = `#version 300 es
      in vec2 a_position; out vec2 v_uv;
      void main() { v_uv = a_position * 0.5 + 0.5; gl_Position = vec4(a_position, 0.0, 1.0); }`;
    const fs = `#version 300 es
      precision highp float; in vec2 v_uv; uniform sampler2D u_texture; out vec4 fragColor;
      void main() { fragColor = texture(u_texture, v_uv); }`;

    const program = this._createProgram(vs, fs);
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    const posBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, posBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, -1,1, 1,-1, 1,1]), gl.STATIC_DRAW);
    const posLoc = gl.getAttribLocation(program, 'a_position');
    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    return { program, vao };
  }

  _blit(texture) {
    const gl = this.gl;
    const { program, vao } = this.blitProgram;
    gl.useProgram(program);
    gl.bindVertexArray(vao);
    const loc = gl.getUniformLocation(program, 'u_texture');
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.uniform1i(loc, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    gl.bindVertexArray(null);
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  _createProgram(vsSource, fsSource) {
    const gl = this.gl;
    const vs = this._compileShader(gl.VERTEX_SHADER, vsSource);
    const fs = this._compileShader(gl.FRAGMENT_SHADER, fsSource);
    const prog = gl.createProgram();
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      const info = gl.getProgramInfoLog(prog);
      gl.deleteProgram(prog);
      throw new Error('TextureHub: program link failed: ' + info);
    }
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    return prog;
  }

  _compileShader(type, source) {
    const gl = this.gl;
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const info = gl.getShaderInfoLog(shader);
      gl.deleteShader(shader);
      throw new Error('TextureHub: shader compile failed: ' + info);
    }
    return shader;
  }

  destroy() {
    const gl = this.gl;
    for (const entry of this.nodeTextures.values()) {
      if (entry.framebuffer) gl.deleteFramebuffer(entry.framebuffer);
      if (entry.texture) gl.deleteTexture(entry.texture);
    }
    this.nodeTextures.clear();
    for (const entry of this.layerComposites.values()) {
      if (entry.framebuffer) gl.deleteFramebuffer(entry.framebuffer);
      if (entry.texture) gl.deleteTexture(entry.texture);
    }
    this.layerComposites.clear();
    this.nodeData.clear();
    if (this.blitProgram) {
      gl.deleteProgram(this.blitProgram.program);
      gl.deleteVertexArray(this.blitProgram.vao);
      this.blitProgram = null;
    }
  }
  clearAll() {
        // Delete all stored WebGL textures to free GPU memory
        if (this.textures) {
            for (const key in this.textures) {
                if (this.textures[key]) {
                    this.gl.deleteTexture(this.textures[key]);
                }
            }
        }
        
        // Delete all framebuffers
        if (this.framebuffers) {
            for (const key in this.framebuffers) {
                if (this.framebuffers[key]) {
                    this.gl.deleteFramebuffer(this.framebuffers[key]);
                }
            }
        }

        // Reset the dictionaries, but KEEP the compiled blit programs!
        this.textures = {};
        this.framebuffers = {};
        
        console.log("[TextureHub] Cleared textures and framebuffers (Shaders preserved).");
  } 
}

