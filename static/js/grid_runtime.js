// GridRuntime: executes a Phase-2 project JSON as a layered (Z) node graph.
// Implements the "Universal Tensor Node" philosophy:
//   - every node publishes a texture output (render or data) into TextureHub
//   - every node can read the previous-layer composite as its main input (prev_layer_texture / u_input0)
//   - every node can read any upstream node output as a side input texture (u_<nodeId> in GLSL)
//   - JS engines also receive a JSON "data dictionary" for multimodal control (mouse/audio/event state)

// (dependencies loaded via separate script tags)
// (dependencies loaded via separate script tags)

class GridRuntime {
  constructor(canvas) {
    this.canvas = canvas;
    this.gl = canvas.getContext('webgl2', { alpha: true, antialias: true });
    if (!this.gl) throw new Error('WebGL2 not supported');

    this.hub = new TextureHub(this.gl, canvas.width, canvas.height);

    this.nodes = []; // { id, z, engine, executor }
    this.inputMap = new Map(); // nodeId -> [inputNodeId,...]

    this.running = false;
    this.lastTime = 0;
    this.frame = 0;

    // Basic pointer state (multimodal input for JS nodes)
    this.pointer = { x: 0.5, y: 0.5, down: false };
    this._installPointerHandlers();

    // Runtime error tracking
    this.errors = [];
    this.errorCache = {}; // Dedup cache for error reporting
    this.sessionId = null;
    this.nodeSpecs = new Map(); // nodeId -> original spec for error context
    this.failedNodes = new Set(); // nodes that failed to init

    // Register shader error callback for RuntimeInspector
    this.hub.onShaderError = (nodeId, errorLog) => {
      this._onShaderError(nodeId, errorLog);
    };
  }

  _onShaderError(nodeId, errorLog) {
    // Prevent infinite loops (dedup same error)
    if (this.errorCache[nodeId] === errorLog) return;
    this.errorCache[nodeId] = errorLog;

    console.error(`[GridRuntime] Shader error for ${nodeId}:`, errorLog);

    // Report to Python RuntimeInspector
    const spec = this.nodeSpecs.get(nodeId);
    if (!spec) return;

    fetch('/api/runtime-errors', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: this.sessionId,
        errors: [{
          node_id: nodeId,
          engine: spec.engine || 'glsl',
          error_message: errorLog,
          code_snippet: spec.code_snippet || '',
          parameters: spec.parameters || {},
          input_nodes: spec.input_nodes || []
        }]
      })
    })
    .then(res => res.json())
    .then(data => {
      if (data.fixes && data.fixes.length > 0) {
        console.log(`[GridRuntime] Received hotfix for ${nodeId}`);
        this.hotReloadNode(nodeId, data.fixes[0].fixed_code, data.fixes[0].fixed_parameters);
      }
    })
    .catch(e => console.warn('[GridRuntime] Error reporting failed:', e));
  }

  _installPointerHandlers() {
    const canvas = this.canvas;

    const toNorm = (evt) => {
      const rect = canvas.getBoundingClientRect();
      const x = (evt.clientX - rect.left) / rect.width;
      const y = 1.0 - (evt.clientY - rect.top) / rect.height;
      return { x: Math.max(0, Math.min(1, x)), y: Math.max(0, Math.min(1, y)) };
    };

    canvas.addEventListener('pointermove', (e) => {
      const p = toNorm(e);
      this.pointer.x = p.x;
      this.pointer.y = p.y;
    });

    canvas.addEventListener('pointerdown', (e) => {
      canvas.setPointerCapture?.(e.pointerId);
      this.pointer.down = true;
      const p = toNorm(e);
      this.pointer.x = p.x;
      this.pointer.y = p.y;
    });

    canvas.addEventListener('pointerup', (e) => {
      this.pointer.down = false;
      canvas.releasePointerCapture?.(e.pointerId);
    });

    canvas.addEventListener('pointerleave', () => {
      this.pointer.down = false;
    });
  }

  async loadProject(projectJson) {
    this.stop();
    this.clear();

    // Reset error tracking
    this.errors = [];
    this.nodeSpecs = new Map();
    this.failedNodes = new Set();
    this.sessionId = projectJson.phase2_meta?.session_id || null;

    // Build input map from connections (authoritative)
    this.inputMap = new Map();
    for (const conn of (projectJson.connections || [])) {
      const to = conn.to_node;
      const from = conn.from_node;
      if (!to || !from) continue;
      if (!this.inputMap.has(to)) this.inputMap.set(to, []);
      this.inputMap.get(to).push(from);
    }

    // Init executors with error capture
    this.nodes = [];
    for (const nodeSpec of (projectJson.nodes || [])) {
      // Skip nodes that failed mason validation — no usable code
      if (nodeSpec.mason_approved === false) {
        console.warn(`[GridRuntime] Skipping ${nodeSpec.id}: mason_approved=false`);
        this.failedNodes.add(nodeSpec.id);
        continue;
      }

      const z = (nodeSpec.grid_position && nodeSpec.grid_position[2]) ? nodeSpec.grid_position[2] : 0;
      const inputs = this.inputMap.get(nodeSpec.id) || nodeSpec.input_nodes || [];

      const execSpec = {
        id: nodeSpec.id,
        engine: nodeSpec.engine,
        code_snippet: nodeSpec.code_snippet,
        parameters: nodeSpec.parameters || {},
        z_layer: z,
        input_nodes: inputs,
      };

      // Store spec for error context
      this.nodeSpecs.set(nodeSpec.id, {
        ...nodeSpec,
        execSpec
      });

      try {
        const executor = createExecutor(execSpec, this.hub);
        await executor.init();

        this.nodes.push({
          id: nodeSpec.id,
          z,
          engine: nodeSpec.engine,
          executor,
        });
      } catch (e) {
        // Capture runtime error with full context
        const errorInfo = {
          node_id: nodeSpec.id,
          category: nodeSpec.category || 'unknown',
          engine: nodeSpec.engine,
          error_message: e.message,
          code_snippet: nodeSpec.code_snippet,
          parameters: nodeSpec.parameters || {},
          input_nodes: inputs,
          timestamp: new Date().toISOString()
        };
        this.errors.push(errorInfo);
        this.failedNodes.add(nodeSpec.id);
        console.error(`[GridRuntime] Node ${nodeSpec.id} failed to init:`, e.message);

        // Create passthrough fallback so the node isn't a transparent hole
        // Shows prev layer with red tint to indicate error
        try {
          const fallbackSpec = {
            id: nodeSpec.id,
            engine: 'glsl',
            code_snippet: `void main() {
              vec2 uv = v_uv;
              vec4 prev = texture(u_input0, uv);
              fragColor = vec4(mix(prev.rgb, vec3(0.8, 0.1, 0.1), 0.3), max(prev.a, 0.5));
            }`,
            parameters: {},
            z_layer: z,
            input_nodes: inputs,
          };
          const fallback = createExecutor(fallbackSpec, this.hub);
          await fallback.init();
          this.nodes.push({ id: nodeSpec.id, z, engine: 'glsl', executor: fallback });
          console.warn(`[GridRuntime] Node ${nodeSpec.id}: using passthrough fallback (will auto-fix)`);
        } catch (_) { /* truly broken, leave transparent */ }
      }
    }

    // Sort nodes by z, then stable by insertion order
    this.nodes.sort((a, b) => a.z - b.z);

    // Pre-create composites for all layers up to maxZ for continuous propagation
    const maxZ = this.getMaxZ();
    for (let z = 0; z <= maxZ; z++) {
      this.hub.publishLayerComposite(z);
    }

    console.log(`[GridRuntime] Loaded ${this.nodes.length} nodes, ${this.failedNodes.size} failed, maxZ=${maxZ}`);

    // Report errors to backend for LLM inspection
    if (this.errors.length > 0) {
      await this.reportErrors();
    }
  }

  getMaxZ() {
    let maxZ = 0;
    for (const n of this.nodes) maxZ = Math.max(maxZ, n.z);
    return maxZ;
  }

  start() {
    if (this.running) return;
    this.running = true;
    this.lastTime = performance.now() * 0.001;
    this.frame = 0;
    requestAnimationFrame(this._tick);
  }

  stop() {
    this.running = false;
  }

  clear() {
        console.log("[GridRuntime] Clearing project state...");
        
        // 1. Clear node runtimes
        this.nodes = {};
        this.connections = [];

        // 2. THE FIX: Only create TextureHub if it doesn't exist
        if (!this.textureHub) {
            console.log("[GridRuntime] Initializing TextureHub...");
            this.textureHub = new TextureHub(this.gl);
        } else {
            // If it already exists, just flush the old textures!
            this.textureHub.clearAll();
        }

        // 3. Clear canvas visually
        this.gl.clearColor(0.0, 0.0, 0.0, 1.0);
        this.gl.clear(this.gl.COLOR_BUFFER_BIT | this.gl.DEPTH_BUFFER_BIT);
    }

  resize(width, height) {
    this.canvas.width = width;
    this.canvas.height = height;
    this.gl.viewport(0, 0, width, height);

    // Replace hub (simplest correct approach)
    const oldHub = this.hub;
    this.hub = new TextureHub(this.gl, width, height);

    // NOTE: executors allocated textures in old hub; in a full impl we'd re-init.
    // For now, user should reload project after resize.
    oldHub.destroy();
  }

  // =========================================================================
  // Real-time Parameter Updates
  // =========================================================================

  setParam(nodeId, key, value) {
    // Find the node by ID
    const nodeEntry = this.nodes.find(n => n.id === nodeId);
    if (!nodeEntry) {
      console.warn(`[GridRuntime] setParam: node ${nodeId} not found`);
      return false;
    }

    // Update the executor's params object directly
    const executor = nodeEntry.executor;
    if (executor && executor.params) {
      executor.params[key] = value;
      // Also update the spec for consistency
      const spec = this.nodeSpecs.get(nodeId);
      if (spec && spec.execSpec && spec.execSpec.parameters) {
        spec.execSpec.parameters[key] = value;
      }
      return true;
    }

    console.warn(`[GridRuntime] setParam: executor for ${nodeId} has no params`);
    return false;
  }

  getParam(nodeId, key) {
    const nodeEntry = this.nodes.find(n => n.id === nodeId);
    if (!nodeEntry || !nodeEntry.executor || !nodeEntry.executor.params) {
      return undefined;
    }
    return nodeEntry.executor.params[key];
  }

  getAllParams(nodeId) {
    const nodeEntry = this.nodes.find(n => n.id === nodeId);
    if (!nodeEntry || !nodeEntry.executor) {
      return {};
    }
    return { ...nodeEntry.executor.params };
  }

  // =========================================================================
  // Node Enable/Disable
  // =========================================================================

  setNodeEnabled(nodeId, enabled) {
    const nodeEntry = this.nodes.find(n => n.id === nodeId);
    if (!nodeEntry) {
      console.warn(`[GridRuntime] setNodeEnabled: node ${nodeId} not found`);
      return false;
    }

    nodeEntry.enabled = enabled;

    // Update spec for consistency
    const spec = this.nodeSpecs.get(nodeId);
    if (spec) {
      spec.enabled = enabled;
    }

    console.log(`[GridRuntime] Node ${nodeId} ${enabled ? 'enabled' : 'disabled'}`);
    return true;
  }

  isNodeEnabled(nodeId) {
    const nodeEntry = this.nodes.find(n => n.id === nodeId);
    if (!nodeEntry) return true; // default to enabled
    return nodeEntry.enabled !== false;
  }

  // =========================================================================
  // Runtime Error Reporting & Hot-Reload
  // =========================================================================

  async reportErrors() {
    if (this.errors.length === 0) return;

    try {
      const response = await fetch('/api/runtime-errors', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: this.sessionId,
          errors: this.errors
        })
      });

      if (response.ok) {
        const result = await response.json();
        console.log(`[GridRuntime] Reported ${this.errors.length} errors to backend`);

        // If backend offers fixes, apply them
        if (result.fixes && result.fixes.length > 0) {
          console.log(`[GridRuntime] Received ${result.fixes.length} fixes from inspector`);
          for (const fix of result.fixes) {
            await this.hotReloadNode(fix.node_id, fix.fixed_code, fix.fixed_parameters);
          }
        }
      }
    } catch (e) {
      console.warn('[GridRuntime] Failed to report errors:', e);
    }
  }

  async requestFix(nodeId) {
    // Request LLM fix for a specific failed node
    const errorInfo = this.errors.find(e => e.node_id === nodeId);
    if (!errorInfo) {
      console.warn(`[GridRuntime] No error info for node ${nodeId}`);
      return null;
    }

    try {
      const response = await fetch('/api/fix-node', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: this.sessionId,
          error: errorInfo
        })
      });

      if (response.ok) {
        const result = await response.json();
        if (result.success && result.fixed_code) {
          console.log(`[GridRuntime] Received fix for ${nodeId}`);
          await this.hotReloadNode(nodeId, result.fixed_code, result.fixed_parameters);
          return result;
        }
      }
    } catch (e) {
      console.warn(`[GridRuntime] Failed to request fix for ${nodeId}:`, e);
    }
    return null;
  }

  async hotReloadNode(nodeId, fixedCode, fixedParameters) {
    // Hot-reload a node with fixed code/parameters
    const spec = this.nodeSpecs.get(nodeId);
    if (!spec) {
      console.warn(`[GridRuntime] No spec found for node ${nodeId}`);
      return false;
    }

    // Remove failed node from tracking
    this.failedNodes.delete(nodeId);
    this.errors = this.errors.filter(e => e.node_id !== nodeId);

    // Destroy old executor if it exists
    const existingIdx = this.nodes.findIndex(n => n.id === nodeId);
    if (existingIdx >= 0) {
      this.nodes[existingIdx].executor.destroy?.();
      this.nodes.splice(existingIdx, 1);
    }

    // Create new executor with fixed code/params
    const z = spec.execSpec.z_layer;
    const inputs = spec.execSpec.input_nodes;

    const newExecSpec = {
      id: nodeId,
      engine: spec.engine,
      code_snippet: fixedCode || spec.execSpec.code_snippet,
      parameters: fixedParameters || spec.execSpec.parameters,
      z_layer: z,
      input_nodes: inputs,
    };

    try {
      const executor = createExecutor(newExecSpec, this.hub);
      await executor.init();

      this.nodes.push({
        id: nodeId,
        z,
        engine: spec.engine,
        executor,
      });

      // Re-sort
      this.nodes.sort((a, b) => a.z - b.z);

      console.log(`[GridRuntime] Hot-reloaded node ${nodeId}`);
      return true;
    } catch (e) {
      console.error(`[GridRuntime] Hot-reload failed for ${nodeId}:`, e.message);
      this.failedNodes.add(nodeId);
      return false;
    }
  }

  getErrors() {
    return this.errors;
  }

  getFailedNodes() {
    return Array.from(this.failedNodes);
  }

  _tick = (ms) => {
    if (!this.running) return;

    const time = ms * 0.001;
    const dt = Math.max(0, time - this.lastTime);
    this.lastTime = time;

    // Never stop the render loop due to per-node errors
    this._renderFrame(time, dt).catch((e) => {
      console.error('[GridRuntime] render frame error (continuing):', e);
    });

    requestAnimationFrame(this._tick);
  };

  async _renderFrame(time, dt) {
    const maxZ = this.getMaxZ();

    const frameData = {
      time,
      dt,
      frame: this.frame++,
      mouse: [this.pointer.x, this.pointer.y],
      mouseDown: this.pointer.down ? 1 : 0,
      resolution: [this.hub.width, this.hub.height],
    };

    // Execute layer by layer
    for (let z = 0; z <= maxZ; z++) {
      const nodesAtZ = this.nodes.filter(n => n.z === z);

      for (const node of nodesAtZ) {
        // Skip disabled nodes
        if (node.enabled === false) {
          continue;
        }

        try {
          const inputNodeIds = this.inputMap.get(node.id) || [];

          // Textures — Lego 1:1 snap: primary = the connected source's texture
          const textures = this.hub.resolveInputs(z - 1, inputNodeIds);

          // Rich tensor data: build per-input metadata so target node knows
          // what it's receiving and can auto-convert
          const inputData = {};
          for (const inId of inputNodeIds) {
            const srcNode = this.nodes.find(n => n.id === inId);
            const srcEntry = this.hub.getNodeEntry(inId);
            const srcData = this.hub.getNodeData(inId);
            const srcSpec = this.nodeSpecs.get(inId);

            inputData[inId] = {
              engine: srcNode?.engine || 'unknown',
              kind: srcEntry?.kind || 'render',     // 'render' or 'data'
              category: srcSpec?.category || '',
              data: srcData || null,                 // structured data from source
              params: srcNode?.executor?.params || {},// source node's live params
              width: srcEntry?.width || this.hub.width,
              height: srcEntry?.height || this.hub.height,
            };
          }

          const data = {
            ...frameData,
            node: { id: node.id, engine: node.engine, z },
            inputs: inputData,
          };

          const maybe = node.executor.execute(textures, time, data);
          if (maybe && typeof maybe.then === 'function') {
            await maybe;
          }
        } catch (e) {
          // Log once per node, don't kill the render loop
          if (!node._errorLogged) {
            console.warn(`[GridRuntime] Node ${node.id} execute error (muting):`, e.message);
            node._errorLogged = true;
          }
        }
      }

      // Composite z-layer (continuous propagation even if layer empty)
      this.hub.publishLayerComposite(z);
    }

    // Present final composite
    this.hub.blitToCanvas(maxZ);
  }
}
