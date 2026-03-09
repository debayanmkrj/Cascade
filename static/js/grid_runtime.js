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
    this._nodeById = new Map(); // nodeId -> node (fast lookup)
    this._nodesByZ = new Map(); // z -> [node,...] (fast per-layer lookup)

    this.running = false;
    this._rendering = false; // guard: prevent concurrent _renderFrame calls
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

    // Callback fired after a successful hot-reload so the frontend can patch
    // currentProject and persist the fix to disk. Set by the page after init.
    // Signature: onNodeFixed(nodeId, fixedCode, fixedParameters)
    this.onNodeFixed = null;

    // Register shader error callback for RuntimeInspector
    this.hub.onShaderError = (nodeId, errorLog) => {
      this._onShaderError(nodeId, errorLog);
    };

    // ── Universal error capture ─────────────────────────────────────────────
    // Intercept ALL console.error calls + uncaught errors/rejections.
    // Any error that can be attributed to a node ID is forwarded to Mason for
    // fixing — no hard-coded pattern filters; Mason decides how to fix.
    const _origConsoleError = console.error.bind(console);
    console.error = (...args) => {
      _origConsoleError(...args);
      if (!this.gl.isContextLost()) this._routeConsoleError(args);
    };
    const _windowErrHandler = (ev) => {
      if (this.gl.isContextLost()) return;
      const msg = ev.message
        || ev.reason?.message
        || String(ev.reason || ev.error || '');
      if (msg) this._routeConsoleError([msg]);
    };
    window.addEventListener('error', _windowErrHandler);
    window.addEventListener('unhandledrejection', _windowErrHandler);

    // WebGL context loss/restore — pause render on loss, full reinit on restore
    this._lastProjectJson = null; // stored by loadProject for restore
    this._contextLostAt = 0;      // timestamp of last context loss (for cooldown)
    canvas.addEventListener('webglcontextlost', (e) => {
      e.preventDefault(); // required to allow restore
      console.warn('[GridRuntime] WebGL context lost — pausing render loop');
      this.running = false;
      this._rendering = false;
      // Stamp the loss time so _onShaderError and hotReloadNode stay silent
      // for the post-loss cooldown window — prevents Mason compile storm mid-crash.
      this._contextLostAt = performance.now();
      // Destroy executors NOW while context IS lost:
      // - GL deletes are safe no-ops (context is gone, nothing to corrupt)
      // - Non-GL cleanup runs correctly: stops webcam streams, ml5 detector loops,
      //   p5 sketches — preventing stale async loops from triggering immediate re-loss
      //   on the restored context.
      for (const node of this.nodes) {
        try { node.executor?.destroy?.(); } catch (_) {}
      }
      this.nodes = [];
      this._nodeById = new Map();
      this._nodesByZ = new Map();
      console.warn('[GridRuntime] All executors destroyed on context loss (non-GL cleanup complete)');
    }, false);

    canvas.addEventListener('webglcontextrestored', async () => {
      console.log('[GridRuntime] WebGL context restored — reinitialising');
      try {
        // nodes array is already empty (cleared in webglcontextlost handler).
        // loadProject → clear() will find nothing to destroy, so no INVALID_OPERATION
        // from stale GL objects of the previous session.
        this.hub = new TextureHub(this.gl, canvas.width, canvas.height);
        this.hub.onShaderError = (nodeId, errorLog) => { this._onShaderError(nodeId, errorLog); };
        if (this._lastProjectJson) {
          await this.loadProject(this._lastProjectJson);
          this.start();
        }
      } catch (err) {
        console.error('[GridRuntime] Failed to reinit after context restore:', err);
      }
    }, false);
  }

  _onShaderError(nodeId, errorLog) {
    // Bail if context is lost — GL no-ops produce false compile failures.
    if (this.gl.isContextLost()) return;
    // Bail during post-loss cooldown (3 s) — GPU is still stabilising.
    if (performance.now() - (this._contextLostAt || 0) < 3000) return;

    // Hard cap: max 5 fix attempts per node before giving up
    if (!this._fixAttempts) this._fixAttempts = {};
    const attempts = this._fixAttempts[nodeId] || 0;
    if (attempts >= 5) {
      console.warn(`[GridRuntime] Max fix attempts (5) reached for ${nodeId}, giving up`);
      return;
    }
    this._fixAttempts[nodeId] = attempts + 1;

    // Prevent identical error dedup
    if (this.errorCache[nodeId] === errorLog) return;
    this.errorCache[nodeId] = errorLog;

    console.error(`[GridRuntime] Shader error for ${nodeId} (attempt ${attempts + 1}/2):`, errorLog);

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

  _routeConsoleError(args) {
    // Build a single string from all logged args
    const msg = args.map(a => {
      if (typeof a === 'string') return a;
      if (a instanceof Error) return a.message + (a.stack ? '\n' + a.stack : '');
      try { return String(a); } catch (_) { return ''; }
    }).join(' ');

    // Match patterns like [JSModuleExecutor:node_6_n6] or [Canvas2DExecutor:node_3]
    const m = msg.match(/\[\w*Executor[:\s]+(node_\d+_\w+)\]/);
    if (!m) return;
    const nodeId = m[1];
    if (!this.nodeSpecs.has(nodeId)) return;

    // Strip executor prefix to get a clean error message for Mason
    const errorText = msg.replace(/\[[\w\s:]+\]\s*/g, '').trim();
    if (!errorText) return;

    this._onShaderError(nodeId, errorText);
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
    this._lastProjectJson = projectJson; // saved for webglcontextrestored replay

    // If the WebGL context is lost, all GL calls are no-ops returning null/false.
    // Attempting to init executors on a lost context produces misleading errors
    // (valid shaders appear to fail, null textures are returned, etc.) and can
    // trigger spurious Mason auto-fix loops. Bail here — webglcontextrestored
    // will replay the load automatically once the GPU recovers.
    if (this.gl.isContextLost()) {
      console.warn('[GridRuntime] loadProject deferred — WebGL context is lost, will replay on restore');
      return;
    }

    this.stop();
    // Yield one microtask after stop() so any in-flight _renderFrame promise can settle
    // before clear() destroys the GL objects it may be referencing.
    await Promise.resolve();
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

      // Yield one macrotask between each node init. Shader compilation is
      // synchronous on the GPU — back-to-back compiles for all nodes spike GPU
      // time and can trigger Windows TDR. Yielding spreads the work over
      // multiple event-loop ticks and lets the browser breathe between nodes.
      await new Promise(r => setTimeout(r, 0));

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
        // If context was lost mid-init, the error is an infrastructure failure,
        // not a code bug. Don't report it to Mason or create a fallback executor
        // (which would also fail). The webglcontextrestored handler will replay.
        if (this.gl.isContextLost()) {
          console.warn(`[GridRuntime] Node ${nodeSpec.id} init aborted — WebGL context lost`);
          break;
        }

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
        console.error(`[GridRuntime] Node ${nodeSpec.id} failed to init — Mason will fix:`, e.message);
      }
    }

    // Sort nodes by z, then stable by insertion order
    this.nodes.sort((a, b) => a.z - b.z);

    // Build fast-lookup caches (avoid O(N²) per-frame scans)
    this._rebuildNodeCaches();

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

  _rebuildNodeCaches() {
    this._nodeById = new Map(this.nodes.map(n => [n.id, n]));
    this._nodesByZ = new Map();
    for (const n of this.nodes) {
      if (!this._nodesByZ.has(n.z)) this._nodesByZ.set(n.z, []);
      this._nodesByZ.get(n.z).push(n);
    }
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

        // Wait for any in-flight async _renderFrame to finish before touching GPU resources.
        // _rendering is an async guard — force-clear it so the next frame check aborts cleanly.
        // The rAF callback checks this.running (already set false by stop()) before proceeding.
        this._rendering = false;

        // Give the browser one microtask to let any awaiting _renderFrame promise settle
        // before we start deleting GL objects it may still be referencing.
        // (This is synchronous clear; async teardown would require a promise chain.)

        // 1. Destroy executors first so they free GL resources before hub.clearAll()
        if (Array.isArray(this.nodes)) {
            for (const node of this.nodes) {
                try { node.executor?.destroy?.(); } catch (_) {}
            }
        }
        this.nodes = [];
        this.connections = [];
        this._nodeById = new Map();
        this._nodesByZ = new Map();

        // 2. Flush hub textures after executors are destroyed
        if (!this.hub) {
            console.log("[GridRuntime] Initializing TextureHub...");
            this.hub = new TextureHub(this.gl, this.canvas.width, this.canvas.height);
        } else {
            this.hub.clearAll();
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

  getExecutorForNode(id) {
    return this.nodes.find(n => n.id === id)?.executor ?? null;
  }

  // Returns all audio + tracking hub data keyed by nodeId — used to broadcast
  // audio/tracking to ALL downstream nodes regardless of connection topology.
  _buildGlobalData() {
    const gd = {};
    for (const n of this.nodes) {
      const d = this.hub.getNodeData(n.id);
      if (d && (d.type === 'audio' || d.trackingData)) gd[n.id] = d;
    }
    return gd;
  }

  // =========================================================================
  // Runtime Error Reporting & Hot-Reload
  // =========================================================================

  async reportErrors() {
    if (this.errors.length === 0) return;
    // Don't report to Mason while context is lost or in post-loss cooldown.
    if (this.gl.isContextLost()) return;
    if (performance.now() - (this._contextLostAt || 0) < 3000) return;

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
    // Bail if context is lost or in post-loss cooldown — compiling on a lost/
    // unstable context produces spurious errors and can trigger re-loss.
    if (this.gl.isContextLost()) {
      console.warn(`[GridRuntime] hotReloadNode deferred — context is lost`);
      return false;
    }
    if (performance.now() - (this._contextLostAt || 0) < 3000) {
      console.warn(`[GridRuntime] hotReloadNode deferred — post-loss cooldown active`);
      return false;
    }

    // Hot-reload a node with fixed code/parameters
    const spec = this.nodeSpecs.get(nodeId);
    if (!spec) {
      console.warn(`[GridRuntime] No spec found for node ${nodeId}`);
      return false;
    }

    // Remove failed node from tracking
    this.failedNodes.delete(nodeId);
    this.errors = this.errors.filter(e => e.node_id !== nodeId);

    // Destroy old executor and free its hub GPU resources before re-allocating.
    // Without hub.dealloc() here, every hot-reload leaks one full-res texture.
    const existingIdx = this.nodes.findIndex(n => n.id === nodeId);
    if (existingIdx >= 0) {
      try { this.nodes[existingIdx].executor.destroy?.(); } catch (_) {}
      this.nodes.splice(existingIdx, 1);
      this.hub.dealloc(nodeId);
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

      // Re-sort and rebuild caches
      this.nodes.sort((a, b) => a.z - b.z);
      this._rebuildNodeCaches();

      // Clear fix attempts counter on success
      if (this._fixAttempts) delete this._fixAttempts[nodeId];
      console.log(`[GridRuntime] Hot-reloaded node ${nodeId}`);

      // Notify frontend to patch currentProject and save to disk
      if (typeof this.onNodeFixed === 'function') {
        this.onNodeFixed(nodeId, newExecSpec.code_snippet, newExecSpec.parameters);
      }
      return true;
    } catch (e) {
      console.error(`[GridRuntime] Hot-reload failed for ${nodeId}:`, e.message);
      this.failedNodes.add(nodeId);

      // Re-queue the error with the latest (still-broken) code so Mason retries.
      // Node stays absent from this.nodes (renders transparent) until a fix succeeds.
      this.errors.push({
        node_id: nodeId,
        category: spec.category || 'unknown',
        engine: spec.engine,
        error_message: e.message,
        code_snippet: newExecSpec.code_snippet,
        parameters: newExecSpec.parameters,
        input_nodes: inputs,
        timestamp: new Date().toISOString()
      });
      // Fire another fix attempt — don't await so we don't block the render loop.
      this.reportErrors().catch(() => {});

      return false;
    }
  }

  removeNode(nodeId) {
    const idx = this.nodes.findIndex(n => n.id === nodeId);
    if (idx < 0) return false;

    const node = this.nodes[idx];
    const z = node.z;

    // Destroy executor (stops webcam streams, GL resources, etc.)
    try { node.executor?.destroy?.(); } catch (_) {}
    this.nodes.splice(idx, 1);

    // Deallocate hub GPU resources
    this.hub.dealloc(nodeId);

    // Clean up inputMap
    this.inputMap.delete(nodeId);
    for (const [toId, fromIds] of this.inputMap.entries()) {
      const filtered = fromIds.filter(id => id !== nodeId);
      this.inputMap.set(toId, filtered);
    }

    // Clean up tracking state
    this.nodeSpecs.delete(nodeId);
    this.failedNodes.delete(nodeId);
    this.errors = this.errors.filter(e => e.node_id !== nodeId);
    if (this._fixAttempts) delete this._fixAttempts[nodeId];
    delete this.errorCache[nodeId];

    // Force re-composite affected Z layers so ghost frame is cleared
    const maxZ = this.getMaxZ();
    for (let zz = z; zz <= maxZ; zz++) {
      this.hub.publishLayerComposite(zz);
    }
    this.hub.blitToCanvas(maxZ >= 0 ? maxZ : 0);

    return true;
  }

  // =========================================================================
  // Incremental Layout Update — NO executor destruction or creation.
  // Use instead of loadProject when only Z-layer positions or connections
  // change (moveNodeZ, handleDrop, setNodeInput).  Costs: a few blit ops to
  // republish layer composites.  No shader compilation, no texture allocation.
  // =========================================================================
  refreshLayout(projectJson) {
    if (!projectJson) return;
    this._lastProjectJson = projectJson;

    // 1. Rebuild inputMap from the updated connections array
    this.inputMap = new Map();
    for (const conn of (projectJson.connections || [])) {
      const to = conn.to_node, from = conn.from_node;
      if (!to || !from) continue;
      if (!this.inputMap.has(to)) this.inputMap.set(to, []);
      this.inputMap.get(to).push(from);
    }

    // 2. Update Z-layer for every node that exists in the runtime
    for (const nodeSpec of (projectJson.nodes || [])) {
      const z = (nodeSpec.grid_position && nodeSpec.grid_position[2]) || 0;
      const entry = this._nodeById.get(nodeSpec.id);
      if (!entry) continue;
      entry.z = z;
      if (entry.executor) entry.executor.zLayer = z;
      const hubEntry = this.hub.nodeTextures.get(nodeSpec.id);
      if (hubEntry) hubEntry.z_layer = z;
    }

    // 3. Re-sort and rebuild fast-lookup caches
    this.nodes.sort((a, b) => a.z - b.z);
    this._rebuildNodeCaches();

    // 4. Re-publish all layer composites (cheap: a few blit operations)
    if (!this.gl.isContextLost()) {
      const maxZ = this.getMaxZ();
      for (let z = 0; z <= maxZ; z++) this.hub.publishLayerComposite(z);
      this.hub.blitToCanvas(maxZ >= 0 ? maxZ : 0);
    }
  }

  // =========================================================================
  // Add a single node to the running runtime without touching existing nodes.
  // Use instead of loadProject when addPredefinedNode inserts one new node.
  // =========================================================================
  async addNode(nodeSpec) {
    if (this.gl.isContextLost()) {
      console.warn('[GridRuntime] addNode deferred — context is lost');
      return false;
    }

    const z = (nodeSpec.grid_position && nodeSpec.grid_position[2]) || 0;
    const inputs = this.inputMap.get(nodeSpec.id) || nodeSpec.input_nodes || [];

    const execSpec = {
      id: nodeSpec.id,
      engine: nodeSpec.engine,
      code_snippet: nodeSpec.code_snippet,
      parameters: nodeSpec.parameters || {},
      z_layer: z,
      input_nodes: inputs,
    };

    this.nodeSpecs.set(nodeSpec.id, { ...nodeSpec, execSpec });

    // Yield one macrotask so the browser can render before GL work starts
    await new Promise(r => setTimeout(r, 0));

    try {
      const executor = createExecutor(execSpec, this.hub);
      await executor.init();

      this.nodes.push({ id: nodeSpec.id, z, engine: nodeSpec.engine, executor });
      this.nodes.sort((a, b) => a.z - b.z);
      this._rebuildNodeCaches();

      if (!this.gl.isContextLost()) {
        const maxZ = this.getMaxZ();
        for (let zz = 0; zz <= maxZ; zz++) this.hub.publishLayerComposite(zz);
      }

      console.log(`[GridRuntime] addNode: ${nodeSpec.id} (${nodeSpec.engine}) at Z=${z}`);
      return true;
    } catch (e) {
      console.error(`[GridRuntime] addNode failed for ${nodeSpec.id}:`, e.message);
      this.failedNodes.add(nodeSpec.id);
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
    requestAnimationFrame(this._tick);

    // Stop rendering on context loss — handlers will restart when context is restored
    if (this.gl.isContextLost()) return;

    const time = ms * 0.001;
    const dt = time - this.lastTime;

    // Cap pipeline at 30fps
    if (dt < 0.033) return;
    this.lastTime = time;

    // Guard: skip frame if previous _renderFrame is still awaiting async nodes
    // Without this, async nodes cause exponential frame pileup → <1fps
    if (this._rendering) return;
    this._rendering = true;

    this._renderFrame(time, dt)
      .catch(e => console.error('[GridRuntime] render frame error (continuing):', e))
      .finally(() => { this._rendering = false; });
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
      const nodesAtZ = this._nodesByZ.get(z) || [];

      for (const node of nodesAtZ) {
        // Skip disabled nodes
        if (node.enabled === false) continue;

        // Adaptive throttle: slow nodes run every Nth frame instead of being disabled
        if (node._frameSkip > 1) {
          node._frameSkipCounter = (node._frameSkipCounter || 0) + 1;
          if (node._frameSkipCounter < node._frameSkip) continue;
          node._frameSkipCounter = 0;
        }

        try {
          const inputNodeIds = this.inputMap.get(node.id) || [];

          // Textures — Lego 1:1 snap: primary = the connected source's texture
          const textures = this.hub.resolveInputs(z - 1, inputNodeIds);

          // Rich tensor data: build per-input metadata so target node knows
          // what it's receiving and can auto-convert
          const inputData = {};
          for (const inId of inputNodeIds) {
            const srcNode = this._nodeById.get(inId);
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
            globalData: this._buildGlobalData(),
          };

          const t0 = performance.now();
          const maybe = node.executor.execute(textures, time, data);
          if (maybe && typeof maybe.then === 'function') {
            await maybe;
          }
          // Context may have been lost during async node execute — abort remaining frame.
          if (this.gl.isContextLost()) return;
          const elapsed = performance.now() - t0;

          // Adaptive throttle: slow nodes run every Nth frame to stay alive.
          // js_module (p5) nodes engage after 3 slow frames — no JIT/CDN startup spike
          // to ignore, and they're consistently slow. GLSL/canvas2d keep the 10-frame
          // window to avoid false-positives from one-time shader compilation spikes.
          // Recovery: any frame under 50ms immediately drops skip by one step.
          const slowGate = node.engine === 'js_module' ? 3 : 10;
          if (elapsed > 50) {
            node._slowFrames = (node._slowFrames || 0) + 1;
            if (node._slowFrames >= slowGate) {
              const prevSkip = node._frameSkip || 1;
              const newSkip = elapsed > 150 ? 4 : 2;
              if (newSkip !== prevSkip) {
                node._frameSkip = newSkip;
                node._frameSkipCounter = 0;
                console.warn(`[GridRuntime] Throttling ${node.id} (${node.engine}): ${elapsed.toFixed(0)}ms → running every ${newSkip} frames`);
              }
            }
          } else {
            // Fast again — step skip down by one level each fast frame for smooth recovery
            const cur = node._frameSkip || 1;
            if (cur > 1) {
              node._frameSkip = cur > 2 ? 2 : 1;
              node._frameSkipCounter = 0;
            }
            node._slowFrames = Math.max(0, (node._slowFrames || 0) - 2);
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
