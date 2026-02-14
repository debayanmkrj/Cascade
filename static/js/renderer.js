/**
 * WebGL Renderer for LowKCDR
 * Executes shader chains from node graph
 */

class WebGLRenderer {
    constructor(canvas) {
        this.canvas = canvas;
        this.gl = canvas.getContext('webgl2');

        if (!this.gl) {
            console.error('[Renderer] WebGL2 not supported');
            return;
        }

        this.isRunning = false;
        this.startTime = performance.now();
        this.currentInput = null;

        // Framebuffers for ping-pong
        this.framebuffers = [];
        this.textures = [];

        this.init();
    }

    init() {
        const gl = this.gl;

        // Set viewport
        this.resize();

        // Create framebuffers
        for (let i = 0; i < 2; i++) {
            const fb = gl.createFramebuffer();
            const tex = gl.createTexture();

            gl.bindTexture(gl.TEXTURE_2D, tex);
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, this.canvas.width, this.canvas.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

            gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
            gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);

            this.framebuffers.push(fb);
            this.textures.push(tex);
        }

        gl.bindFramebuffer(gl.FRAMEBUFFER, null);

        // Create fullscreen quad
        const quadVerts = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);
        this.quadBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, quadVerts, gl.STATIC_DRAW);

        // Compile default shader
        this.defaultProgram = this.compileProgram(this.defaultVertexShader, this.defaultFragmentShader);

        console.log('[Renderer] Initialized');
    }

    get defaultVertexShader() {
        return `#version 300 es
            in vec2 a_position;
            out vec2 v_uv;
            void main() {
                v_uv = a_position * 0.5 + 0.5;
                gl_Position = vec4(a_position, 0.0, 1.0);
            }
        `;
    }

    get defaultFragmentShader() {
        return `#version 300 es
            precision highp float;
            in vec2 v_uv;
            out vec4 fragColor;
            uniform float u_time;
            uniform vec2 u_resolution;

            // Simplex noise
            vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
            vec2 mod289(vec2 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
            vec3 permute(vec3 x) { return mod289(((x*34.0)+1.0)*x); }

            float snoise(vec2 v) {
                const vec4 C = vec4(0.211324865405187, 0.366025403784439, -0.577350269189626, 0.024390243902439);
                vec2 i = floor(v + dot(v, C.yy));
                vec2 x0 = v - i + dot(i, C.xx);
                vec2 i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
                vec4 x12 = x0.xyxy + C.xxzz;
                x12.xy -= i1;
                i = mod289(i);
                vec3 p = permute(permute(i.y + vec3(0.0, i1.y, 1.0)) + i.x + vec3(0.0, i1.x, 1.0));
                vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy), dot(x12.zw,x12.zw)), 0.0);
                m = m*m; m = m*m;
                vec3 x = 2.0 * fract(p * C.www) - 1.0;
                vec3 h = abs(x) - 0.5;
                vec3 ox = floor(x + 0.5);
                vec3 a0 = x - ox;
                m *= 1.79284291400159 - 0.85373472095314 * (a0*a0 + h*h);
                vec3 g;
                g.x = a0.x * x0.x + h.x * x0.y;
                g.yz = a0.yz * x12.xz + h.yz * x12.yw;
                return 130.0 * dot(m, g);
            }

            void main() {
                vec2 uv = v_uv;
                float t = u_time * 0.5;

                // Layered noise
                float n = 0.0;
                n += snoise(uv * 4.0 + t) * 0.5;
                n += snoise(uv * 8.0 - t * 0.7) * 0.25;
                n += snoise(uv * 16.0 + t * 0.3) * 0.125;
                n = n * 0.5 + 0.5;

                // Color gradient
                vec3 col1 = vec3(0.1, 0.3, 0.6);
                vec3 col2 = vec3(0.9, 0.4, 0.1);
                vec3 color = mix(col1, col2, n);

                // Vignette
                float vignette = 1.0 - length(uv - 0.5) * 0.8;
                color *= vignette;

                fragColor = vec4(color, 1.0);
            }
        `;
    }

    compileProgram(vertSource, fragSource) {
        const gl = this.gl;

        const vertShader = gl.createShader(gl.VERTEX_SHADER);
        gl.shaderSource(vertShader, vertSource);
        gl.compileShader(vertShader);

        if (!gl.getShaderParameter(vertShader, gl.COMPILE_STATUS)) {
            console.error('[Renderer] Vertex shader error:', gl.getShaderInfoLog(vertShader));
            return null;
        }

        const fragShader = gl.createShader(gl.FRAGMENT_SHADER);
        gl.shaderSource(fragShader, fragSource);
        gl.compileShader(fragShader);

        if (!gl.getShaderParameter(fragShader, gl.COMPILE_STATUS)) {
            console.error('[Renderer] Fragment shader error:', gl.getShaderInfoLog(fragShader));
            return null;
        }

        const program = gl.createProgram();
        gl.attachShader(program, vertShader);
        gl.attachShader(program, fragShader);
        gl.linkProgram(program);

        if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
            console.error('[Renderer] Program link error:', gl.getProgramInfoLog(program));
            return null;
        }

        return program;
    }

    setInput(input) {
        this.currentInput = input;
    }

    resize() {
        const canvas = this.canvas;
        const container = canvas.parentElement;
        if (container) {
            canvas.width = container.clientWidth;
            canvas.height = container.clientHeight;
        }
        this.gl.viewport(0, 0, canvas.width, canvas.height);
    }

    render() {
        const gl = this.gl;
        if (!gl) return;

        const time = (performance.now() - this.startTime) / 1000;

        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.viewport(0, 0, this.canvas.width, this.canvas.height);
        gl.clearColor(0, 0, 0, 1);
        gl.clear(gl.COLOR_BUFFER_BIT);

        const program = this.defaultProgram;
        if (!program) return;

        gl.useProgram(program);

        // Set uniforms
        const uTime = gl.getUniformLocation(program, 'u_time');
        const uResolution = gl.getUniformLocation(program, 'u_resolution');

        gl.uniform1f(uTime, time);
        gl.uniform2f(uResolution, this.canvas.width, this.canvas.height);

        // Draw fullscreen quad
        const aPosition = gl.getAttribLocation(program, 'a_position');
        gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer);
        gl.enableVertexAttribArray(aPosition);
        gl.vertexAttribPointer(aPosition, 2, gl.FLOAT, false, 0, 0);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }

    start() {
        if (this.isRunning) return;
        this.isRunning = true;
        this.animate();
    }

    stop() {
        this.isRunning = false;
    }

    toggle() {
        if (this.isRunning) this.stop();
        else this.start();
    }

    animate() {
        if (!this.isRunning) return;
        this.render();
        requestAnimationFrame(() => this.animate());
    }
}

// Auto-start on load
window.addEventListener('load', () => {
    const canvas = document.getElementById('previewCanvas');
    if (canvas) {
        window.renderer = new WebGLRenderer(canvas);
        window.renderer.start();
    }
});
