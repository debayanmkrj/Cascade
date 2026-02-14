"""
Mason Agent
-------------
Takes a list of node archetypes (NodeTensor objects) and generates runnable code snippets
for each node, with engine-specific prompting + validation + repair loops.

This version implements:
  - Validator Dispatcher (engine-specific validators; GLSL is NOT parsed as JS)
  - Universal Tensor Node contract alignment (side sampler decls, param naming, inputs arg)
  - Robust sanitization against LLM artifacts (e.g., <|begin_of_sentence|> tokens)
  - Optional module-style nodes via engine='js_module' (imports allowed; syntax-only validation)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import re
import subprocess
import tempfile
import shutil
import requests
from typing import Dict, List, Optional, Any

from phase2.data_types import NodeTensor
from config import OLLAMA_URL, MODEL_NAME_CODING, MODEL_NAME_FALLBACK, SESSIONS_DIR
from phase2.agents.uniform_validator import get_uniform_validator

import random
import hashlib


# ---------------------------------------------------------------------------
# Pre-defined code templates for common nodes (no LLM needed)
# These are drag-and-drop nodes that Mason should use as-is without code generation
# ---------------------------------------------------------------------------

# NOTE: PREDEFINED_CATEGORIES is set after PREDEFINED_CODE is defined (see below).

# Display names for predefined node types (used to override semantic labels)
PREDEFINED_DISPLAY_NAMES = {
    "video_input": "Video Input",
    "webcam_input": "Webcam Input",
    "image_input": "Image Input",
    "audio_input": "Audio Input",
    "media_input": "Media Input",
    "p5_tracking": "P5 Tracking",
    "face_tracking": "Face Tracking",
    "hand_tracking": "Hand Tracking",
    "body_tracking": "Body Tracking",
    "pose_tracking": "Pose Tracking",
    "color_node": "Color Node",
    "color_grade": "Color Node",
    "color_adjustment": "Color Node",
    "color_tint": "Color Tint",
    "noise_generator": "Noise Generator",
    "noise_perlin": "Noise Generator",
    "noise_worley": "Noise Generator",
    "noise_simplex": "Noise Generator",
    "noise_fbm": "Noise Generator",
    "blend_node": "Blend Node",
    "layer_blend": "Blend Node",
    "blend": "Blend Node",
    "blur_gaussian": "Gaussian Blur",
    "blur": "Gaussian Blur",
    "gaussian_blur": "Gaussian Blur",
    "clamp": "Clamp",
    "chromatic_aberration": "Chromatic Aberration",
    "vignette": "Vignette",
    "bloom": "Bloom",
    "glow": "Glow",
    "glitch": "Glitch",
    "displacement": "Displacement",
    "kaleidoscope": "Kaleidoscope",
    "feedback_effect": "Feedback",
    "particle_node": "Particle Node",
    "particles": "Particle Node",
    "particle_system": "Particle Node",
    "composite_output": "Output",
    "post_process": "Post Process",
    "output": "Output",
}

PREDEFINED_CODE: Dict[str, Dict[str, Any]] = {
    # =========================================================================
    # TASK 4a: Video/Image/Audio Input Node
    # =========================================================================
    "video_input": {
        "engine": "html_video",
        "is_source": True,
        "code": "",  # Handled by runtime - just config
        "parameters": {
            "source_type": {"type": "dropdown", "default": "webcam", "options": ["webcam", "file", "url"], "ui": "dropdown"},
            "src": {"type": "string", "default": "", "ui": "file_picker"},
            "flip_horizontal": {"type": "bool", "default": False, "ui": "toggle"},
            "playback_rate": {"type": "float", "default": 1.0, "range": [0.1, 4.0], "ui": "slider"},
        },
        "manifest": {"semantic": "IMAGE_RGBA", "dynamic": True}
    },

    "webcam_input": {
        "engine": "html_video",
        "is_source": True,
        "code": "",
        "parameters": {
            "device_id": {"type": "string", "default": "", "ui": "dropdown"},
            "flip_horizontal": {"type": "bool", "default": True, "ui": "toggle"},
            "resolution": {"type": "dropdown", "default": "720p",
                          "options": ["480p", "720p", "1080p"], "ui": "dropdown"},
        },
        "manifest": {"semantic": "IMAGE_RGBA", "dynamic": True}
    },

    "audio_input": {
        "engine": "webaudio",
        "is_source": True,
        "code": """const audioCtx = ctx.audioContext || new AudioContext();
const analyser = audioCtx.createAnalyser();
analyser.fftSize = params.fft_size || 256;
const dataArray = new Uint8Array(analyser.frequencyBinCount);

// Get microphone or audio element
const source_type = params.source_type || 'microphone';
let sourceNode;

if (source_type === 'microphone') {
    navigator.mediaDevices.getUserMedia({ audio: true })
        .then(stream => {
            sourceNode = audioCtx.createMediaStreamSource(stream);
            sourceNode.connect(analyser);
        });
}

return {
    analyser,
    dataArray,
    update: () => {
        analyser.getByteFrequencyData(dataArray);
        return {
            payload: dataArray,
            manifest: { semantic: 'AUDIO_FFT', shape: [dataArray.length] }
        };
    },
    dispose: () => {
        if (sourceNode) sourceNode.disconnect();
    }
};""",
        "parameters": {
            "source_type": {"type": "dropdown", "default": "microphone",
                          "options": ["microphone", "audio_file", "system"], "ui": "dropdown"},
            "fft_size": {"type": "dropdown", "default": 256,
                        "options": [64, 128, 256, 512, 1024, 2048], "ui": "dropdown"},
            "smoothing": {"type": "float", "default": 0.8, "range": [0.0, 1.0], "ui": "slider"},
            "gain": {"type": "float", "default": 1.0, "range": [0.0, 3.0], "ui": "slider"},
        },
        "manifest": {"semantic": "AUDIO_FFT", "dynamic": True}
    },

    "image_input": {
        "engine": "glsl",
        "is_source": True,
        "code": """void main() {
    vec2 uv = v_uv;
    uv = (uv - 0.5) / u_scale + 0.5;
    uv.x += u_offset_x;
    uv.y += u_offset_y;
    vec4 col = texture(u_input0, uv);
    fragColor = col;
}""",
        "parameters": {
            "src": {"type": "string", "default": "", "ui": "file_picker"},
            "scale": {"type": "float", "default": 1.0, "range": [0.1, 4.0], "ui": "slider"},
            "offset_x": {"type": "float", "default": 0.0, "range": [-1.0, 1.0], "ui": "slider"},
            "offset_y": {"type": "float", "default": 0.0, "range": [-1.0, 1.0], "ui": "slider"},
        },
        "manifest": {"semantic": "IMAGE_RGBA", "dynamic": False}
    },

    # =========================================================================
    # TASK 4b: P5.js Tracking Node (face, hands, body)
    # =========================================================================
    "p5_tracking": {
        "engine": "canvas2d",
        "is_source": False,
        "code": """// Tracking placeholder — draws animated landmark visualization
// (ml5/mediapipe requires external setup; this provides a visual stand-in)
const W = width, H = height;
const numPoints = params.max_detections || 12;
const points = [];
for (let i = 0; i < numPoints; i++) {
    points.push({
        x: W * 0.3 + Math.random() * W * 0.4,
        y: H * 0.2 + Math.random() * H * 0.6,
        vx: (Math.random() - 0.5) * 0.5,
        vy: (Math.random() - 0.5) * 0.5,
    });
}

function draw(ctx, w, h, t, inp, p) {
    // Draw upstream texture if available
    if (inp && inp.readTexture) {
        for (const [id, tex] of Object.entries(inp.textures || {})) {
            if (id === 'prev_layer_texture') continue;
            const img = inp.readTexture(id);
            if (img) { ctx.putImageData(img, 0, 0); break; }
        }
    } else {
        ctx.fillStyle = 'rgba(0,0,0,0.1)';
        ctx.fillRect(0, 0, w, h);
    }
    // Animate landmark points
    const speed = p.confidence_threshold || 0.5;
    for (const pt of points) {
        pt.x += pt.vx * speed;
        pt.y += pt.vy * speed;
        if (pt.x < w*0.2 || pt.x > w*0.8) pt.vx *= -1;
        if (pt.y < h*0.1 || pt.y > h*0.9) pt.vy *= -1;
    }
    // Draw connections
    ctx.strokeStyle = 'rgba(0, 255, 128, 0.4)';
    ctx.lineWidth = 1;
    for (let i = 0; i < points.length - 1; i++) {
        ctx.beginPath();
        ctx.moveTo(points[i].x, points[i].y);
        ctx.lineTo(points[i+1].x, points[i+1].y);
        ctx.stroke();
    }
    // Draw points
    for (const pt of points) {
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 3, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(0, 255, 128, 0.8)';
        ctx.fill();
    }
}""",
        "parameters": {
            "tracking_type": {"type": "dropdown", "default": "face",
                            "options": ["face", "hand", "body"], "ui": "dropdown"},
            "max_detections": {"type": "int", "default": 12, "range": [1, 30], "ui": "slider"},
            "confidence_threshold": {"type": "float", "default": 0.5, "range": [0.0, 1.0], "ui": "slider"},
        },
        "manifest": {"semantic": "LANDMARKS_FACE", "dynamic": True}
    },

    # =========================================================================
    # TASK 4c: Color Node with HUE PICKER (not RGB)
    # =========================================================================
    "color_node": {
        "engine": "glsl",
        "is_source": False,
        "code": """// RGB to HSV conversion
vec3 rgb2hsv(vec3 c) {
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

// HSV to RGB conversion
vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main() {
    vec2 uv = v_uv;
    vec4 col = texture(u_input0, uv);

    // Convert to HSV
    vec3 hsv = rgb2hsv(col.rgb);

    // Apply hue shift (u_hue is 0-1, maps to 0-360 degrees)
    hsv.x = fract(hsv.x + u_hue);

    // Apply saturation adjustment
    hsv.y *= u_saturation;

    // Convert back to RGB
    col.rgb = hsv2rgb(hsv);

    // Apply exposure (EV stops)
    col.rgb *= pow(2.0, u_exposure);

    // Apply brightness offset
    col.rgb += u_brightness - 1.0;

    // Apply contrast
    col.rgb = (col.rgb - 0.5) * u_contrast + 0.5;

    // Apply transparency/opacity
    col.a *= u_transparency;

    fragColor = s_vec4(clamp(col.rgb, 0.0, 1.0), col.a);
}""",
        "parameters": {
            "hue": {"type": "float", "default": 0.0, "range": [0.0, 1.0], "ui": "hue_picker"},
            "exposure": {"type": "float", "default": 0.0, "range": [-3.0, 3.0], "ui": "slider"},
            "saturation": {"type": "float", "default": 1.0, "range": [0.0, 2.0], "ui": "slider"},
            "brightness": {"type": "float", "default": 1.0, "range": [0.0, 2.0], "ui": "slider"},
            "contrast": {"type": "float", "default": 1.0, "range": [0.5, 2.0], "ui": "slider"},
            "transparency": {"type": "float", "default": 1.0, "range": [0.0, 1.0], "ui": "slider"},
        },
        "manifest": {"semantic": "IMAGE_RGBA", "dynamic": True}
    },

    # Alias for color_grade to also use color_node
    "color_grade": {
        "engine": "glsl",
        "is_source": False,
        "alias_of": "color_node"
    },

    "color_adjustment": {
        "engine": "glsl",
        "is_source": False,
        "alias_of": "color_node"
    },

    # =========================================================================
    # TASK 4d: Noise Generator Node with Type Controls
    # =========================================================================
    "noise_generator": {
        "engine": "glsl",
        "is_source": True,
        "code": """void main() {
    vec2 uv = v_uv;
    vec2 p = uv * u_frequency;
    float n = 0.0;
    float t = u_time * u_speed;

    int ntype = int(u_noise_type);

    if (ntype == 0 || ntype == 3) {
        float amp = u_amplitude;
        float freq = 1.0;
        for (int i = 0; i < 8; i++) {
            if (i >= int(u_octaves)) break;
            n += amp * snoise(vec3(p * freq, t * 0.2));
            freq *= u_lacunarity;
            amp *= u_persistence;
        }
    } else if (ntype == 1) {
        n = snoise(vec3(p, t)) * u_amplitude;
    } else if (ntype == 2) {
        vec2 pAnim = p + vec2(t * 0.1);
        vec2 ip = floor(pAnim);
        vec2 fp = fract(pAnim);
        float minDist = 1.0;
        for (int y = -1; y <= 1; y++) {
            for (int x = -1; x <= 1; x++) {
                vec2 neighbor = vec2(float(x), float(y));
                vec2 cellPos = ip + neighbor;
                float h = fract(sin(dot(cellPos, vec2(127.1, 311.7))) * 43758.5453);
                vec2 offset = vec2(h, fract(h * 17.3));
                vec2 diff = neighbor + offset - fp;
                minDist = min(minDist, length(diff));
            }
        }
        n = minDist * u_amplitude;
    }

    n = n * 0.5 + 0.5;
    fragColor = vec4(vec3(n), 1.0);
}""",
        "parameters": {
            "noise_type": {"type": "dropdown", "default": 0,
                         "options": [{"label": "Perlin", "value": 0},
                                    {"label": "Simplex", "value": 1},
                                    {"label": "Worley", "value": 2},
                                    {"label": "FBM", "value": 3}], "ui": "dropdown"},
            "frequency": {"type": "float", "default": 4.0, "range": [0.5, 20.0], "ui": "slider"},
            "amplitude": {"type": "float", "default": 1.0, "range": [0.0, 2.0], "ui": "slider"},
            "speed": {"type": "float", "default": 0.3, "range": [0.0, 3.0], "ui": "slider"},
            "octaves": {"type": "int", "default": 4, "range": [1, 8], "ui": "slider"},
            "lacunarity": {"type": "float", "default": 2.0, "range": [1.0, 4.0], "ui": "slider"},
            "persistence": {"type": "float", "default": 0.5, "range": [0.0, 1.0], "ui": "slider"},
        },
        "manifest": {"semantic": "IMAGE_RGBA", "dynamic": True}
    },

    # Keep specific noise types as aliases
    "noise_perlin": {"engine": "glsl", "is_source": True, "alias_of": "noise_generator"},
    "noise_worley": {"engine": "glsl", "is_source": True, "alias_of": "noise_generator"},
    "noise_simplex": {"engine": "glsl", "is_source": True, "alias_of": "noise_generator"},
    "noise_fbm": {"engine": "glsl", "is_source": True, "alias_of": "noise_generator"},

    # =========================================================================
    # TASK 4e: Blend Node (Z-layer blending)
    # =========================================================================
    "blend_node": {
        "engine": "glsl",
        "is_source": False,
        "z_layer_effect": True,  # Affects entire Z-layer
        "code": """void main() {
    vec2 uv = v_uv;
    vec4 base = texture(u_input0, uv);

    // Blend uses u_input0 as both base and blend source
    // When wired, the prev-layer composite IS the blend target
    vec4 bl = base;

    vec3 result;
    int bm = int(u_blend_mode);

    if (bm == 0) {
        result = mix(base.rgb, bl.rgb, bl.a * u_opacity);
    } else if (bm == 1) {
        result = base.rgb + bl.rgb * u_opacity;
    } else if (bm == 2) {
        result = mix(base.rgb, base.rgb * bl.rgb, u_opacity);
    } else if (bm == 3) {
        result = mix(base.rgb, 1.0 - (1.0 - base.rgb) * (1.0 - bl.rgb), u_opacity);
    } else if (bm == 4) {
        // Overlay: step() replaces vec3 < comparison (invalid in GLSL ES 3.00)
        vec3 s = step(0.5, base.rgb);
        vec3 ov = mix(2.0 * base.rgb * bl.rgb, 1.0 - 2.0 * (1.0 - base.rgb) * (1.0 - bl.rgb), s);
        result = mix(base.rgb, ov, u_opacity);
    } else if (bm == 5) {
        // Soft Light
        vec3 s = step(0.5, bl.rgb);
        vec3 lo = base.rgb - (1.0 - 2.0 * bl.rgb) * base.rgb * (1.0 - base.rgb);
        vec3 hi = base.rgb + (2.0 * bl.rgb - 1.0) * (sqrt(base.rgb) - base.rgb);
        vec3 sf = mix(lo, hi, s);
        result = mix(base.rgb, sf, u_opacity);
    } else if (bm == 6) {
        // Hard Light
        vec3 s = step(0.5, bl.rgb);
        vec3 hl = mix(2.0 * base.rgb * bl.rgb, 1.0 - 2.0 * (1.0 - base.rgb) * (1.0 - bl.rgb), s);
        result = mix(base.rgb, hl, u_opacity);
    } else if (bm == 7) {
        result = mix(base.rgb, abs(base.rgb - bl.rgb), u_opacity);
    } else {
        result = mix(base.rgb, base.rgb + bl.rgb - 2.0 * base.rgb * bl.rgb, u_opacity);
    }

    fragColor = s_vec4(clamp(result, 0.0, 1.0), base.a);
}""",
        "parameters": {
            "blend_mode": {"type": "dropdown", "default": 0,
                         "options": [{"label": "Normal", "value": 0},
                                    {"label": "Add", "value": 1},
                                    {"label": "Multiply", "value": 2},
                                    {"label": "Screen", "value": 3},
                                    {"label": "Overlay", "value": 4},
                                    {"label": "Soft Light", "value": 5},
                                    {"label": "Hard Light", "value": 6},
                                    {"label": "Difference", "value": 7},
                                    {"label": "Exclusion", "value": 8}], "ui": "dropdown"},
            "opacity": {"type": "float", "default": 1.0, "range": [0.0, 1.0], "ui": "slider"},
        },
        "manifest": {"semantic": "IMAGE_RGBA", "dynamic": True}
    },

    "layer_blend": {"engine": "glsl", "is_source": False, "alias_of": "blend_node"},

    # =========================================================================
    # Common post-processing nodes (hardcoded — bypasses LLM entirely)
    # mason:latest leaks tokens for these trivial shaders, so we hardcode them.
    # =========================================================================
    "blur_gaussian": {
        "engine": "glsl",
        "is_source": False,
        "code": """void main() {
    vec2 uv = v_uv;
    vec2 px = 1.0 / u_resolution;
    float r = u_radius;
    vec4 sum = vec4(0.0);
    float total = 0.0;
    for (float x = -4.0; x <= 4.0; x += 1.0) {
        for (float y = -4.0; y <= 4.0; y += 1.0) {
            float w = exp(-(x*x + y*y) / (2.0 * r * r));
            sum += texture(u_input0, uv + vec2(x, y) * px * r) * w;
            total += w;
        }
    }
    fragColor = sum / total;
}""",
        "parameters": {
            "radius": {"type": "float", "default": 2.0, "range": [0.5, 10.0], "ui": "slider"},
        },
    },
    "blur": {"engine": "glsl", "is_source": False, "alias_of": "blur_gaussian"},
    "gaussian_blur": {"engine": "glsl", "is_source": False, "alias_of": "blur_gaussian"},

    "clamp": {
        "engine": "glsl",
        "is_source": False,
        "code": """void main() {
    vec2 uv = v_uv;
    vec4 col = texture(u_input0, uv);
    col.rgb = clamp(col.rgb, vec3(u_min_val), vec3(u_max_val));
    fragColor = col;
}""",
        "parameters": {
            "min_val": {"type": "float", "default": 0.0, "range": [0.0, 1.0], "ui": "slider"},
            "max_val": {"type": "float", "default": 1.0, "range": [0.0, 1.0], "ui": "slider"},
        },
    },

    "chromatic_aberration": {
        "engine": "glsl",
        "is_source": False,
        "code": """void main() {
    vec2 uv = v_uv;
    vec2 dir = (uv - 0.5) * u_intensity * 0.01;
    float r = texture(u_input0, uv + dir).r;
    float g = texture(u_input0, uv).g;
    float b = texture(u_input0, uv - dir).b;
    float a = texture(u_input0, uv).a;
    fragColor = vec4(r, g, b, a);
}""",
        "parameters": {
            "intensity": {"type": "float", "default": 3.0, "range": [0.0, 20.0], "ui": "slider"},
        },
    },

    "vignette": {
        "engine": "glsl",
        "is_source": False,
        "code": """void main() {
    vec2 uv = v_uv;
    vec4 col = texture(u_input0, uv);
    float d = distance(uv, vec2(0.5));
    float vig = smoothstep(u_radius, u_radius - u_softness, d);
    col.rgb *= vig;
    fragColor = col;
}""",
        "parameters": {
            "radius": {"type": "float", "default": 0.75, "range": [0.1, 1.5], "ui": "slider"},
            "softness": {"type": "float", "default": 0.45, "range": [0.0, 1.0], "ui": "slider"},
        },
    },

    "bloom": {
        "engine": "glsl",
        "is_source": False,
        "code": """void main() {
    vec2 uv = v_uv;
    vec2 px = 1.0 / u_resolution;
    vec4 col = texture(u_input0, uv);

    // Extract bright areas
    vec3 bright = max(col.rgb - vec3(u_threshold), vec3(0.0));

    // Simple blur of bright areas
    vec3 bloom_col = vec3(0.0);
    float total = 0.0;
    for (float x = -3.0; x <= 3.0; x += 1.0) {
        for (float y = -3.0; y <= 3.0; y += 1.0) {
            float w = exp(-(x*x + y*y) / 8.0);
            vec4 s = texture(u_input0, uv + vec2(x, y) * px * u_radius);
            bloom_col += max(s.rgb - vec3(u_threshold), vec3(0.0)) * w;
            total += w;
        }
    }
    bloom_col /= total;

    fragColor = vec4(col.rgb + bloom_col * u_intensity, col.a);
}""",
        "parameters": {
            "threshold": {"type": "float", "default": 0.6, "range": [0.0, 1.0], "ui": "slider"},
            "intensity": {"type": "float", "default": 0.8, "range": [0.0, 3.0], "ui": "slider"},
            "radius": {"type": "float", "default": 3.0, "range": [1.0, 10.0], "ui": "slider"},
        },
    },
    "glow": {"engine": "glsl", "is_source": False, "alias_of": "bloom"},

    "glitch": {
        "engine": "glsl",
        "is_source": False,
        "code": """void main() {
    vec2 uv = v_uv;
    float t = u_time * u_speed;
    float glitchLine = step(0.99 - u_intensity * 0.1, fract(sin(floor(uv.y * 20.0 + t * 5.0)) * 43758.5));
    float shift = glitchLine * (hash(vec2(floor(uv.y * 20.0), floor(t * 3.0))) - 0.5) * u_intensity * 0.1;
    vec4 col;
    col.r = texture(u_input0, uv + vec2(shift, 0.0)).r;
    col.g = texture(u_input0, uv).g;
    col.b = texture(u_input0, uv - vec2(shift, 0.0)).b;
    col.a = texture(u_input0, uv).a;
    fragColor = col;
}""",
        "parameters": {
            "intensity": {"type": "float", "default": 0.5, "range": [0.0, 2.0], "ui": "slider"},
            "speed": {"type": "float", "default": 1.0, "range": [0.0, 5.0], "ui": "slider"},
        },
    },

    "displacement": {
        "engine": "glsl",
        "is_source": False,
        "code": """void main() {
    vec2 uv = v_uv;
    vec2 px = 1.0 / u_resolution;
    // Use noise as displacement map
    float dx = (noise(uv * u_scale + u_time * 0.2) - 0.5) * u_amount;
    float dy = (noise(uv * u_scale + 100.0 + u_time * 0.2) - 0.5) * u_amount;
    vec2 displaced = uv + vec2(dx, dy) * px * 20.0;
    fragColor = texture(u_input0, displaced);
}""",
        "parameters": {
            "amount": {"type": "float", "default": 1.0, "range": [0.0, 5.0], "ui": "slider"},
            "scale": {"type": "float", "default": 3.0, "range": [0.5, 20.0], "ui": "slider"},
        },
    },

    "kaleidoscope": {
        "engine": "glsl",
        "is_source": False,
        "code": """void main() {
    vec2 uv = v_uv - 0.5;
    float angle = atan(uv.y, uv.x);
    float r = length(uv);
    float seg = PI * 2.0 / u_segments;
    angle = mod(angle, seg);
    if (angle > seg * 0.5) angle = seg - angle;
    uv = vec2(cos(angle), sin(angle)) * r + 0.5;
    fragColor = texture(u_input0, uv);
}""",
        "parameters": {
            "segments": {"type": "float", "default": 6.0, "range": [2.0, 16.0], "ui": "slider"},
        },
    },

    "feedback_effect": {
        "engine": "glsl",
        "is_source": False,
        "code": """void main() {
    vec2 uv = v_uv;
    vec4 current = texture(u_input0, uv);
    // Slight zoom toward center for feedback trail
    vec2 offset = (uv - 0.5) * (1.0 - u_zoom * 0.01) + 0.5;
    vec4 prev = texture(u_input0, offset);
    fragColor = mix(current, prev, u_decay);
}""",
        "parameters": {
            "decay": {"type": "float", "default": 0.85, "range": [0.0, 0.99], "ui": "slider"},
            "zoom": {"type": "float", "default": 1.0, "range": [0.0, 5.0], "ui": "slider"},
        },
    },

    # =========================================================================
    # TASK 4f: Particle.js Node
    # =========================================================================
    "particle_node": {
        "engine": "canvas2d",
        "is_source": True,
        "code": """// Pure Canvas2D particle system (no external libraries)
const W = width, H = height;
const count = params.count || 80;
const spd = params.speed || 2;
const sz = params.size || 3;
const sizeRandom = params.size_random !== false;
const linkDist = params.link_distance || 150;
const linksOn = params.links_enabled !== false;
const baseOpacity = params.opacity || 0.5;

// Parse hex color
function hexToRgb(hex) {
    hex = (hex || '#ffffff').replace('#', '');
    return {
        r: parseInt(hex.substring(0,2), 16) || 255,
        g: parseInt(hex.substring(2,4), 16) || 255,
        b: parseInt(hex.substring(4,6), 16) || 255
    };
}

const col = hexToRgb(params.color);
const linkCol = hexToRgb(params.link_color);

// Init particles
const particles = [];
for (let i = 0; i < count; i++) {
    const angle = Math.random() * Math.PI * 2;
    particles.push({
        x: Math.random() * W,
        y: Math.random() * H,
        vx: Math.cos(angle) * spd * (0.5 + Math.random()),
        vy: Math.sin(angle) * spd * (0.5 + Math.random()),
        r: sizeRandom ? sz * (0.3 + Math.random() * 0.7) : sz,
        phase: Math.random() * Math.PI * 2,
    });
}

function draw(ctx, w, h, t, inputs, params) {
    ctx.clearRect(0, 0, w, h);

    const dt = 0.016;
    const curSpd = params.speed || spd;

    // Update & draw particles
    for (const p of particles) {
        p.x += p.vx * dt * curSpd;
        p.y += p.vy * dt * curSpd;

        // Wrap around
        if (p.x < -10) p.x = w + 10;
        if (p.x > w + 10) p.x = -10;
        if (p.y < -10) p.y = h + 10;
        if (p.y > h + 10) p.y = -10;

        const alpha = baseOpacity * (0.6 + 0.4 * Math.sin(t * 2 + p.phase));
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(' + col.r + ',' + col.g + ',' + col.b + ',' + alpha + ')';
        ctx.fill();
    }

    // Draw links
    if (linksOn) {
        const maxD2 = linkDist * linkDist;
        const linkOpacity = params.link_opacity || 0.4;
        for (let i = 0; i < particles.length; i++) {
            for (let j = i + 1; j < particles.length; j++) {
                const dx = particles[i].x - particles[j].x;
                const dy = particles[i].y - particles[j].y;
                const d2 = dx * dx + dy * dy;
                if (d2 < maxD2) {
                    const a = linkOpacity * (1 - Math.sqrt(d2) / linkDist);
                    ctx.beginPath();
                    ctx.moveTo(particles[i].x, particles[i].y);
                    ctx.lineTo(particles[j].x, particles[j].y);
                    ctx.strokeStyle = 'rgba(' + linkCol.r + ',' + linkCol.g + ',' + linkCol.b + ',' + a + ')';
                    ctx.lineWidth = 0.5;
                    ctx.stroke();
                }
            }
        }
    }
}""",
        "parameters": {
            "count": {"type": "int", "default": 80, "range": [10, 500], "ui": "slider"},
            "color": {"type": "color", "default": "#ffffff", "ui": "color_picker"},
            "size": {"type": "float", "default": 3.0, "range": [1.0, 20.0], "ui": "slider"},
            "size_random": {"type": "bool", "default": True, "ui": "toggle"},
            "opacity": {"type": "float", "default": 0.5, "range": [0.0, 1.0], "ui": "slider"},
            "speed": {"type": "float", "default": 2.0, "range": [0.1, 10.0], "ui": "slider"},
            "shape": {"type": "dropdown", "default": "circle",
                     "options": ["circle", "edge", "triangle", "polygon", "star"], "ui": "dropdown"},
            "links_enabled": {"type": "bool", "default": True, "ui": "toggle"},
            "link_distance": {"type": "float", "default": 150, "range": [50, 300], "ui": "slider"},
            "link_color": {"type": "color", "default": "#ffffff", "ui": "color_picker"},
            "link_opacity": {"type": "float", "default": 0.4, "range": [0.0, 1.0], "ui": "slider"},
            "hover_enabled": {"type": "bool", "default": True, "ui": "toggle"},
            "hover_mode": {"type": "dropdown", "default": "repulse",
                          "options": ["grab", "repulse", "bubble"], "ui": "dropdown"},
            "direction": {"type": "dropdown", "default": "none",
                         "options": ["none", "top", "bottom", "left", "right"], "ui": "dropdown"},
        },
        "manifest": {"semantic": "IMAGE_RGBA", "dynamic": True}
    },

    "particles": {"engine": "canvas2d", "is_source": True, "alias_of": "particle_node"},
    "particle_system": {"engine": "canvas2d", "is_source": True, "alias_of": "particle_node"},

    # =========================================================================
    # Legacy pre-defined nodes (kept for backward compatibility)
    # =========================================================================
    "color_tint": {
        "engine": "glsl",
        "code": """void main() {
    vec2 uv = v_uv;
    vec4 col = texture(u_input0, uv);
    vec3 tc = vec3(u_tint_r, u_tint_g, u_tint_b);
    col.rgb = mix(col.rgb, col.rgb * tc, u_tint_strength);
    fragColor = col;
}""",
        "parameters": {
            "tint_r": {"type": "float", "default": 1.0, "range": [0.0, 1.0], "ui": "color_picker", "group": "tint"},
            "tint_g": {"type": "float", "default": 1.0, "range": [0.0, 1.0], "ui": "color_picker", "group": "tint"},
            "tint_b": {"type": "float", "default": 1.0, "range": [0.0, 1.0], "ui": "color_picker", "group": "tint"},
            "tint_strength": {"type": "float", "default": 0.5, "range": [0.0, 1.0], "ui": "slider"},
        }
    },

    "composite_output": {
        "engine": "glsl",
        "code": """void main() {
    vec2 uv = v_uv;
    vec4 col = texture(u_input0, uv);
    col.a *= u_opacity;
    fragColor = col;
}""",
        "parameters": {
            "opacity": {"type": "float", "default": 1.0, "range": [0.0, 1.0], "ui": "slider"},
        }
    },

    "post_process": {
        "engine": "glsl",
        "code": """void main() {
    vec2 uv = v_uv;
    vec4 col = texture(u_input0, uv);
    vec2 vc = uv - 0.5;
    float vig = 1.0 - dot(vc, vc) * u_vignette * 2.0;
    col.rgb *= vig;
    float gr = (fract(sin(dot(uv + u_time, vec2(12.9898, 78.233))) * 43758.5453) - 0.5) * u_grain;
    col.rgb += gr;
    fragColor = s_vec4(clamp(col.rgb, 0.0, 1.0), col.a);
}""",
        "parameters": {
            "vignette": {"type": "float", "default": 0.3, "range": [0.0, 1.0], "ui": "slider"},
            "grain": {"type": "float", "default": 0.05, "range": [0.0, 0.3], "ui": "slider"},
        }
    },

}


def _meta_attr(meta, key, default=""):
    """Safely get a meta field whether meta is a dataclass or dict."""
    if meta is None:
        return default
    if hasattr(meta, key):
        return getattr(meta, key, default) or default
    if isinstance(meta, dict):
        return meta.get(key, default) or default
    return default


# PREDEFINED_CATEGORIES: exact keys in PREDEFINED_CODE. Only these trigger
# template usage — no fuzzy/keyword hijacking.
PREDEFINED_CATEGORIES = set(PREDEFINED_CODE.keys())

# -----------------------------------------------------------------------------
# Engine templates (prompt fragments)
# -----------------------------------------------------------------------------

ENGINE_TEMPLATES: Dict[str, str] = {
    "glsl": """You are writing a **GLSL ES 3.00 fragment shader** for WebGL2.

Output the shader code. You may use markdown code fences if helpful.

Available identifiers (already declared for you):
- in vec2 v_uv;
- out vec4 fragColor;
- uniform float u_time;
- uniform vec2 u_resolution;
- uniform sampler2D u_input0;              // main input texture (prev layer composite)
- uniform sampler2D u_<inputNodeId>;       // side input textures (one per connected input node)
- uniform float u_<paramName>;             // numeric params from node.parameters

Utility functions available (already injected):
- float hash(float)          // Simple hash, takes float ONLY
- vec2 hash2(vec2)           // Hash function for 2D
- vec3 hash3(vec3)           // Hash function for 3D
- float noise(vec2)          // Perlin noise
- float snoise(vec2)         // Simplex noise, returns [-1,1]
- float fbm(vec2)            // Fractal Brownian Motion
- float worley(vec2)         // Cellular/Worley noise
- float voronoi(vec2)        // Voronoi distance field
- float simplex(vec2)        // Simplex noise variant
- float perlin(vec2)         // Perlin noise variant

IMPORTANT - These functions DO NOT EXIST (don't use them):
- s_vec2(), s_vec3(), s_vec4()  // NOT available - use standard vec2/vec3/vec4 constructors
- ridgedFBM, turbulence         // Only fbm() exists
- hashVec, hash3, hash2 as functions with vec arguments

STRICT RULES:
- DO NOT include: #version, precision, in/out declarations, uniform declarations, layout(), gl_FragColor, texture2D.
- DO use: texture(u_input0, uv) and write to fragColor.
- CRITICAL: NEVER declare local variables with uniform names (NO "float u_speed = u_speed;"). Uniforms are already declared - use them directly!
- IMPORTANT: NEVER reference parameter names without the u_ prefix (use u_scale, not scale).
- IMPORTANT: Use standard GLSL constructors: vec2(x,y), vec3(x,y,z), vec4(x,y,z,w) — NOT s_vec* wrappers.
- If reference code uses non-standard functions - replace them with available functions above.

Output must include:
void main() {
    vec2 uv = v_uv;
    // your code here
    fragColor = vec4(color, 1.0);
}
""",
    "regl": """Same as GLSL. Output only the GLSL fragment shader body compatible with WebGL2.

Follow the exact rules from the GLSL template.
""",
    "three_js": """You are a Three.js Creative Coder.
Write an ES module for a Three.js node. Use ES imports from CDN.

REQUIREMENTS:
1. IMPORT: import * as THREE from 'https://unpkg.com/three@0.160.0/build/three.module.js';
2. EXPORT: export default function init(canvas, width, height, params)
3. RETURN: The init function MUST return a closure (time, inputs) => { ... } called every frame.
4. SCENE: Setup a Renderer (with alpha:true), Scene, and Camera inside init.
5. OUTPUT: The renderer must target the passed canvas.

CODE TEMPLATE:
```javascript
import * as THREE from 'https://unpkg.com/three@0.160.0/build/three.module.js';

export default function init(canvas, width, height, params) {
    const renderer = new THREE.WebGLRenderer({ canvas, alpha: true });
    renderer.setSize(width, height, false);
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
    camera.position.z = 5;

    // Lights
    scene.add(new THREE.DirectionalLight(0xffffff, 1));
    scene.add(new THREE.AmbientLight(0x404040));

    // Create objects based on keywords / params
    const geometry = new THREE.TorusGeometry(1.5, 0.5, 32, 64);
    const material = new THREE.MeshStandardMaterial({
        color: params.color || 0x44aaff,
        metalness: params.metalness || 0.8,
        roughness: params.roughness || 0.2
    });
    const mesh = new THREE.Mesh(geometry, material);
    scene.add(mesh);

    // Return frame update function
    return (time, inputs) => {
        mesh.rotation.x = time * 0.5;
        mesh.rotation.y = time * 0.7;
        renderer.render(scene, camera);
    };
}
```

OUTPUT ONLY THE JAVASCRIPT CODE (full ES module). No markdown fences.
""",
    "webaudio": """You are writing a function that will run inside a WebAudio node (no imports).

Context:
  function audioFactory(audioCtx, params, inputs) {
    // YOUR BODY HERE
  }

Where:
- audioCtx is a real AudioContext
- params is a plain object with numeric params
- inputs is { textures: {...}, data: {...} }

STRICT RULES:
- DO NOT write imports, require(), exports, HTML, window/document/navigator calls.
- You MUST return an object.
- Recommended return fields:
  - analyser: AnalyserNode
  - dataArray: Uint8Array (frequency bins or waveform)
  - update(t, inputs): fills dataArray and returns it
""",
    "events": """You are writing a function that will run inside an Events/Control node (no imports).

Context:
  function eventFactory(params, inputs) {
    // YOUR BODY HERE
  }

Where:
- params is a plain object
- inputs is { textures: {...}, data: {...} }

CRITICAL RULES:
1. Initialize state with simple values (e.g., const state = { value: 0 };)
2. DO NOT try to read from inputs during initialization (inputs.textures.xxx.get() will fail!)
3. You can read from inputs inside methods (get/set/update)
4. MUST end with "return { ... };"

Example structure:
const state = { value: params.default || 0 };  // Use params, NOT inputs!
return {
  state,
  get: () => state.value,
  set: (v) => { state.value = v; },
  update: (t, inputs) => {
    // You CAN read from inputs here
  }
};

STRICT RULES:
- DO NOT write imports, require(), exports, HTML.
- MUST include "return { ... };" at the end
  }
""",
    "canvas2d": """You are writing code for a Canvas2D node. Output ONLY the JavaScript code body.

CONTEXT (already provided — do NOT redeclare):
- ctx (alias: ctx2d): CanvasRenderingContext2D
- canvas: the canvas element
- width, height: canvas dimensions (numbers)
- params: object with numeric parameters (e.g., params.radius, params.speed)
- inputs: { textures: {...}, data: { time: <seconds>, ... } }

YOUR CODE must define a draw() function:

function draw(ctx, w, h, t, inputs, params) {
    // clear + draw each frame
    // t = time in seconds
}

EXAMPLE — Animated circles:
```
function draw(ctx, w, h, t, inputs, params) {
    ctx.clearRect(0, 0, w, h);
    const count = params.count || 5;
    const radius = params.radius || 40;
    for (let i = 0; i < count; i++) {
        const angle = (i / count) * Math.PI * 2 + t;
        const x = w/2 + Math.cos(angle) * w * 0.3;
        const y = h/2 + Math.sin(angle) * h * 0.3;
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fillStyle = `hsl(${(i / count) * 360}, 80%, 60%)`;
        ctx.fill();
    }
}
```

EXAMPLE — Flow field particles:
```
const particles = [];
for (let i = 0; i < 200; i++) {
    particles.push({ x: Math.random() * width, y: Math.random() * height,
                      vx: 0, vy: 0, hue: Math.random() * 360 });
}

function draw(ctx, w, h, t, inputs, params) {
    ctx.fillStyle = 'rgba(0,0,0,0.05)';
    ctx.fillRect(0, 0, w, h);
    const speed = params.speed || 2;
    const scale = params.scale || 0.01;
    for (const p of particles) {
        const angle = Math.sin(p.x * scale) * Math.cos(p.y * scale) * Math.PI * 2 + t * 0.5;
        p.vx += Math.cos(angle) * 0.3;
        p.vy += Math.sin(angle) * 0.3;
        p.vx *= 0.95; p.vy *= 0.95;
        p.x += p.vx * speed; p.y += p.vy * speed;
        if (p.x < 0) p.x = w; if (p.x > w) p.x = 0;
        if (p.y < 0) p.y = h; if (p.y > h) p.y = 0;
        ctx.beginPath();
        ctx.arc(p.x, p.y, 2, 0, Math.PI * 2);
        ctx.fillStyle = `hsla(${p.hue}, 90%, 60%, 0.8)`;
        ctx.fill();
    }
}
```

EXAMPLE — Rectangle grid:
```
function draw(ctx, w, h, t, inputs, params) {
    ctx.clearRect(0, 0, w, h);
    const cols = params.columns || 8;
    const rows = params.rows || 8;
    const gap = params.gap || 4;
    const cellW = (w - gap * (cols + 1)) / cols;
    const cellH = (h - gap * (rows + 1)) / rows;
    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            const x = gap + c * (cellW + gap);
            const y = gap + r * (cellH + gap);
            const hue = ((c + r) / (cols + rows)) * 360 + t * 30;
            const scale = 0.8 + 0.2 * Math.sin(t * 2 + c * 0.5 + r * 0.3);
            ctx.save();
            ctx.translate(x + cellW/2, y + cellH/2);
            ctx.scale(scale, scale);
            ctx.fillStyle = `hsl(${hue}, 70%, 55%)`;
            ctx.fillRect(-cellW/2, -cellH/2, cellW, cellH);
            ctx.restore();
        }
    }
}
```

RULES:
- Define persistent state (arrays, objects) OUTSIDE draw().
- If you create arrays/collections for animation (particles, points, etc.), you MUST populate them with initial data BEFORE draw(). An empty array with no push() is a bug that renders nothing.
- draw() MUST produce visible output every frame. Do not just clear the canvas — you must draw shapes, particles, or other visual elements.
- draw() is called every frame — keep it fast.
- Use params.xxx for controllable parameters.
- Do NOT use requestAnimationFrame, imports, require, or DOM manipulation.
- Do NOT redeclare ctx, canvas, width, height, params, inputs.
""",
    "js_module": """You are a JavaScript Creative Coder writing an ES module.
If you need a library (D3, Particle.js, etc.), IMPORT it from a CDN URL (esm.sh or unpkg).

REQUIRED CONTRACT:
export default function init(canvas, width, height, params) {
    // canvas: HTMLCanvasElement (you own it)
    // Use canvas.getContext('2d') or canvas.getContext('webgl2') as needed
    // Return a frame update function

    const ctx = canvas.getContext('2d');

    return (time, inputs) => {
        // Draw each frame. time = seconds since start.
        ctx.clearRect(0, 0, width, height);
        // your creative code here
    };
}

EXAMPLE — Canvas confetti:
```javascript
import confetti from 'https://esm.sh/canvas-confetti@1.6.0';

export default function init(canvas, width, height, params) {
    const myConfetti = confetti.create(canvas, { resize: true });

    return (time, inputs) => {
        if (Math.random() < 0.05) {
            myConfetti({ particleCount: 10, spread: 70, origin: { y: 0.6 } });
        }
    };
}
```

EXAMPLE — Pure canvas particles:
```javascript
export default function init(canvas, width, height, params) {
    const ctx = canvas.getContext('2d');
    const particles = [];
    for (let i = 0; i < 100; i++) {
        particles.push({ x: Math.random() * width, y: Math.random() * height, vx: 0, vy: 0 });
    }
    return (time, inputs) => {
        ctx.fillStyle = 'rgba(0,0,0,0.05)';
        ctx.fillRect(0, 0, width, height);
        for (const p of particles) {
            p.x += p.vx; p.y += p.vy;
            p.vx += (Math.random() - 0.5) * 0.5;
            p.vy += (Math.random() - 0.5) * 0.5;
            p.vx *= 0.98; p.vy *= 0.98;
            if (p.x < 0) p.x = width; if (p.x > width) p.x = 0;
            if (p.y < 0) p.y = height; if (p.y > height) p.y = 0;
            ctx.beginPath();
            ctx.arc(p.x, p.y, 2, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(255,255,255,0.6)';
            ctx.fill();
        }
    };
}
```

RULES:
- ALWAYS export default function init(canvas, width, height, params)
- ALWAYS return a (time, inputs) => { ... } function
- You can import from CDN URLs (esm.sh, unpkg, cdnjs)
- OUTPUT ONLY THE JAVASCRIPT CODE (full ES module). No markdown fences.
""",
    "p5": """You are a P5.js Generative Artist.
Write a P5.js Instance Mode sketch as an ES module.

REQUIREMENTS:
1. IMPORT: import p5 from 'https://esm.sh/p5@1.9.0';
2. EXPORT: export default function init(canvas, width, height, params)
3. Create a new p5 instance attached to canvas.parentElement (or handle manually).
4. RETURN: a (time, inputs) => { ... } function OR a cleanup function.
5. Use sketch.clear() in draw loop for transparent background (layering).

CODE TEMPLATE:
```javascript
import p5 from 'https://esm.sh/p5@1.9.0';

export default function init(canvas, width, height, params) {
    let myP5 = new p5((s) => {
        s.setup = () => {
            s.createCanvas(width, height, s.WEBGL, canvas);
            s.noStroke();
        };
        s.draw = () => {
            s.clear();
            let t = s.millis() / 1000;
            // Your creative code here
        };
    });

    return (time, inputs) => {
        // p5 runs its own loop, but you can update params here
    };
}
```

OUTPUT ONLY THE JAVASCRIPT CODE (full ES module). No markdown fences.
""",
    "html_video": """Output either:
- the string "webcam" (to use getUserMedia), OR
- a URL string to a video file/stream.

No code fences, no extra text.
""",
    "taichi": """Use engine js_module for taichi-based nodes. Output an ESM module as described in js_module template.
""",
    "tensorflow_js": """Use engine js_module for TFJS/MediaPipe nodes. Output an ESM module as described in js_module template.
""",
    "mediapipe": """Use engine js_module for MediaPipe nodes. Output an ESM module as described in js_module template.
""",
}

# Engine template aliases
ENGINE_TEMPLATES["p5js"] = ENGINE_TEMPLATES["p5"]


# -----------------------------------------------------------------------------
# Mason Agent
# -----------------------------------------------------------------------------

class MasonAgent:
    def __init__(self, model: str = None, max_retries: int = 2):
        self.model = model or MODEL_NAME_CODING
        self.ollama_url = OLLAMA_URL
        self.max_retries = max_retries

    # -------------------------------------------------------------------------
    # Main API
    # -------------------------------------------------------------------------

    def generate_node_code(self, nodes: List[NodeTensor]) -> List[NodeTensor]:
        """Generate code for nodes using LLM (qwen2.5-coder or configured model)."""
        updated_nodes: List[NodeTensor] = []

        for node in nodes:
            engine = (node.engine or "").strip()
            node.meta = node.meta or {}
            node.parameters = node.parameters or {}
            node.input_nodes = node.input_nodes or []

            # Special case: html_video is just a string config
            if engine == "html_video":
                node.code_snippet = self._generate_html_video_snippet(node)
                node.mason_approved = True
                node.validation_errors = []
                updated_nodes.append(node)
                continue

            category = _meta_attr(node.meta, "category", "unknown")

            # Get semantic keywords if available (from SemanticReasoner)
            keywords = getattr(node, 'keywords', []) or []
            if not keywords and isinstance(node.meta, dict):
                keywords = node.meta.get('keywords', [])

            # PRIORITY 1: Check for pre-defined code templates FIRST (Task 4 requirement)
            # This ensures the 6 predefined node types are ALWAYS used instead of LLM
            predef_key = self._find_predefined_category(category)

            if predef_key:
                predef = PREDEFINED_CODE[predef_key]

                # Handle aliases - follow the alias chain
                while predef.get("alias_of"):
                    alias_target = predef["alias_of"]
                    if alias_target in PREDEFINED_CODE:
                        predef = PREDEFINED_CODE[alias_target]
                    else:
                        break

                predef_code = predef.get("code", "")
                predef_engine = predef.get("engine", "glsl")

                # ALWAYS use the predefined template's engine — it knows what executor
                # the code was written for (e.g. particle_node → canvas2d, not three_js)
                node.engine = predef_engine
                engine = predef_engine

                # Convert NodeMeta object to dict if needed (so we can add extra fields)
                if not isinstance(node.meta, dict):
                    # Convert NodeMeta dataclass to dict
                    if hasattr(node.meta, '__dataclass_fields__'):
                        node.meta = {
                            'concept_id': getattr(node.meta, 'concept_id', ''),
                            'label': getattr(node.meta, 'label', ''),
                            'level': getattr(node.meta, 'level', 'surface'),
                            'modality': getattr(node.meta, 'modality', 'texture'),
                            'domain': getattr(node.meta, 'domain', 'visual'),
                            'description': getattr(node.meta, 'description', ''),
                            'category': getattr(node.meta, 'category', 'process'),
                            'role': getattr(node.meta, 'role', 'process'),
                        }

                # Apply predefined parameters to node (with UI metadata)
                if predef.get("parameters"):
                    # Convert parameters to simple dict for node.parameters
                    for param_name, param_info in predef["parameters"].items():
                        if param_name not in node.parameters:
                            if isinstance(param_info, dict):
                                node.parameters[param_name] = param_info.get("default", 0)
                            else:
                                node.parameters[param_name] = param_info
                    # Store UI metadata in node.meta for frontend
                    node.meta["parameter_ui"] = predef["parameters"]

                # Mark as predefined for frontend
                node.meta["is_predefined"] = True
                node.meta["predefined_type"] = predef_key
                # Do NOT override category/label — respect the semantic reasoner's output
                if predef.get("is_source"):
                    node.meta["is_source"] = True
                if predef.get("z_layer_effect"):
                    node.meta["z_layer_effect"] = True
                if predef.get("manifest"):
                    node.meta["manifest"] = predef["manifest"]

                # Wrap and validate (skip validation for non-GLSL engines like html_video)
                if predef_engine in ("html_video", "canvas2d", "p5"):
                    node.code_snippet = predef_code
                    node.mason_approved = True
                    node.validation_errors = []
                    print(f"  [MASON] {node.id} (category='{category}'): Using PREDEFINED template '{predef_key}' ✓")
                    updated_nodes.append(node)
                    continue

                wrapped_code = self._wrap_code(node, predef_code)
                errors = self._node_validate(node, wrapped_code)

                if not errors:
                    node.code_snippet = wrapped_code
                    node.mason_approved = True
                    node.validation_errors = []
                    print(f"  [MASON] {node.id} (category='{category}'): Using PREDEFINED template '{predef_key}' ✓")
                    updated_nodes.append(node)
                    continue
                else:
                    print(f"  [MASON] {node.id}: Predefined template validation failed: {errors[:2]}, falling back to LLM")

            # ----- LLM code generation -----
            template = ENGINE_TEMPLATES.get(engine, ENGINE_TEMPLATES.get("glsl", ""))
            prompt = self._build_prompt(node, template)
            print(f"  [MASON] {node.id} (category='{category}'): {engine} → {self.model}")

            last_errors: List[str] = []
            ok = False
            final_code = ""

            for attempt in range(1, self.max_retries + 1):
                if attempt == 1:
                    raw = self._call_llm(prompt)
                else:
                    repair_prompt = self._build_repair_prompt(node, template, final_code, last_errors)
                    raw = self._call_llm(repair_prompt)

                code = self._extract_code(raw)
                code = self._clean_llm_output(node, code)

                code = self._wrap_code(node, code)


                errors = self._node_validate(node, code)

                # Also check that nodes with inputs actually use them
                if not errors:
                    input_errors = self._validate_input_usage(node, code)
                    if input_errors:
                        errors = input_errors

                if not errors:
                    ok = True
                    final_code = code
                    last_errors = []
                    break

                final_code = code
                last_errors = errors
                print(f"  [MASON] {node.id} attempt {attempt}/{self.max_retries} failed: {errors[0][:120]}")

            # FALLBACK: If primary model failed, retry with llama3.2
            if not ok:
                fallback_model = MODEL_NAME_FALLBACK
                print(f"  [MASON] {node.id} primary failed — retrying with {fallback_model}")

                # For GLSL nodes: try llama3.2 with the full GLSL template
                # For JS nodes: try llama3.2 again with repair prompt
                fb_template = ENGINE_TEMPLATES.get(engine, ENGINE_TEMPLATES.get("glsl", ""))
                if last_errors and final_code:
                    fb_prompt = self._build_repair_prompt(node, fb_template, final_code, last_errors)
                else:
                    fb_prompt = self._build_prompt(node, fb_template, None)

                raw = self._call_llm(fb_prompt, model=fallback_model)
                code = self._extract_code(raw)
                code = self._clean_llm_output(node, code)
                code = self._wrap_code(node, code)
                errors = self._node_validate(node, code)
                if not errors:
                    errors = self._validate_input_usage(node, code)
                if not errors:
                    ok = True
                    final_code = code
                    last_errors = []
                    print(f"  [MASON] {node.id} ({engine}) -> PASS with {fallback_model}")

            # PASSTHROUGH FALLBACK: If both LLMs failed, inject a passthrough/identity node
            # so the render doesn't break with a white/black screen. This ensures visual continuity.
            if not ok:
                print(f"  [MASON] {node.id} ({engine}) -> Both LLMs failed. Injecting passthrough rendering.")
                passthrough_code = self._generate_passthrough_code(node, engine)
                if passthrough_code:
                    final_code = passthrough_code
                    node.mason_approved = True  # Mark as passthrough so UI knows to show it differently
                    node.is_passthrough = True  # Flag for UI
                    last_errors = ["[PASSTHROUGH] LLM generation failed; rendering input without modification"]
                    ok = True
                    print(f"  [MASON] {node.id} ({engine}) -> PASSTHROUGH activated (will render without modification)")

            node.code_snippet = final_code
            node.mason_approved = ok
            node.validation_errors = last_errors

            # Post-generation: validate and reconcile uniforms with parameters
            if ok and final_code and not getattr(node, 'is_passthrough', False):
                validator = get_uniform_validator()
                validation_result = validator.validate_and_reconcile(
                    final_code, node.parameters or {}, engine
                )
                if validation_result["needs_fix"]:
                    node.parameters = validation_result["fixed_params"]
                    print(f"  [MASON] {node.id} uniform validation: {validation_result['analysis']}")

            if ok:
                if getattr(node, 'is_passthrough', False):
                    print(f"  [MASON] {node.id} (category='{category}', {engine}) -> PASSTHROUGH")
                else:
                    print(f"  [MASON] {node.id} (category='{category}', {engine}) -> PASS")
            else:
                print(f"  [MASON] {node.id} (category='{category}', {engine}) -> FAIL after all attempts")
            updated_nodes.append(node)

        return updated_nodes

    def retry_passthrough_node(self, node: NodeTensor, runtime_error: str, engine: str = None) -> bool:
        """Retry a failed passthrough node with runtime error context.
        
        This is called by RuntimeInspector when a passthrough node fails at runtime.
        We clear the passthrough flag and attempt to regenerate using the error as context.
        
        Args:
            node: The node to retry
            runtime_error: The actual error message from the browser
            engine: Override engine if provided
            
        Returns:
            True if regeneration succeeded, False if it failed
        """
        if engine is None:
            engine = getattr(node, 'engine', 'glsl')
        
        node_id = node.id
        print(f"\n[MASON] PASSTHROUGH RETRY: {node_id} (engine={engine})")
        print(f"  Runtime error: {runtime_error[:120]}")
        
        # Clear passthrough flag to allow normal retry flow
        node.is_passthrough = False
        node.mason_approved = False
        node.validation_errors = []
        node.code_snippet = ""
        
        category = _meta_attr(node.meta, "category", "unknown")
        template = ENGINE_TEMPLATES.get(engine, ENGINE_TEMPLATES.get("glsl", ""))
        
        # Build repair prompt with runtime error context
        prompt = self._build_repair_prompt(node, template, node.code_snippet, [runtime_error])
        print(f"  Attempting LLM fix with error context...")
        
        # Try both models with error context (like normal flow)
        last_errors = [runtime_error]
        ok = False
        final_code = ""
        
        for model_idx, model in enumerate([MODEL_NAME_CODING, MODEL_NAME_FALLBACK]):
            raw = self._call_llm(prompt, model=model)
            code = self._extract_code(raw)
            code = self._clean_llm_output(node, code)
            code = self._wrap_code(node, code)
            errors = self._node_validate(node, code)
            
            if not errors:
                errors = self._validate_input_usage(node, code)
            
            if not errors:
                ok = True
                final_code = code
                print(f"  [MASON] {node_id} retry SUCCESS with {model}")
                break
            else:
                print(f"  [MASON] {node_id} retry attempt {model_idx + 1} failed: {errors[0][:80]}")
                last_errors = errors
        
        if ok:
            node.code_snippet = final_code
            node.mason_approved = True
            node.validation_errors = []
            node.is_passthrough = False
            print(f"  [MASON] {node_id} RETRY SUCCESSFUL - code regenerated from passthrough")
            return True
        else:
            # Still failing after retry - restore passthrough for now
            print(f"  [MASON] {node_id} retry FAILED - restoring passthrough")
            node.is_passthrough = True
            node.mason_approved = True
            node.validation_errors = last_errors
            node.code_snippet = self._generate_passthrough_code(node, engine)
            return False

    def _generate_passthrough_code(self, node: NodeTensor, engine: str) -> Optional[str]:
        """Generate simple passthrough code that renders without modification.
        
        This is used when both LLMs fail, to prevent white/black screens.
        The passthrough simply outputs input or renders a basic shape.
        """
        if engine == "glsl":
            # Passthrough GLSL: output the input texture or a simple color
            code = """void main() {
    vec2 uv = v_uv;
    if (length(u_input_nodes) > 0) {
        fragColor = texture(u_input0, uv);
    } else {
        fragColor = vec4(uv, 0.5, 1.0);
    }
}"""
            return self._wrap_code(node, code)
        
        elif engine == "canvas2d":
            # Passthrough Canvas2D: draw a simple grid or gradient
            code = """function draw(ctx, width, height, time, inputs, params) {
    // Passthrough: draw a simple animated gradient
    const gradient = ctx.createLinearGradient(0, 0, width, height);
    const hue = (time * 30 % 360);
    gradient.addColorStop(0, `hsl(${hue}, 100%, 50%)`);
    gradient.addColorStop(1, `hsl(${hue + 120}, 100%, 50%)`);
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, width, height);
    
    // Draw grid overlay
    ctx.strokeStyle = 'rgba(255,255,255,0.1)';
    ctx.lineWidth = 1;
    for (let x = 0; x < width; x += 16) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
    }
    for (let y = 0; y < height; y += 16) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
    }
}

return (time, inputs) => { draw(ctx, width, height, time, inputs, params); };"""
            return self._wrap_code(node, code)
        
        elif engine == "p5":
            # Passthrough P5.js: simple animated background
            code = """import p5 from 'https://esm.sh/p5@1.9.0';

export default function init(canvas, width, height, params) {
    let myP5 = new p5((s) => {
        s.setup = () => {
            s.createCanvas(width, height, s.WEBGL, canvas);
        };
        s.draw = () => {
            let t = s.millis() / 1000;
            let hue = (t * 30 % 360);
            s.background(hue, 100, 50);
            s.noStroke();
            s.fill(0, 0, 100, 0.5);
            s.rect(-width/2, -height/2, width, height);
        };
    });
    return (time, inputs) => {};
}"""
            return self._wrap_code(node, code)
        
        elif engine == "js_module":
            # Passthrough ESM: minimal export
            code = """export default async function create(ctx) {
    return {
        name: '[PASSTHROUGH]',
        description: 'LLM generation failed; passthrough module active',
        async process(input) {
            return input;
        }
    };
}"""
            return self._wrap_code(node, code)
        
        elif engine == "three_js":
            # Passthrough Three.js: simple box
            code = """const THREE = require('three');

return function init(scene, camera, params, inputs) {
    const geometry = new THREE.BoxGeometry(1, 1, 1);
    const material = new THREE.MeshBasicMaterial({color: 0x00ff00});
    const box = new THREE.Mesh(geometry, material);
    scene.add(box);
    
    return (time, inputs) => {
        box.rotation.x += 0.01;
        box.rotation.y += 0.01;
    };
};"""
            return self._wrap_code(node, code)
        
        else:
            # Generic passthrough
            code = """// Passthrough: LLM generation failed
return (time, inputs) => { };"""
            return self._wrap_code(node, code)

    def _find_predefined_category(self, category: str) -> Optional[str]:
        """Find matching predefined template key for a category.

        RULE: Only use predefined templates when the semantic reasoner's
        category is an EXACT match (or a direct alias) in PREDEFINED_CODE.
        No aggressive keyword hijacking — if the reasoner says 'tile', we
        let the LLM generate code for 'tile', not force it into 'blur_gaussian'.

        Returns:
            The key in PREDEFINED_CODE to use, or None if no match.
        """
        if not category:
            return None

        category_lower = category.lower().strip()

        # ONLY exact match in PREDEFINED_CODE keys
        if category in PREDEFINED_CODE:
            return category
        if category_lower in PREDEFINED_CODE:
            return category_lower

        # No match — LLM generates the code. No keyword hijacking.
        return None

    def _build_prompt(self, node: NodeTensor, template: str) -> str:
        meta = node.meta
        category = _meta_attr(meta, "category", "unknown")
        label = _meta_attr(meta, "label", node.id)
        desc = _meta_attr(meta, "description", "")

        # Get semantic keywords
        keywords = []
        if isinstance(meta, dict):
            keywords = meta.get('keywords', [])
        if not keywords:
            keywords = getattr(node, 'keywords', []) or []

        # Build input context
        has_inputs = bool(node.input_nodes)
        input_hint = ""
        if has_inputs and node.engine in ("glsl", "regl"):
            input_hint = "- This node receives input via texture(u_input0, uv). You MUST sample and use it."
        elif has_inputs and node.engine == "canvas2d":
            input_hint = "- This node receives input via inputs.textures. You should use it."

        glsl_guidance = ""
        if node.engine in ("glsl", "regl"):
            glsl_guidance = """
IMPORTANT for GLSL:
- Use ONLY these constructor forms: vec2(x, y), vec3(r, g, b), vec4(r, g, b, a)
- Available noise functions: hash(), hash2(), hash3(), noise(), snoise(), fbm(), worley(), simplex(), perlin(), voronoi()
- Do NOT use: s_vec2(), s_vec3(), s_vec4() — only use standard vec constructors
- Arrays must be pre-populated in draw/init sections, not declared empty
"""

        canvas2d_guidance = ""
        if node.engine == "canvas2d":
            canvas2d_guidance = """
IMPORTANT for Canvas2D:
- Declare and populate arrays WITHIN the same function scope (before draw() uses them)
- Example: let particles = []; for (let i = 0; i < 10; i++) particles.push({x: Math.random() * width});
- Use ctx.beginPath(), ctx.arc(), ctx.fill(), ctx.stroke() for drawing
- Arrays cannot be empty when draw() is called
"""

        return f"""You are a Senior Graphics Engineer writing code for a node-graph visual system.

Task: Write SELF-CONTAINED code for a node named "{label}" (category: {category}).

CONTEXT:
- Node ID: {node.id}
- Engine: {node.engine}
- Visual Keywords: {', '.join(keywords) if keywords else category.replace('_', ' ')}
- Description: {desc or category.replace('_', ' ') + ' effect'}
- Parameters: {json.dumps(node.parameters, ensure_ascii=False) if node.parameters else 'none'}
{input_hint}
{glsl_guidance}{canvas2d_guidance}

REQUIREMENTS:
{template}

OUTPUT FORMAT:
Return ONLY the raw code block. No markdown fences, no explanations, no prose.
"""

    def _build_repair_prompt(self, node: NodeTensor, template: str, last_code: str, errors: List[str]) -> str:
        category = _meta_attr(node.meta, "category", "unknown")
        err_text = "\n".join(f"- {e}" for e in (errors or ["Unknown error"]))

        engine_guidance = ""
        if node.engine in ("glsl", "regl"):
            engine_guidance = """
GLSL FIX GUIDE:
1. "no matching overloaded function found" → Use built-in constructors: vec2(x,y), vec3(x,y,z), vec4(x,y,z,w)
   Avoid custom wrappers like s_vec4() — the runtime provides proper constructors.
2. "undeclared identifier" → Check function names: use hash, noise, snoise, fbm, worley, simplex, perlin, voronoi
3. "unbalanced braces" → Count all {}, [], () pairs
4. "array declared but never populated" → GLSL errors — move to JS engine if array logic needed
"""
        elif node.engine == "canvas2d":
            engine_guidance = """
Canvas2D FIX GUIDE:
1. "array declared but never populated outside draw()" → Populate arrays immediately after declaration:
   let arr = []; for (let i = 0; i < count; i++) arr.push({...});
2. "nothing will render" → Verify your draw() function calls ctx.arc(), ctx.fill(), ctx.stroke(), etc.
3. Use ctx for all drawing operations: ctx.beginPath(), ctx.arc(), ctx.fillStyle, ctx.fill()
4. Test: Does your code have at least one ctx.arc() or ctx.fillRect() call?
"""

        return f"""You are Senior Graphics Engineer writing code for a node-graph visual system. Your previous output did not validate. Fix the errors.

Node:
- id: {node.id}
- engine: {node.engine}
- category: {category}
- parameters: {json.dumps(node.parameters, ensure_ascii=False)}

ENGINE TEMPLATE:
{template}

VALIDATION ERRORS:
{err_text}

PREVIOUS CODE:
{last_code}

{engine_guidance}

CRITICAL RULES:
- For GLSL: Do NOT use s_vec2(), s_vec3(), s_vec4() wrappers — use standard GLSL vec constructors only
- For Canvas2D: Populate all arrays before draw() is called, not inside it
- Return ONLY fixed code. No explanations.
"""

    # ---------- LLM Call (Streaming) ----------

    def _stream_ollama(self, payload: dict, timeout_per_chunk: float = 120.0) -> str:
        """Stream response from Ollama, accumulating tokens.
        Aborts if no token arrives within timeout_per_chunk seconds.
        Returns partial output on stall (better than nothing)."""
        payload["stream"] = True
        accumulated = []
        resp = requests.post(
            self.ollama_url,
            json=payload,
            stream=True,
            timeout=(15, timeout_per_chunk),  # (connect, read-per-chunk)
        )
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}")

        try:
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                token = chunk.get("response", "")
                if token:
                    accumulated.append(token)
                if chunk.get("done", False):
                    break
        except requests.exceptions.ReadTimeout:
            if accumulated:
                partial = "".join(accumulated).strip()
                print(f"  [MASON] Stream stalled after {len(partial)} chars, using partial response")
                return partial
            raise

        return "".join(accumulated).strip()

    def _call_llm(self, prompt: str, model: str = None) -> str:
        """Call Ollama API with streaming."""
        model_to_use = model or self.model

        try:
            text = self._stream_ollama({
                "model": model_to_use,
                "prompt": prompt,
                "raw": False,
                "options": {
                    "temperature": 0.6,
                    "num_predict": 2048,
                }
            }, timeout_per_chunk=120.0)

            if text.strip():
                return text
        except Exception as e:
            print(f"  [MASON] LLM call failed: {e}")

        return ""

    # -------------------------------------------------------------------------
    # Extraction / Cleaning
    # -------------------------------------------------------------------------

    def _extract_code(self, text: str) -> str:
        # Prefer explicit markers if present
        m = re.search(r"BEGIN_CODE(.*?)END_CODE", text, flags=re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()

        # Otherwise strip outer fences if any
        fence = re.search(r"```(?:\w+)?\s*(.*?)```", text, flags=re.DOTALL)
        if fence:
            return fence.group(1).strip()

        return text.strip()

    def _strip_llm_artifacts(self, s: str) -> str:
        if not s:
            return ""
        s = re.sub(r"<\|[^>]*\|>", "", s)                 # <|...|>
        s = re.sub(r"<\uff5c[^>]*\uff5c>", "", s)         # <｜...｜>
        s = s.replace("\u2581", " ").replace("\uff5c", " ")
        s = s.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")

        # Strip common mason:latest training-data leaks (bare tokens in GLSL)
        # These cause "undeclared identifier" errors: Add, SESSION, etc.
        s = re.sub(r'\b(SESSION|DOCUMENT|INSTRUCTION|RESPONSE|USER|ASSISTANT|SYSTEM)\b', '', s, flags=re.IGNORECASE)
        # Strip lines that are pure prose (no GLSL/JS syntax)
        lines = s.split('\n')
        cleaned = []
        for line in lines:
            stripped = line.strip()
            # Skip lines that look like natural language (no code chars)
            if stripped and not any(c in stripped for c in '{}();=+-*/<>[],.#') and len(stripped) > 20:
                # Likely prose — skip it
                continue
            cleaned.append(line)
        s = '\n'.join(cleaned)
        return s

    def _clean_llm_output(self, node: NodeTensor, code: str) -> str:
        engine = (node.engine or "").strip()
        code = self._strip_llm_artifacts(code)

        # If the model outputs refusal/prose, treat as empty (will fail validation)
        lower = code.lower()
        if "i can't assist" in lower or "i cannot assist" in lower or "i'm sorry" in lower:
            return ""

        # Remove markdown fences if any survived
        code = re.sub(r"```[\s\S]*?```", lambda m: m.group(0).replace("```", ""), code)

        lines = code.splitlines()
        cleaned: List[str] = []

        # Shared JS body filters (engines that use new Function(), NOT ESM)
        # three_js, p5, p5js, js_module are ESM — don't strip their imports/exports
        def is_js_body_engine(e: str) -> bool:
            return e in {"webaudio", "events", "canvas2d"}

        for line in lines:
            t = line.strip()

            # Drop empty lines
            if not t:
                continue

            # Strip HTML drift
            if t.startswith("<") or "<script" in t or "</" in t:
                continue

            # For body-only JS engines, strip imports/exports/require
            if is_js_body_engine(engine):
                if re.match(r"^\s*import\s+", t):
                    continue
                if re.match(r"^\s*export\s+", t):
                    continue
                if "require(" in t:
                    continue

                # Remove redeclarations of provided vars (but NOT user-defined vars like 'state')
                if engine == "webaudio" and re.search(r"\b(const|let|var)\s+(audioCtx|params|inputs)\b", t):
                    continue
                if engine in {"events", "canvas2d"} and re.search(r"\b(const|let|var)\s+(params|inputs)\b", t):
                    continue  # Removed 'state' - users need to define it!

                # Avoid browser-only globals in webaudio init bodies
                if engine == "webaudio" and re.search(r"\b(window|document|navigator)\b", t):
                    continue

            # For GLSL, strip ONLY version/precision - keep all declarations
            if engine in {"glsl", "regl"}:
                if t.startswith("#version"):
                    continue  # Runtime handles version
                if t.startswith("precision"):
                    continue  # Runtime handles precision

                # Strip JSON training data leaks — LLM sometimes hallucinates
                # JSON objects/arrays into GLSL (e.g. {"nodes": [...]} )
                if t.startswith('{"') or t.startswith('[{"') or t.startswith('"nodes"'):
                    continue

                # Remove pointless uniform shadowing: "float u_foo = u_foo;"
                if re.match(r"^\s*float\s+(u_\w+)\s*=\s*\1\s*[;,]?\s*$", t):
                    continue  # Skip "float u_speed = u_speed;"

                # KEEP: in, out, uniform declarations - let validation decide if they're wrong
                # This prevents stripping away the entire code!
                # WebGL1 fallbacks
                line = line.replace("gl_FragColor", "fragColor")
                line = re.sub(r"texture2D\s*\(", "texture(", line)

            cleaned.append(line)

        out = "\n".join(cleaned).strip()

        # GLSL: rewrite bare param identifiers to u_<param>
        if engine in {"glsl", "regl"} and out:
            out = self._rewrite_glsl_param_idents(out, node)

        # GLSL: Sanitize any s_vec* calls back to standard constructors (GLSL ES doesn't support overloading)
        if engine in {"glsl", "regl"} and out:
            # Convert s_vec*() back to vec*() since GLSL doesn't support function overloading
            out = re.sub(r'\bs_vec4\s*\(', 'vec4(', out)
            out = re.sub(r'\bs_vec3\s*\(', 'vec3(', out)
            out = re.sub(r'\bs_vec2\s*\(', 'vec2(', out)

            # Rewrite common LLM mistakes: "color" → "col"
            out = re.sub(r'\bcolor\b', 'col', out)

            # Fix invalid function calls from reference code
            # hash3/hash2 DO exist in utilities - don't replace them
            # Only replace hashVec and similar non-standard names
            out = re.sub(r'\bhashVec\s*\(', 'hash(', out)

            # If the result was assigned to vec3/vec2 from hash, we need to fix the type
            # Pattern: vec3 h = hash(x); -> float h = hash(x);
            # Pattern: vec2 h = hash(x); -> float h = hash(x);
            out = re.sub(r'\bvec3\s+(\w+)\s*=\s*hash\s*\(', r'float \1 = hash(', out)
            out = re.sub(r'\bvec2\s+(\w+)\s*=\s*hash\s*\(', r'float \1 = hash(', out)

            # Fix non-existent noise functions (but keep fbm, snoise, noise, worley, voronoi, simplex, perlin)
            out = re.sub(r'\bridgedFBM\s*\(', 'fbm(', out)
            out = re.sub(r'\bturbulence\s*\(', 'fbm(', out)
            out = re.sub(r'\bcellular\s*\(', 'worley(', out)
            out = re.sub(r'\bPerlin\s*\(', 'perlin(', out)
            out = re.sub(r'\bSimplex\s*\(', 'simplex(', out)

            # Fix JS-style param access: params.xxx → u_xxx (common LLM mistake)
            out = re.sub(r'\bparams\.(\w+)', r'u_\1', out)
            # Also fix: params["xxx"] or params['xxx']
            out = re.sub(r'''params\[['"](\w+)['"]\]''', r'u_\1', out)

            # Fix uppercase loop variable I → i (common LLM mistake: 'I' undeclared)
            # Replace for(int I → for(int i, then I in loop body
            def _fix_uppercase_loop_vars(code):
                # First fix for-loop declarations
                code = re.sub(r'\bfor\s*\(\s*int\s+I\b', 'for (int i', code)
                code = re.sub(r'\bfor\s*\(\s*float\s+I\b', 'for (float i', code)
                # Only replace standalone uppercase I if it looks like a loop var usage
                # (next to operators like ++, --, +=, <, <=, >, >=, or as array index)
                code = re.sub(r'\bI\s*\+\+', 'i++', code)
                code = re.sub(r'\bI\s*--', 'i--', code)
                code = re.sub(r'\+\+\s*I\b', '++i', code)
                code = re.sub(r'--\s*I\b', '--i', code)
                code = re.sub(r'\bI\s*<\s*', 'i < ', code)
                code = re.sub(r'\bI\s*<=\s*', 'i <= ', code)
                code = re.sub(r'\bI\s*>\s*', 'i > ', code)
                code = re.sub(r'\bI\s*>=\s*', 'i >= ', code)
                code = re.sub(r'\bI\s*\+=', 'i +=', code)
                code = re.sub(r'\bI\s*-=', 'i -=', code)
                # float(I) → float(i)
                code = re.sub(r'\bfloat\s*\(\s*I\s*\)', 'float(i)', code)
                code = re.sub(r'\bint\s*\(\s*I\s*\)', 'int(i)', code)
                return code
            out = _fix_uppercase_loop_vars(out)

            # Transform named function signatures to void main() format
            # Handles: vec4 name_main(vec2 uv, vec4 input0) { ... return X; }
            # Also:    vec4 process(vec4 color) { ... return X; }
            if "void main" not in out:
                # Pattern 1: vec4 <name>(vec2 uv, vec4 input0)
                m = re.match(r'^\s*vec4\s+\w+\s*\(\s*vec2\s+(\w+)\s*,\s*vec4\s+(\w+)\s*\)\s*\{',
                             out, re.MULTILINE)
                if m:
                    uv_name, input_name = m.group(1), m.group(2)
                    body_start = out.find('{') + 1
                    body_end = out.rfind('}')
                    body = out[body_start:body_end].strip()
                    body = re.sub(r'\breturn\s+([^;]+);', r'fragColor = \1;', body)
                    out = f"void main() {{\n    vec2 {uv_name} = v_uv;\n    vec4 {input_name} = texture(u_input0, {uv_name});\n    {body}\n}}"

                # Pattern 2: vec4 process(vec4 color)  (legacy mason format)
                if "void main" not in out:
                    m2 = re.match(r'^\s*vec4\s+process\s*\(\s*vec4\s+(\w+)\s*\)\s*\{',
                                  out, re.MULTILINE)
                    if m2:
                        param_name = m2.group(1)
                        body_start = out.find('{') + 1
                        body_end = out.rfind('}')
                        body = out[body_start:body_end].strip()
                        body = re.sub(r'\breturn\s+([^;]+);', r'fragColor = \1;', body)
                        out = f"void main() {{\n    vec2 uv = v_uv;\n    vec4 {param_name} = texture(u_input0, uv);\n    {body}\n}}"

        return out

    def _rewrite_glsl_param_idents(self, snippet: str, node: NodeTensor) -> str:
        params = node.parameters or {}
        numeric_keys = [k for k, v in params.items() if isinstance(v, (int, float))]

        # Replace bare identifiers with u_<name> unless already u_<name>
        for k in numeric_keys:
            pattern = re.compile(rf"(?<!u_)\b{re.escape(k)}\b")
            snippet = pattern.sub(f"u_{k}", snippet)
        return snippet

    def _wrap_code(self, node: NodeTensor, code: str) -> str:
        engine = (node.engine or "").strip()
        category = _meta_attr(node.meta, "category", "unknown")

        if engine in {"glsl", "regl"}:
            # Auto-composite DISABLED - was causing type mismatch errors
            # If nodes need to sample inputs, the LLM should generate that code explicitly
            # The reference code examples should show proper input sampling patterns
            return f"// {engine.upper()} BODY | category={category} | node={node.id}\n{code}\n"

        if engine in {"js_module", "three_js", "p5", "p5js"}:
            # ESM engines: don't add comment headers that break import statements
            return code.strip() + "\n"

        # JS body engines: keep a header comment but don't add wrappers
        return f"// {engine} BODY | category={category} | node={node.id}\n{code}\n"

    def _generate_html_video_snippet(self, node: NodeTensor) -> str:
        p = node.parameters or {}
        src = p.get("src")
        if isinstance(src, str) and src.strip():
            return src.strip()
        return "webcam"

    # -------------------------------------------------------------------------
    # Input usage validation — ensures non-source nodes use their inputs
    # -------------------------------------------------------------------------

    def _validate_input_usage(self, node: NodeTensor, code: str) -> List[str]:
        """Check that nodes with input_nodes actually reference their inputs in code."""
        input_nodes = getattr(node, 'input_nodes', None) or []
        if not input_nodes:
            return []

        # Skip if node is explicitly a source (generates its own content)
        meta = node.meta or {}
        if isinstance(meta, dict) and meta.get("is_source"):
            return []

        engine = (node.engine or "").strip()
        body = code.strip()

        if engine in ("glsl", "regl"):
            if "u_input0" not in body and "texture(u_input" not in body:
                return [f"Node has {len(input_nodes)} input(s) but code never samples u_input0. "
                        f"You MUST use texture(u_input0, uv) to read the input texture and incorporate it into your output."]
        elif engine == "canvas2d":
            if "inputs" not in body:
                return [f"Node has {len(input_nodes)} input(s) but code never references 'inputs'. "
                        f"You MUST use inputs.textures or inputs.data to read upstream node data."]

        return []

    # -------------------------------------------------------------------------
    # Validation (Dispatcher)
    # -------------------------------------------------------------------------

    def _node_validate(self, node: NodeTensor, code: str) -> List[str]:
        engine = (node.engine or "").strip()

        if engine in {"glsl", "regl"}:
            return self._validate_glsl(node, code)

        if engine in {"webaudio", "events", "canvas2d"}:
            return self._validate_js_body(node, code)

        if engine in {"three_js", "p5", "p5js", "js_module", "taichi", "tensorflow_js", "mediapipe"}:
            return self._validate_js_module(node, code)

        if engine == "html_video":
            return self._validate_html_video(code)

        # Unknown engine: no validation
        return []

    # -------------------------------------------------------------------------
    # GLSL validation
    # -------------------------------------------------------------------------

    _GLSL_UTILS_FOR_VALIDATION = r"""
// Hash functions (no s_vec* overloads — GLSL ES doesn't support overloading)
float hash(float n) { return fract(sin(n) * 43758.5453123); }
float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123); }
vec3 hash3(vec3 p) {
  p = vec3(dot(p, vec3(127.1, 311.7, 74.7)),
           dot(p, vec3(269.5, 183.3, 246.1)),
           dot(p, vec3(113.5, 271.9, 124.6)));
  return -1.0 + 2.0 * fract(sin(p) * 43758.5453123);
}
vec3 hash3(vec2 p) {
  return hash3(vec3(p, 0.0));
}

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
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y) * 0.5 + 0.5;
}

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

// Worley/cellular noise
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

// Simplex-like noise alias
float simplex(vec2 p) { return fbm(p); }
float simplex(vec3 p) { return fbm(p.xy) * 0.5 + fbm(p.yz) * 0.3; }

// Common Perlin-like variants
float perlin(vec2 p) { return fbm(p); }
float perlin(vec3 p) { return fbm(p.xy); }

// Voronoi distance field
float voronoi(vec2 p) { return worley(p); }
"""

    def _validate_glsl(self, node: NodeTensor, wrapped_code: str) -> List[str]:
        errors: List[str] = []

        # Extract shader body after header comment (if any)
        lines = wrapped_code.splitlines()
        body = "\n".join([ln for ln in lines if not ln.strip().startswith("//")]).strip()

        if not body:
            return ["GLSL snippet is empty"]

        # Detect JSON training data leak in GLSL code
        if '{"' in body or '"nodes"' in body or '"category"' in body:
            return ["GLSL code contains JSON data (LLM training data leak). Regenerate."]

        # Basic structural checks
        if "#version" in body or "precision" in body:
            errors.append("GLSL BODY must not include #version or precision declarations.")
        if "gl_FragColor" in body:
            errors.append("Use fragColor (WebGL2), not gl_FragColor.")
        if "texture2D" in body:
            errors.append("Use texture(), not texture2D().")
        if "uniform " in body:
            errors.append("Do not declare uniforms in BODY; runtime injects them.")
        if not re.search(r"\bvoid\s+main\s*\(\s*\)", body):
            errors.append("Missing required `void main()` function.")

        # Parameter naming check
        params = node.parameters or {}
        for k, v in params.items():
            if isinstance(v, (int, float)):
                if re.search(rf"(?<!u_)\b{re.escape(k)}\b", body):
                    errors.append(f"Parameter `{k}` must be referenced as `u_{k}` (not `{k}`).")

        if errors:
            return errors

        # Try compiler validation if available
        glslang = shutil.which("glslangValidator")
        if not glslang:
            return []

        # Build a complete fragment shader that matches the runtime wrapper
        side_uniforms = "\n".join([f"uniform sampler2D u_{nid};" for nid in (node.input_nodes or [])])
        param_uniforms = "\n".join([f"uniform float u_{k};" for k, v in params.items() if isinstance(v, (int, float))])

        # --- AUTO-DECLARATION: scan for undeclared u_* uniforms ---
        # Build set of already-declared uniform names
        declared = {"u_time", "u_resolution", "u_input0"}
        declared.update(f"u_{nid}" for nid in (node.input_nodes or []))
        declared.update(f"u_{k}" for k in params.keys())

        # Find all u_* identifiers in body
        found_uniforms = set(re.findall(r'\bu_[a-zA-Z_][a-zA-Z0-9_]*\b', body))
        missing = found_uniforms - declared

        # Infer types for missing uniforms and auto-declare
        auto_uniforms = []
        for u in sorted(missing):
            # Heuristic: common naming patterns
            lower = u.lower()
            if any(kw in lower for kw in ("resolution", "size", "offset", "pos", "coord", "mouse")):
                auto_uniforms.append(f"uniform vec2 {u};")
            elif any(kw in lower for kw in ("color", "col", "rgb", "tint")):
                auto_uniforms.append(f"uniform vec3 {u};")
            elif any(kw in lower for kw in ("tex", "texture", "sampler", "input", "map")):
                auto_uniforms.append(f"uniform sampler2D {u};")
            else:
                # Default to float for unknowns
                auto_uniforms.append(f"uniform float {u};")
        auto_uniform_block = "\n".join(auto_uniforms)

        full = f"""#version 300 es
precision highp float;

in vec2 v_uv;
out vec4 fragColor;

uniform float u_time;
uniform vec2 u_resolution;
uniform sampler2D u_input0;

{side_uniforms}
{param_uniforms}
{auto_uniform_block}

{self._GLSL_UTILS_FOR_VALIDATION}

{body}
"""

        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "shader.frag"
            src.write_text(full, encoding="utf-8")

            # Try multiple flag combos — glslangValidator versions differ
            for flags in [
                [glslang, "--client", "opengl100", "-S", "frag", str(src)],
                [glslang, "--target-env", "opengl", "-S", "frag", str(src)],
                [glslang, "-S", "frag", str(src)],
            ]:
                proc = subprocess.run(flags, capture_output=True, text=True)
                stderr = (proc.stderr or proc.stdout or "").strip()
                # If error is about flags/env, try next combo
                if proc.returncode != 0 and ("SPIR-V" in stderr or "target-env" in stderr or "client" in stderr):
                    continue
                break  # Either success or a real shader error
            if proc.returncode != 0:
                stderr = (proc.stderr or proc.stdout or "").strip()
                msg = "\n".join(stderr.splitlines()[:20])
                return [f"GLSL compile failed:\n{msg}"]

        return []

    # -------------------------------------------------------------------------
    # JS validation (body-only)
    # -------------------------------------------------------------------------

    def _validate_js_body(self, node: NodeTensor, wrapped_code: str) -> List[str]:
        engine = (node.engine or "").strip()
        errors: List[str] = []

        # Extract body after header comment
        lines = wrapped_code.splitlines()
        body = "\n".join([ln for ln in lines if not ln.strip().startswith("//")]).strip()

        if not body:
            return [f"{engine}: snippet is empty"]

        if "<script" in body or "<html" in body:
            return [f"{engine}: contains HTML/script tags"]

        # Reject code that looks like GLSL (LLM ignored the engine template)
        glsl_indicators = ["vec4", "vec3", "vec2", "fragColor", "texture(", "void main()",
                           "#version", "#include", "precision ", "uniform ", "sampler2D"]
        glsl_count = sum(1 for g in glsl_indicators if g in body)
        if glsl_count >= 3:
            return [f"{engine}: code looks like GLSL, not JavaScript ({glsl_count} GLSL indicators found)"]

        # Check balanced braces/parens
        opens = body.count("{") + body.count("(") + body.count("[")
        closes = body.count("}") + body.count(")") + body.count("]")
        if abs(opens - closes) > 2:
            errors.append(f"{engine}: unbalanced braces/parens (opens={opens}, closes={closes})")

        # Canvas2D: check for draw() or update() function
        if engine == "canvas2d":
            has_draw = "function draw" in body or "draw =" in body or ".draw" in body
            has_update = "function update" in body or "update =" in body
            has_ctx = "ctx." in body or "ctx2d." in body
            if not has_draw and not has_update and not has_ctx:
                errors.append(f"{engine}: no draw()/update() function and no ctx usage found")

            # Detect and repair empty-collection antipattern: array declared but never populated
            # This is common for particles, cells, etc.
            empty_array_issues = self._detect_empty_array_antipattern(body)
            if empty_array_issues:
                for issue in empty_array_issues:
                    errors.append(issue)

        if errors:
            return errors

        # Build engine-specific Node.js harness
        if engine == "three_js":
            harness = self._node_harness_three(body)
        elif engine == "webaudio":
            harness = self._node_harness_webaudio(body)
        elif engine == "events":
            harness = self._node_harness_events(body)
        elif engine == "canvas2d":
            harness = self._node_harness_canvas2d(body)
        else:
            harness = self._node_harness_generic(body)

        project_root = str(Path(__file__).resolve().parent.parent.parent)

        try:
            proc = subprocess.run(
                ["node", "-e", harness],
                capture_output=True,
                text=True,
                cwd=project_root,
                timeout=10,
            )
        except Exception as e:
            return [f"{engine}: Node.js validation failed to run: {e}"]

        if proc.returncode != 0:
            stderr = (proc.stderr or proc.stdout or "").strip()
            msg = "\n".join(stderr.splitlines()[:20])
            errors.append(f"{engine}: Node.js validation failed:\n{msg}")

        return errors

    def _node_harness_three(self, body: str) -> str:
        payload = json.dumps(body)
        return f"""
const THREE = require('three');
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 1000);
const params = {{}};
const inputs = {{ textures: {{}}, data: {{ time: 0.0, mouse: [0.5,0.5] }} }};

const userBody = {payload};
// Auto-bridge: allow "function update(t, inputs) {{...}}" without requiring an explicit return.
const wrappedBody = `"use strict";
${{userBody}}

; if (typeof update === 'function') return {{ update }}; return {{}};`;

let fn;
try {{
  fn = new Function('THREE','scene','camera','params','inputs', wrappedBody);
}} catch (e) {{
  console.error(e);
  process.exit(1);
}}

let res;
try {{
  res = fn(THREE, scene, camera, params, inputs);
}} catch (e) {{
  console.error(e);
  process.exit(1);
}}

if (!res || typeof res !== 'object') res = {{}};

if (typeof res.update === 'function') {{
  try {{
    res.update(0.1, inputs);
  }} catch (e) {{
    console.error(e);
    process.exit(1);
  }}
}}

console.log('OK');
"""

    def _node_harness_webaudio(self, body: str) -> str:
        payload = json.dumps(body)
        return f"""
class MockAnalyser {{
  constructor() {{ this.frequencyBinCount = 128; this.fftSize = 256; }}
  getByteFrequencyData(arr) {{ for (let i=0;i<arr.length;i++) arr[i] = i % 256; }}
  getByteTimeDomainData(arr) {{ for (let i=0;i<arr.length;i++) arr[i] = 128; }}
  connect(dest) {{ return dest; }}
}}
class MockAudioParam {{
  constructor(v) {{ this.value = v; this.defaultValue = v; }}
  setValueAtTime() {{}}
  linearRampToValueAtTime() {{}}
  exponentialRampToValueAtTime() {{}}
}}
class MockOscillator {{
  constructor() {{ this.type = 'sine'; this.frequency = new MockAudioParam(440); this.detune = new MockAudioParam(0); }}
  connect(dest) {{ return dest; }}
  start() {{}}
  stop() {{}}
}}
class MockGain {{
  constructor() {{ this.gain = new MockAudioParam(1.0); }}
  connect(dest) {{ return dest; }}
}}
class MockBiquadFilter {{
  constructor() {{ this.type = 'lowpass'; this.frequency = new MockAudioParam(350); this.Q = new MockAudioParam(1); this.gain = new MockAudioParam(0); }}
  connect(dest) {{ return dest; }}
}}
class MockDelay {{
  constructor() {{ this.delayTime = new MockAudioParam(0); }}
  connect(dest) {{ return dest; }}
}}
class MockAudioCtx {{
  constructor() {{ this.destination = {{}}; this.sampleRate = 44100; this.currentTime = 0; }}
  createAnalyser() {{ return new MockAnalyser(); }}
  createOscillator() {{ return new MockOscillator(); }}
  createGain() {{ return new MockGain(); }}
  createBiquadFilter() {{ return new MockBiquadFilter(); }}
  createDelay() {{ return new MockDelay(); }}
  createDynamicsCompressor() {{ return {{ threshold: new MockAudioParam(-24), knee: new MockAudioParam(30), ratio: new MockAudioParam(12), attack: new MockAudioParam(0.003), release: new MockAudioParam(0.25), connect(d){{ return d; }} }}; }}
  createConvolver() {{ return {{ buffer: null, connect(d){{ return d; }} }}; }}
  createChannelSplitter() {{ return {{ connect(d){{ return d; }} }}; }}
  createChannelMerger() {{ return {{ connect(d){{ return d; }} }}; }}
  createStereoPanner() {{ return {{ pan: new MockAudioParam(0), connect(d){{ return d; }} }}; }}
  createWaveShaper() {{ return {{ curve: null, oversample: 'none', connect(d){{ return d; }} }}; }}
}}
const audioCtx = new MockAudioCtx();
const params = {{}};
const inputs = {{ textures: {{}}, data: {{ time: 0.0 }} }};

let fn;
try {{
  fn = new Function('audioCtx','params','inputs', {payload});
}} catch (e) {{
  console.error(e);
  process.exit(1);
}}

let res;
try {{
  res = fn(audioCtx, params, inputs);
}} catch (e) {{
  console.error(e);
  process.exit(1);
}}

if (!res || typeof res !== 'object') {{
  console.error('webaudio must return an object');
  process.exit(1);
}}

if (typeof res.update === 'function') {{
  try {{
    res.update(0.1, inputs);
  }} catch (e) {{
    console.error(e);
    process.exit(1);
  }}
}}

console.log('OK');
"""

    def _node_harness_events(self, body: str) -> str:
        payload = json.dumps(body)
        return f"""
const params = {{}};
const inputs = {{ textures: {{}}, data: {{ time: 0.0, mouse: [0.5,0.5], mouseDown: 0 }} }};
let fn;
try {{
  fn = new Function('params','inputs', {payload});
}} catch (e) {{
  console.error(e);
  process.exit(1);
}}
let res;
try {{
  res = fn(params, inputs);
}} catch (e) {{
  console.error(e);
  process.exit(1);
}}
if (!res || typeof res !== 'object') {{
  console.error('events must return an object');
  process.exit(1);
}}
if (typeof res.update === 'function') {{
  try {{
    res.update(0.1, inputs);
  }} catch (e) {{
    console.error(e);
    process.exit(1);
  }}
}}
console.log('OK');
"""

    def _node_harness_canvas2d(self, body: str) -> str:
        payload = json.dumps(body)
        return f"""
// Canvas2D validation harness — tracks actual draw calls
let _drawCallCount = 0;
const _DRAW_METHODS = new Set([
  'arc','fillRect','strokeRect','lineTo','moveTo','drawImage',
  'fill','stroke','putImageData','fillText','strokeText',
  'bezierCurveTo','quadraticCurveTo','rect','ellipse'
]);

const ctx2d = new Proxy({{}}, {{
  get(target, prop) {{
    if (prop === 'canvas') return {{ width: 64, height: 64 }};
    if (!(prop in target)) {{
      target[prop] = function() {{
        if (_DRAW_METHODS.has(prop)) _drawCallCount++;
      }};
    }}
    return target[prop];
  }},
  set(target, prop, value) {{
    target[prop] = value;
    return true;
  }}
}});

const canvas = {{ width: 64, height: 64 }};
const width = canvas.width;
const height = canvas.height;
const ctx = ctx2d;

const params = {{}};
const inputs = {{ textures: {{}}, data: {{ time: 0.0 }} }};

const userBody = {payload};

// Auto-bridge:
// - allow per-frame draw BODY by wrapping it into an update()
// - allow defining draw() or update() without requiring explicit return
const wrappedBody = `"use strict";
${{userBody}}

;
` +
  `if (typeof draw === 'function') {{ return {{ update: (t, inputs) => draw(ctx, width, height, t, inputs, params) }}; }}
` +
  `if (typeof update === 'function') {{ return {{ update }}; }}
` +
  `return {{ update: (t, inputs) => {{ ${{userBody}} }} }};`;

let fn;
try {{
  fn = new Function('ctx2d','canvas','width','height','ctx','params','inputs', wrappedBody);
}} catch (e) {{
  console.error(e);
  process.exit(1);
}}

let res;
try {{
  res = fn(ctx2d, canvas, width, height, ctx, params, inputs);
}} catch (e) {{
  console.error(e);
  process.exit(1);
}}

if (!res || typeof res !== 'object') res = {{}};

if (typeof res.update === 'function') {{
  try {{
    res.update(0.1, inputs);
  }} catch (e) {{
    console.error(e);
    process.exit(1);
  }}
}}

if (_drawCallCount === 0) {{
  console.error('canvas2d: draw() produced no visible output (zero draw calls detected)');
  process.exit(1);
}}

console.log('OK');
"""

    def _node_harness_generic(self, body: str) -> str:
        payload = json.dumps(body)
        return f"""
const params = {{}};
const inputs = {{ textures: {{}}, data: {{}} }};
let fn;
try {{
  fn = new Function('params','inputs', {payload});
}} catch (e) {{
  console.error(e);
  process.exit(1);
}}
try {{
  fn(params, inputs);
}} catch (e) {{
  console.error(e);
  process.exit(1);
}}
console.log('OK');
"""

    # -------------------------------------------------------------------------
    # JS module validation (syntax-only + contract hints)
    # -------------------------------------------------------------------------

    def _detect_empty_array_antipattern(self, code: str) -> List[str]:
        """Detect canvas2d pattern: array declared empty but never populated.
        
        Returns a list of error messages if the pattern is found.
        """
        errors = []
        
        # Find array declarations at top level (outside draw/update functions)
        lines = code.splitlines()
        in_draw = False
        pre_draw_lines = []
        
        for line in lines:
            if "function draw" in line or "draw =" in line or "function update" in line or "update =" in line:
                in_draw = True
            if not in_draw:
                pre_draw_lines.append(line)
        
        pre_draw_code = "\n".join(pre_draw_lines)
        
        # Look for empty array declarations: = [] or = new Array()
        empty_decls = re.findall(r'(let|var|const)\s+(\w+)\s*=\s*(?:\[\]|new\s+Array\s*\(\s*\))', pre_draw_code)
        
        if not empty_decls:
            return []
        
        # For each empty array, check if it's ever populated
        for _, arr_name in empty_decls:
            # Check if array is populated (push, splice, unshift, assignment with values)
            populated = bool(re.search(
                rf'{re.escape(arr_name)}\s*\.(?:push|splice|unshift)\s*\(',
                pre_draw_code
            )) or bool(re.search(
                rf'{re.escape(arr_name)}\s*=\s*\[.+\]',
                pre_draw_code
            ))
            
            if not populated:
                errors.append(
                    f"{code.count(';')} array declared but never populated: {arr_name}. "
                    f"Populate arrays immediately after declaration: {arr_name}.push(...) or {arr_name} = [...]"
                )
        
        return errors

    def _validate_js_module(self, node: NodeTensor, code: str) -> List[str]:
        errors: List[str] = []
        if not code.strip():
            return ["js_module: snippet is empty"]

        if "export default" not in code:
            errors.append("js_module: missing `export default` create() function.")
        if "function create" not in code and "export default async function" not in code:
            errors.append("js_module: expected `export default async function create(ctx) { ... }`.")

        # Syntax check with Node --check (parse only)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "mod.mjs"
            p.write_text(code, encoding="utf-8")
            proc = subprocess.run(["node", "--check", str(p)], capture_output=True, text=True)
            if proc.returncode != 0:
                msg = (proc.stderr or proc.stdout or "").strip()
                errors.append(f"js_module: syntax error:\n{msg}")

        return errors

    def _validate_html_video(self, code: str) -> List[str]:
        c = code.strip()
        if not c:
            return ["html_video: empty snippet (expected 'webcam' or URL)"]
        if c != "webcam" and not re.match(r"^(https?://|/|\.{0,2}/)", c):
            return ["html_video: expected 'webcam' or a URL/path string"]
        return []
