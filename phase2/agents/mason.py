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
from typing import Dict, List, Optional, Any

from phase2.data_types import NodeTensor
from config import EFFECTIVE_MODEL_CODING, EFFECTIVE_MODEL_FALLBACK, EFFECTIVE_MODEL_CODING, EFFECTIVE_MODEL_FALLBACK
from phase2.agents.uniform_validator import get_uniform_validator
from phase2.aider_llm import get_aider_llm

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
    "p5_tracking": "Tracking",
    "face_tracking": "Face Tracking",
    "hand_tracking": "Hand Tracking",
    "body_tracking": "Body Tracking",
    "pose_tracking": "Pose Tracking",
    "color_node": "Color Node",
    "color_grade": "Color Node",
    "color_adjustment": "Color Node",
    "noise_generator": "Noise Generator",
    "noise_perlin": "Noise Generator",
    "noise_worley": "Noise Generator",
    "noise_simplex": "Noise Generator",
    "noise_fbm": "Noise Generator",
    "blend_node": "Blend Node",
    "layer_blend": "Blend Node",
    "blend": "Blend Node",
    "particle_node": "Particle Node",
    "particles": "Particle Node",
    "particle_system": "Particle Node",
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
            "source_type": {"type": "dropdown", "default": 0,
                           "options": [{"label": "Webcam", "value": 0}, {"label": "File", "value": 1}, {"label": "URL", "value": 2}], "ui": "dropdown"},
            "src": {"type": "string", "default": "", "ui": "file_picker"},
            "flip_horizontal": {"type": "bool", "default": False, "ui": "toggle"},
            "playback_rate": {"type": "float", "default": 1.0, "range": [0.1, 4.0], "ui": "slider"},
        },
        "manifest": {"semantic": "IMAGE_RGBA", "dynamic": True}
    },

    "audio_input": {
        "engine": "webaudio",
        "is_source": True,
        "code": """// Audio input: microphone, audio file, or system audio -> FFT data
// NOTE: No document/window refs — WebAudioExecutor._cleanBody strips them
const analyser = audioCtx.createAnalyser();
analyser.fftSize = params.fft_size || 256;
analyser.smoothingTimeConstant = params.smoothing || 0.8;
const dataArray = new Uint8Array(analyser.frequencyBinCount);

const gainNode = audioCtx.createGain();
gainNode.gain.value = params.gain || 1.0;
gainNode.connect(analyser);

let sourceNode = null;
let bufferSource = null;
const srcType = params.source_type || 'microphone';

if (srcType === 'microphone') {
    navigator.mediaDevices.getUserMedia({ audio: true })
        .then(function(stream) {
            sourceNode = audioCtx.createMediaStreamSource(stream);
            sourceNode.connect(gainNode);
            console.log('[AudioInput] Microphone connected');
        })
        .catch(function(e) { console.warn('[AudioInput] Mic error:', e.message); });
} else if (srcType === 'audio_file' || srcType === 1) {
    // File loading via fetch + decodeAudioData (no DOM elements needed)
    var src = params.src || '';
    if (src) {
        console.log('[AudioInput] Loading audio file:', src);
        fetch(src)
            .then(function(r) { return r.arrayBuffer(); })
            .then(function(buf) { return audioCtx.decodeAudioData(buf); })
            .then(function(decoded) {
                bufferSource = audioCtx.createBufferSource();
                bufferSource.buffer = decoded;
                bufferSource.loop = true;
                bufferSource.connect(gainNode);
                gainNode.connect(audioCtx.destination);
                bufferSource.start(0);
                console.log('[AudioInput] Audio file playing, duration:', decoded.duration.toFixed(1) + 's');
            })
            .catch(function(e) { console.warn('[AudioInput] File load error:', e.message); });
    } else {
        console.warn('[AudioInput] No src provided. Use the file picker in the node parameters to select an audio file.');
    }
} else if (srcType === 'system') {
    navigator.mediaDevices.getDisplayMedia({ audio: true, video: false })
        .then(function(stream) {
            sourceNode = audioCtx.createMediaStreamSource(stream);
            sourceNode.connect(gainNode);
            console.log('[AudioInput] System audio connected');
        })
        .catch(function(e) { console.warn('[AudioInput] System audio error:', e.message); });
}

return {
    analyser: analyser,
    gainNode: gainNode,  // exposed so WebAudioExecutor can update gain live
    dataArray: dataArray,
    update: function() {
        analyser.getByteFrequencyData(dataArray);
        return dataArray;
    },
    dispose: function() {
        if (sourceNode) sourceNode.disconnect();
        if (bufferSource) { try { bufferSource.stop(); } catch(e) {} bufferSource.disconnect(); }
        gainNode.disconnect();
    }
};""",
        "parameters": {
            "source_type": {"type": "dropdown", "default": "microphone",
                          "options": ["microphone", "audio_file", "system"], "ui": "dropdown"},
            "src": {"type": "string", "default": "", "ui": "file_picker"},
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
    # TASK 4b: Tracking Node — real ml5.js face/hand/body detection
    # =========================================================================
    "p5_tracking": {
        "engine": "canvas2d",
        "is_source": False,
        "code": """// ml5.js v1 tracking — face, hand, or body pose. Reactive tracking_type + horizontal flip.
var ml5Ready = false;
var videoEl = null;
var detector = null;
var detections = [];
var statusMsg = 'Loading ml5.js...';
var _lastTrackingType = -1;
var _startingModel = false;

var BODY_CONNECTIONS = [
    [5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],
    [11,12],[11,13],[13,15],[12,14],[14,16],[0,1],[0,2],
    [1,3],[2,4],[0,5],[0,6]
];
var HAND_CONNECTIONS = [
    [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],
    [0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],
    [15,16],[0,17],[17,18],[18,19],[19,20],[5,9],[9,13],[13,17]
];

var ML5_CDNS = [
    'https://unpkg.com/ml5@1/dist/ml5.min.js',
    'https://cdn.jsdelivr.net/npm/ml5@1/dist/ml5.min.js'
];
var cdnIdx = 0;

function forceCpuBackend() {
    // Redirect TF.js (bundled inside ml5) to CPU backend so it does not
    // compete with our WebGL2 rendering context and trigger a GPU TDR reset.
    if (typeof tf !== 'undefined') {
        try {
            var p = tf.setBackend('cpu');
            if (p && typeof p.then === 'function') { p.catch(function(){}); }
        } catch(e) {}
    }
}

function tryLoadMl5() {
    if (typeof ml5 !== 'undefined') { ml5Ready = true; forceCpuBackend(); initWebcam(); return; }
    if (cdnIdx >= ML5_CDNS.length) { statusMsg = 'ml5 unavailable - video only'; initWebcam(); return; }
    var s = document.createElement('script');
    s.src = ML5_CDNS[cdnIdx];
    s.onload = function() { ml5Ready = true; forceCpuBackend(); initWebcam(); };
    s.onerror = function() { cdnIdx++; tryLoadMl5(); };
    document.head.appendChild(s);
}

function initWebcam() {
    if (videoEl) return;
    statusMsg = 'Starting webcam...';
    videoEl = document.createElement('video');
    videoEl.setAttribute('playsinline', ''); videoEl.setAttribute('autoplay', '');
    videoEl.muted = true; videoEl.width = 640; videoEl.height = 480;
    videoEl.style.cssText = 'position:fixed;top:-9999px;left:-9999px;width:1px;height:1px;opacity:0.01;';
    document.body.appendChild(videoEl);
    navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 640 }, height: { ideal: 480 } } })
        .then(function(stream) {
            videoEl.srcObject = stream;
            videoEl.onloadeddata = function() {
                videoEl.play()
                    .then(function() {
                        if (ml5Ready) startModel(params.tracking_type || 0);
                        else statusMsg = 'Webcam ready (waiting for ml5)';
                    })
                    .catch(function(e) { statusMsg = 'Video play error: ' + e.message; });
            };
        })
        .catch(function(e) { statusMsg = 'Webcam error: ' + e.message; });
}

function startModel(type) {
    if (_startingModel) return;
    _startingModel = true;
    if (detector && typeof detector.detectStop === 'function') {
        try { detector.detectStop(); } catch(_) {}
    }
    detector = null; detections = [];
    var maxDet = params.max_detections || 1;
    statusMsg = 'Loading ML model (10-30s)...';
    var modelResult;
    try {
        if (type === 0 || type === 'face') {
            modelResult = ml5.faceMesh({ maxFaces: maxDet });
        } else if (type === 1 || type === 'hand') {
            modelResult = ml5.handPose({ maxHands: maxDet });
        } else {
            modelResult = ml5.bodyPose('MoveNet', { maxPoses: maxDet });
        }
    } catch(e) { statusMsg = 'Model error: ' + e.message; _startingModel = false; return; }

    function onModelReady(model) {
        detector = model; _lastTrackingType = type; _startingModel = false;
        try {
            detector.detectStart(videoEl, function(results) {
                detections = results || [];
                if (detections.length > 0) statusMsg = '';
            });
            statusMsg = 'Model loaded, detecting...';
        } catch(e) { statusMsg = 'Detection error: ' + e.message; }
    }

    if (modelResult && typeof modelResult.then === 'function') {
        modelResult.then(function(m) { onModelReady(m); })
                   .catch(function(e) { statusMsg = 'Model load error: ' + e.message; _startingModel = false; });
    } else if (modelResult) {
        if (typeof modelResult.detectStart === 'function') { onModelReady(modelResult); }
        else if (modelResult.ready && typeof modelResult.ready.then === 'function') {
            modelResult.ready.then(function() { onModelReady(modelResult); });
        } else { _startingModel = false; }
    } else { _startingModel = false; }
}

tryLoadMl5();
setTimeout(function() { if (!videoEl) initWebcam(); }, 20000);

function cleanup() {
    if (detector && typeof detector.detectStop === 'function') {
        try { detector.detectStop(); } catch(_) {}
    }
    detector = null;
    if (videoEl) {
        if (videoEl.srcObject) {
            var trks = videoEl.srcObject.getTracks ? videoEl.srcObject.getTracks() : [];
            trks.forEach(function(tr) { tr.stop(); });
        }
        try { videoEl.remove(); } catch(_) {}
        videoEl = null;
    }
}

function draw(ctx, w, h, t, inp, p) {
    var flipH = p.flip_horizontal !== false;  // default true — mirrors selfie view
    var curType = (p.tracking_type !== undefined ? p.tracking_type : 0);

    // Reactive mode switch: reinit when tracking_type changes
    if (ml5Ready && videoEl && videoEl.readyState >= 2 && curType !== _lastTrackingType && !_startingModel) {
        startModel(curType);
    }

    if (videoEl && videoEl.readyState >= 2) {
        ctx.save();
        if (flipH) { ctx.translate(w, 0); ctx.scale(-1, 1); }
        ctx.drawImage(videoEl, 0, 0, w, h);
        ctx.restore();
    } else {
        ctx.fillStyle = '#111'; ctx.fillRect(0, 0, w, h);
        ctx.fillStyle = '#0f0'; ctx.font = '14px monospace'; ctx.textAlign = 'center';
        ctx.fillText(statusMsg || 'Initializing...', w / 2, h / 2);
        ctx.textAlign = 'start'; return;
    }

    if (statusMsg) {
        ctx.fillStyle = 'rgba(0,0,0,0.5)'; ctx.fillRect(0, h - 30, w, 30);
        ctx.fillStyle = '#0f0'; ctx.font = '12px monospace'; ctx.textAlign = 'center';
        ctx.fillText(statusMsg, w / 2, h - 10); ctx.textAlign = 'start';
    }

    if (!detections || !detections.length) { canvas._trackingData = []; return; }

    var showLm = p.draw_landmarks !== false && p.draw_landmarks !== 0;
    var showSk = p.draw_skeleton !== false && p.draw_skeleton !== 0;
    var minConf = p.confidence_threshold || 0.3;
    var maxDet = p.max_detections || detections.length;
    var vidW = videoEl.videoWidth || w;
    var vidH = videoEl.videoHeight || h;
    var sx = w / vidW, sy = h / vidH;

    // Flip-aware coordinate helpers
    var dispX = flipH ? function(x) { return w - x * sx; } : function(x) { return x * sx; };
    var dispY = function(y) { return y * sy; };

    for (var di = 0; di < Math.min(detections.length, maxDet); di++) {
        var det = detections[di];
        var kps = det.keypoints || [];
        if (!kps.length) continue;

        if (showSk && kps.length > 1) {
            ctx.strokeStyle = 'rgba(0, 255, 128, 0.6)'; ctx.lineWidth = 2;
            var conns = null;
            if (curType === 2 || curType === 'body') conns = BODY_CONNECTIONS;
            else if (curType === 1 || curType === 'hand') conns = HAND_CONNECTIONS;

            if (conns) {
                for (var ci = 0; ci < conns.length; ci++) {
                    var a = conns[ci][0], b = conns[ci][1];
                    if (a >= kps.length || b >= kps.length) continue;
                    var ka = kps[a], kb = kps[b];
                    if ((ka.confidence || ka.score || 1) < minConf) continue;
                    if ((kb.confidence || kb.score || 1) < minConf) continue;
                    ctx.beginPath();
                    ctx.moveTo(dispX(ka.x), dispY(ka.y));
                    ctx.lineTo(dispX(kb.x), dispY(kb.y));
                    ctx.stroke();
                }
            } else {
                var contour = [10,338,297,332,284,251,389,356,454,323,361,288,
                    397,365,379,378,400,377,152,148,176,149,150,136,172,58,
                    132,93,234,127,162,21,54,103,67,109,10];
                ctx.strokeStyle = 'rgba(0, 200, 255, 0.4)';
                for (var fi = 0; fi < contour.length - 1; fi++) {
                    var ai2 = contour[fi], bi2 = contour[fi + 1];
                    if (ai2 >= kps.length || bi2 >= kps.length) continue;
                    ctx.beginPath();
                    ctx.moveTo(dispX(kps[ai2].x), dispY(kps[ai2].y));
                    ctx.lineTo(dispX(kps[bi2].x), dispY(kps[bi2].y));
                    ctx.stroke();
                }
            }
        }

        if (showLm) {
            for (var ki = 0; ki < kps.length; ki++) {
                var kp = kps[ki];
                if ((kp.confidence || kp.score || 1) < minConf) continue;
                ctx.beginPath();
                ctx.arc(dispX(kp.x), dispY(kp.y), curType === 0 ? 1.5 : 4, 0, Math.PI * 2);
                ctx.fillStyle = 'rgba(0, 255, 128, 0.9)'; ctx.fill();
            }
        }
    }

    // Publish normalised keypoints (0-1) so downstream uniforms and particle code
    // can scale to any canvas size without knowing the video resolution.
    canvas._trackingData = detections.slice(0, maxDet).map(function(d) {
        return {
            score: d.score, box: d.box,
            keypoints: (d.keypoints || []).filter(function(kp) {
                return (kp.confidence || kp.score || 1) >= minConf;
            }).map(function(kp) {
                return { name: kp.name,
                         x: (flipH ? vidW - kp.x : kp.x) / vidW,
                         y: kp.y / vidH,
                         confidence: kp.confidence || kp.score || 1 };
            })
        };
    });
}

return {
    update: function(t, inp) { draw(ctx, width, height, t, inp, params); },
    cleanup: cleanup
};""",
        "parameters": {
            "tracking_type": {"type": "dropdown", "default": 0,
                            "options": [{"label": "Face", "value": 0}, {"label": "Hand", "value": 1}, {"label": "Body", "value": 2}], "ui": "dropdown"},
            "max_detections": {"type": "int", "default": 1, "range": [1, 4], "ui": "slider"},
            "confidence_threshold": {"type": "float", "default": 0.3, "range": [0.0, 1.0], "ui": "slider"},
            "draw_landmarks": {"type": "bool", "default": True, "ui": "toggle"},
            "draw_skeleton": {"type": "bool", "default": True, "ui": "toggle"},
            "flip_horizontal": {"type": "bool", "default": True, "ui": "toggle"},
        },
        "manifest": {"semantic": "LANDMARKS", "dynamic": True}
    },

    # =========================================================================
    # TASK 4c: Color Node with HUE PICKER (not RGB)
    # =========================================================================
    "color_node": {
        "engine": "glsl",
        "is_source": False,
        "code": """// rgb2hsv, hsv2rgb are provided by the shader prefix (GLSL_UTILS)
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

    fragColor = vec4(clamp(col.rgb, 0.0, 1.0), col.a);
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

    fragColor = vec4(clamp(result, 0.0, 1.0), base.a);
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
    # TASK 4f: Particle System Node
    # =========================================================================
    "particle_node": {
        "engine": "p5",
        "is_source": True,
        "code": """import p5 from 'https://esm.sh/p5@1.9.0';

export default function init(canvas, width, height, params) {
    let currentTime = 0;
    let currentInputs = [];
    let currentInputData = {};  // rich data: audio FFT, tracking keypoints from wired nodes
    let _audioData = null;
    let _trackingKp = null;

    function hexToRgb(hex) {
        hex = (hex || '#ffffff').replace('#', '');
        return [
            parseInt(hex.substring(0,2), 16) || 255,
            parseInt(hex.substring(2,4), 16) || 255,
            parseInt(hex.substring(4,6), 16) || 255
        ];
    }

    const MAX_COUNT = 500;
    const pool = [];
    for (let i = 0; i < MAX_COUNT; i++) {
        const angle = Math.random() * Math.PI * 2;
        pool.push({
            x: Math.random() * width, y: Math.random() * height,
            vx: Math.cos(angle) * (0.5 + Math.random()),
            vy: Math.sin(angle) * (0.5 + Math.random()),
            sizeMul: 0.3 + Math.random() * 0.7,
            phase: Math.random() * Math.PI * 2,
        });
    }

    const sketch = new p5((s) => {
        s.setup = () => {
            s.createCanvas(width, height, s.P2D, canvas);
            s.pixelDensity(1);
            s.colorMode(s.RGB, 255, 255, 255, 1.0);
            s.noLoop();
        };
        s.draw = () => { try {
            s.clear();
            const t = currentTime;
            const p = params;
            const count = Math.min(p.count || 80, MAX_COUNT);
            const baseSz = p.size || 3;
            const baseSpd = p.speed || 2;
            const sizeRandom = p.size_random !== false && p.size_random !== 0;
            const opacity = p.opacity !== undefined ? p.opacity : 0.5;
            const shape = p.shape || 0;
            const col = hexToRgb(p.color);
            const linksOn = p.links_enabled !== false && p.links_enabled !== 0;
            const linkDist = p.link_distance || 150;
            const linkCol = hexToRgb(p.link_color);
            const linkOp = p.link_opacity !== undefined ? p.link_opacity : 0.4;

            // Audio-driven multipliers — auto-activate when audio node is wired
            const szMul = _audioData ? 1.0 + _audioData.bass * 3.0 : 1.0;
            const spdMul = _audioData ? 1.0 + _audioData.mid * 2.0 : 1.0;
            const sz = baseSz * szMul;
            const spd = baseSpd * spdMul;

            // Tracking attraction target — keypoints are normalised 0-1, scale to canvas
            let attractX = null, attractY = null;
            if (_trackingKp && _trackingKp.length > 0) {
                attractX = _trackingKp[0].x * width;
                attractY = _trackingKp[0].y * height;
            }

            const dir = p.direction || 0;
            let bx = 0, by = 0;
            if (dir === 1 || dir === 'top') by = -0.3;
            else if (dir === 2 || dir === 'bottom') by = 0.3;
            else if (dir === 3 || dir === 'left') bx = -0.3;
            else if (dir === 4 || dir === 'right') bx = 0.3;

            s.noStroke();
            for (let i = 0; i < count; i++) {
                const pt = pool[i];
                // Attraction toward tracking keypoint
                if (attractX !== null) {
                    pt.vx += (attractX - pt.x) * 0.002;
                    pt.vy += (attractY - pt.y) * 0.002;
                    // Dampen velocity to avoid runaway
                    pt.vx *= 0.95; pt.vy *= 0.95;
                }
                pt.x += (pt.vx * spd + bx);
                pt.y += (pt.vy * spd + by);
                if (pt.x < -10) pt.x = width + 10;
                if (pt.x > width + 10) pt.x = -10;
                if (pt.y < -10) pt.y = height + 10;
                if (pt.y > height + 10) pt.y = -10;

                const r = sizeRandom ? sz * pt.sizeMul : sz;
                const alpha = opacity * (0.6 + 0.4 * Math.sin(t * 2 + pt.phase));
                s.fill(col[0], col[1], col[2], alpha);

                if (shape === 1 || shape === 'edge') {
                    s.rect(pt.x - r, pt.y - 0.5, r * 2, 1);
                } else if (shape === 2 || shape === 'triangle') {
                    s.triangle(pt.x, pt.y - r, pt.x - r * 0.866, pt.y + r * 0.5, pt.x + r * 0.866, pt.y + r * 0.5);
                } else {
                    s.ellipse(pt.x, pt.y, r * 2, r * 2);
                }
            }

            if (linksOn) {
                const maxD2 = linkDist * linkDist;
                for (let i = 0; i < count; i++) {
                    for (let j = i + 1; j < count; j++) {
                        const dx = pool[i].x - pool[j].x;
                        const dy = pool[i].y - pool[j].y;
                        const d2 = dx * dx + dy * dy;
                        if (d2 < maxD2) {
                            const a = linkOp * (1 - Math.sqrt(d2) / linkDist);
                            s.stroke(linkCol[0], linkCol[1], linkCol[2], a);
                            s.strokeWeight(0.5);
                            s.line(pool[i].x, pool[i].y, pool[j].x, pool[j].y);
                            s.noStroke();
                        }
                    }
                }
            }
        } catch (_err) { /* mute draw errors */ } };
    });
    const updateFn = (time, inputs, inputDataMap) => {
        currentTime = time; currentInputs = inputs;
        currentInputData = inputDataMap || {};
        _audioData = null; _trackingKp = null;
        for (const data of Object.values(currentInputData)) {
            if (data && data.type === 'audio' && data.audioData) _audioData = data.audioData;
            if (data && data.trackingData && data.trackingData[0]) _trackingKp = data.trackingData[0].keypoints;
        }
        sketch.redraw();
    };
    return {
        update: updateFn,
        dispose: () => sketch.remove(),
    };
}""",
        "parameters": {
            "count": {"type": "int", "default": 80, "range": [10, 500], "ui": "slider"},
            "color": {"type": "color", "default": "#ffffff", "ui": "color_picker"},
            "size": {"type": "float", "default": 3.0, "range": [1.0, 20.0], "ui": "slider"},
            "size_random": {"type": "bool", "default": True, "ui": "toggle"},
            "opacity": {"type": "float", "default": 0.5, "range": [0.0, 1.0], "ui": "slider"},
            "speed": {"type": "float", "default": 2.0, "range": [0.1, 10.0], "ui": "slider"},
            "shape": {"type": "dropdown", "default": 0,
                     "options": [{"label": "Circle", "value": 0}, {"label": "Edge", "value": 1},
                                {"label": "Triangle", "value": 2}, {"label": "Polygon", "value": 3},
                                {"label": "Star", "value": 4}], "ui": "dropdown"},
            "links_enabled": {"type": "bool", "default": True, "ui": "toggle"},
            "link_distance": {"type": "float", "default": 150, "range": [50, 300], "ui": "slider"},
            "link_color": {"type": "color", "default": "#ffffff", "ui": "color_picker"},
            "link_opacity": {"type": "float", "default": 0.4, "range": [0.0, 1.0], "ui": "slider"},
            "direction": {"type": "dropdown", "default": 0,
                         "options": [{"label": "None", "value": 0}, {"label": "Up", "value": 1},
                                    {"label": "Down", "value": 2}, {"label": "Left", "value": 3},
                                    {"label": "Right", "value": 4}], "ui": "dropdown"},
        },
        "manifest": {"semantic": "IMAGE_RGBA", "dynamic": True}
    },

    "particles": {"engine": "p5", "is_source": True, "alias_of": "particle_node"},
    "particle_system": {"engine": "p5", "is_source": True, "alias_of": "particle_node"},

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
- uniform sampler2D u_input0;              // main input texture (first connected input node)
- uniform sampler2D u_<inputNodeId>;       // side input textures — also aliased as u_input1, u_input2, ... by index
- uniform float u_<paramName>;             // numeric params from node.parameters
// You may use u_input0, u_input1, u_input2... as convenient sequential aliases for connected inputs.

Utility functions available (already in shader prefix — do NOT redefine):
- float hash(float n)        // hash a single float
- float hash(vec2 p)         // hash a 2D point
- vec2  hash2(vec2 p)        // 2D -> 2D hash
- vec3  hash3(vec3 p)        // 3D -> 3D hash
- vec3  hash3(vec2 p)        // 2D -> 3D hash
- float noise(vec2 x)        // Perlin noise — ONLY takes vec2
- float snoise(vec2 p)       // signed noise [-1,1] — ONLY vec2
- float snoise(vec3 p)       // signed noise from vec3 (uses .xy+.z internally)
- float fbm(vec2 x)          // fractal Brownian motion — ONLY vec2
- float worley(vec2 p)       // cellular/Worley noise — ONLY vec2
- float voronoi(vec2 p)      // Voronoi distance field — ONLY vec2
- float simplex(vec2 p)      // simplex alias — ONLY vec2
- float simplex(vec3 p)      // simplex from vec3
- float perlin(vec2 p)       // Perlin alias — ONLY vec2
- float perlin(vec3 p)       // Perlin from vec3

Color space conversions (available — already in shader prefix):
- vec3 rgb2hsv(vec3 c)   // RGB -> HSV
- vec3 hsv2rgb(vec3 c)   // HSV -> RGB
- vec3 hsl2rgb(vec3 c)   // HSL -> RGB

FUNCTIONS THAT DO NOT EXIST (never call these):
- noise(float), noise(vec3), fbm(vec3), worley(vec3), voronoi(vec3)
- ridgedFBM(), turbulence()
- s_vec2(), s_vec3(), s_vec4()

RESERVED WORDS — never use as variable or function names:
  smooth, sample, input, output, filter, common, flat, buffer, shared

STRICT RULES:
- DO NOT include: #version, precision, in/out declarations, uniform declarations, layout(), gl_FragColor, texture2D.
- DO use: texture(u_input0, uv) and write to fragColor.
- CRITICAL: NEVER declare local variables with uniform names (NO "float u_speed = u_speed;"). Uniforms are already declared - use them directly!
- IMPORTANT: NEVER reference parameter names without the u_ prefix (use u_scale, not scale).
- IMPORTANT: Use standard GLSL constructors: vec2(x,y), vec3(x,y,z), vec4(x,y,z,w) — NOT s_vec* wrappers.
- vec3 takes 3 floats max. vec4 takes 4 floats max. Do NOT pass extra arguments.
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
    "three_js": """You are a Three.js Creative Coder writing a code body for a Three.js node.

EXECUTION CONTEXT:
The runtime provides: THREE, scene, camera, params, inputs.
You do NOT create a renderer, scene, or camera — they already exist.
You add objects to `scene`, set up lighting, and return an update function.

REQUIREMENTS:
1. Add lights and 3D objects to the provided `scene`.
2. Use the provided `camera` — do NOT create a new camera.
3. RETURN an object with an `update(time, inputs)` method called every frame.
4. Use `params.*` for all controllable values.
5. Use `inputs.textures` to access upstream node textures as THREE.CanvasTexture.

CODE TEMPLATE:
```javascript
// THREE, scene, camera, params, inputs are provided — do NOT redeclare them

// Add lights
scene.add(new THREE.DirectionalLight(0xffffff, 1));
scene.add(new THREE.AmbientLight(0x404040));

// Create geometry and material
const geometry = new THREE.TorusGeometry(1.5, 0.5, 32, 64);
const material = new THREE.MeshStandardMaterial({
    color: params.color || 0x44aaff,
    metalness: params.metalness || 0.8,
    roughness: params.roughness || 0.2
});
const mesh = new THREE.Mesh(geometry, material);
scene.add(mesh);

// Return update function — called every frame
return {
    update(time, inputs) {
        mesh.rotation.x = time * 0.5;
        mesh.rotation.y = time * 0.7;
    }
};
```

CRITICAL RULES:
- NEVER create a WebGLRenderer, Scene, or PerspectiveCamera — they are provided.
- NEVER use import, export, require() — write a plain code body only.
- NEVER use TextureLoader, fetch, or any external file path ('path/to/...', '.jpg', '.png').
- ALL textures MUST be procedurally generated using CanvasTexture or DataTexture with math/noise.
- For CanvasTexture: create an OffscreenCanvas(256, 256), draw on it, wrap in new THREE.CanvasTexture(offscreen).
- NEVER use document.createElement — use new OffscreenCanvas(w, h) for any offscreen canvas needs.
- NEVER redeclare THREE, scene, camera, or params — they are already in scope.
- Use params.* for ALL magic numbers that should be controllable.

OUTPUT ONLY THE JAVASCRIPT BODY (no imports, no exports, no function wrapper). No markdown fences.
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
    "p5": """You are a P5.js Generative Artist writing instance-mode p5 sketches.

EXACT BOILERPLATE (copy this structure every time):

import p5 from 'https://esm.sh/p5@1.9.0';

export default function init(canvas, width, height, params) {
    let currentTime = 0;
    let currentInputs = [];

    const sketch = new p5((s) => {
        s.setup = () => {
            s.createCanvas(width, height, s.P2D, canvas);
            s.pixelDensity(1);
            s.colorMode(s.HSB, 360, 100, 100, 100);
            s.noLoop();
        };
        s.draw = () => {
            try {
                s.clear();
                const t = currentTime;
                const inputs = currentInputs;
                // --- YOUR DRAWING CODE HERE ---
                // All p5 functions use s. prefix: s.ellipse(), s.fill(), s.bezier(), etc.
                // Access upstream textures via: inputs[0], inputs[1], etc. (these are canvas elements)
            } catch (_err) { /* mute draw errors — prevents p5 from flooding the console every frame */ }
        };
    });

    return (time, inputs) => {
        currentTime = time;
        currentInputs = inputs;
        sketch.redraw();
    };
}

CRITICAL RULES:
- ALWAYS use s. prefix for ALL p5 functions: s.ellipse(), s.fill(), s.stroke(), s.push(), s.pop(), s.translate(), s.rotate(), s.bezier(), s.beginShape(), s.vertex(), s.endShape(), s.rect(), s.arc(), s.noFill(), s.noStroke(), s.strokeWeight(), s.scale(), s.TWO_PI, s.PI, s.cos(), s.sin(), s.map(), s.lerp(), s.noise(), s.random()
- ALWAYS use s.clear() at the start of s.draw (transparent background for layering)
- ALWAYS use s.pixelDensity(1) in setup — prevents devicePixelRatio scaling that breaks canvas bounds
- ALWAYS use s.noLoop() in setup — the pipeline calls sketch.redraw() each frame (driven mode, no competing RAF loop)
- In the return function: after updating state, call sketch.redraw() — this triggers one synchronous s.draw() per pipeline tick
- ALWAYS wrap the ENTIRE s.draw body in try { ... } catch (_err) {} — if s.draw throws uncaught, p5 re-throws every frame and crashes the browser
- ALWAYS use s.P2D (not s.WEBGL) as renderer
- Use currentTime (updated from the return function each pipeline frame) for time-based animation
- Use `const inputs = currentInputs;` at the top of s.draw to access upstream textures
- inputs[0], inputs[1], etc. are HTMLCanvasElement objects — composite them with s.drawingContext.drawImage(inputs[0], 0, 0, width, height)
- NEVER use s.image() with input canvases — p5.image() only accepts p5.Image/p5.Graphics, not raw HTMLCanvasElement. Always use s.drawingContext.drawImage() for input compositing.
- NEVER reference `inputs` directly from the return function scope inside s.draw — use the currentInputs closure variable
- Use params.xxx for controllable parameters
- DO NOT use requestAnimationFrame, setTimeout, or DOM manipulation
- s.drawingContext is the native Canvas2D API and is fine to use alongside p5 drawing functions
- SPATIAL PLACEMENT: Do NOT place all elements at width/2, height/2. Use semantically meaningful positions. If this node is a "planet" it should orbit at a radius from center. If this node adds "petals" they should radiate outward. If this is a "background" fill the whole canvas. Source nodes define regions; downstream nodes draw in different regions to avoid overlap.
- PERFORMANCE — PARTICLE/OBJECT ARRAYS: NEVER push() to an array inside s.draw() or inside the return update function. Arrays MUST be created once (in init/setup) with a fixed pre-allocated size (e.g., `const particles = Array.from({length: params.count || 80}, () => spawnParticle())`). In s.draw(), UPDATE existing array elements in-place — do NOT grow the array. If a particle dies, replace it with a new one at the same index: `particles[i] = spawnParticle()`. Unbounded arrays grow to tens of thousands of objects and cause 400ms+ frame times.

AVAILABLE P5 FUNCTIONS (always with s. prefix):
  Drawing: s.ellipse(x,y,w,h), s.rect(x,y,w,h), s.triangle(x1,y1,x2,y2,x3,y3)
  Curves:  s.bezier(x1,y1,cx1,cy1,cx2,cy2,x2,y2), s.curve(x1,y1,x2,y2,x3,y3,x4,y4)
  Shapes:  s.beginShape(), s.vertex(x,y), s.bezierVertex(cx1,cy1,cx2,cy2,x,y), s.endShape(s.CLOSE)
  Style:   s.fill(h,sat,bri), s.stroke(h,sat,bri), s.noFill(), s.noStroke(), s.strokeWeight(n)
  Transform: s.push(), s.pop(), s.translate(x,y), s.rotate(angle), s.scale(sx,sy)
  Math:    s.cos(a), s.sin(a), s.map(v,a,b,c,d), s.lerp(a,b,t), s.noise(x,y), s.random(lo,hi)
  Constants: s.PI, s.TWO_PI, s.HALF_PI, s.CLOSE

EXAMPLE — Flower with petals (source node, no inputs):
import p5 from 'https://esm.sh/p5@1.9.0';

export default function init(canvas, width, height, params) {
    let currentTime = 0;
    let currentInputs = [];
    const sketch = new p5((s) => {
        s.setup = () => {
            s.createCanvas(width, height, s.P2D, canvas);
            s.pixelDensity(1);
            s.colorMode(s.HSB, 360, 100, 100, 100);
            s.noLoop();
        };
        s.draw = () => {
            try {
                s.clear();
                const t = currentTime;
                const cx = width / 2;
                const cy = height / 2;
                const petalCount = params.count || 6;
                const petalLen = params.radius || width * 0.3;
                const bloom = s.map(s.sin(t * 0.5), -1, 1, 0.3, 1.0);
                for (let i = 0; i < petalCount; i++) {
                    const angle = (i / petalCount) * s.TWO_PI + t * 0.2;
                    s.push();
                    s.translate(cx, cy);
                    s.rotate(angle);
                    s.noStroke();
                    s.fill(50 + i * 30, 80, 90, 80);
                    s.beginShape();
                    s.vertex(0, 0);
                    s.bezierVertex(petalLen * 0.3 * bloom, -petalLen * 0.4 * bloom,
                                   petalLen * 0.8 * bloom, -petalLen * 0.2 * bloom,
                                   petalLen * bloom, 0);
                    s.bezierVertex(petalLen * 0.8 * bloom, petalLen * 0.2 * bloom,
                                   petalLen * 0.3 * bloom, petalLen * 0.4 * bloom,
                                   0, 0);
                    s.endShape(s.CLOSE);
                    s.pop();
                }
                s.fill(45, 90, 95);
                s.noStroke();
                s.ellipse(cx, cy, petalLen * 0.2, petalLen * 0.2);
            } catch (_err) { /* mute draw errors */ }
        };
    });
    return (time, inputs) => { currentTime = time; currentInputs = inputs; sketch.redraw(); };
}

EXAMPLE — Animated particles (source node, no inputs):
import p5 from 'https://esm.sh/p5@1.9.0';

export default function init(canvas, width, height, params) {
    let currentTime = 0;
    let currentInputs = [];
    const particles = [];
    for (let i = 0; i < 150; i++) {
        particles.push({ x: Math.random() * width, y: Math.random() * height, vx: 0, vy: 0, hue: Math.random() * 360 });
    }
    const sketch = new p5((s) => {
        s.setup = () => {
            s.createCanvas(width, height, s.P2D, canvas);
            s.pixelDensity(1);
            s.colorMode(s.HSB, 360, 100, 100, 100);
            s.noLoop();
        };
        s.draw = () => {
            try {
                s.background(0, 0, 0, 5);
                const t = currentTime;
                const speed = params.speed || 2;
                s.noStroke();
                for (const p of particles) {
                    const angle = s.noise(p.x * 0.005, p.y * 0.005, t * 0.3) * s.TWO_PI * 2;
                    p.vx += s.cos(angle) * 0.3;
                    p.vy += s.sin(angle) * 0.3;
                    p.vx *= 0.96; p.vy *= 0.96;
                    p.x += p.vx * speed; p.y += p.vy * speed;
                    if (p.x < 0) p.x = width; if (p.x > width) p.x = 0;
                    if (p.y < 0) p.y = height; if (p.y > height) p.y = 0;
                    s.fill(p.hue, 80, 90, 70);
                    s.ellipse(p.x, p.y, 4, 4);
                }
            } catch (_err) { /* mute draw errors */ }
        };
    });
    return (time, inputs) => { currentTime = time; currentInputs = inputs; sketch.redraw(); };
}

EXAMPLE — Compositing node (uses upstream input textures):
import p5 from 'https://esm.sh/p5@1.9.0';

export default function init(canvas, width, height, params) {
    let currentTime = 0;
    let currentInputs = [];
    const sketch = new p5((s) => {
        s.setup = () => {
            s.createCanvas(width, height, s.P2D, canvas);
            s.pixelDensity(1);
            s.colorMode(s.HSB, 360, 100, 100, 100);
            s.noLoop();
        };
        s.draw = () => {
            try {
                s.clear();
                const t = currentTime;
                const inputs = currentInputs;
                // Draw upstream input textures (each input is an HTMLCanvasElement)
                // Use drawingContext.drawImage — s.image() does NOT accept raw HTMLCanvasElement
                if (inputs[0]) s.drawingContext.drawImage(inputs[0], 0, 0, width, height);
                if (inputs[1]) s.drawingContext.drawImage(inputs[1], 0, 0, width, height);
                // Add own effect on top
                s.noStroke();
                s.fill(0, 0, 100, 10);
                const glow = s.map(s.sin(t), -1, 1, 0.5, 1.0);
                s.ellipse(width / 2, height / 2, width * glow * 0.3, height * glow * 0.3);
            } catch (_err) { /* mute draw errors */ }
        };
    });
    return (time, inputs) => { currentTime = time; currentInputs = inputs; sketch.redraw(); };
}

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
# Mason Debugging Tools (OpenAI function-calling format)
# -----------------------------------------------------------------------------

MASON_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "update_node_code",
            "description": "Submit new or revised code for this node. Always call compile_and_get_errors after this to check for issues.",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {"type": "string", "description": "The node ID"},
                    "new_code": {"type": "string", "description": "The complete code (raw, no markdown fences)"},
                },
                "required": ["node_id", "new_code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compile_and_get_errors",
            "description": "Compile the current code and return any errors. Returns 'Compilation successful.' if no errors.",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {"type": "string", "description": "The node ID to compile"},
                },
                "required": ["node_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ponder",
            "description": "Think step-by-step about the problem before writing or fixing code. Use this to reason about errors or plan your approach.",
            "parameters": {
                "type": "object",
                "properties": {
                    "thoughts": {"type": "string", "description": "Your reasoning about the problem"},
                },
                "required": ["thoughts"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_node_contract",
            "description": "Read the BuildSheet contract for this node: intent, inputs, influence rules, output protocol.",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {"type": "string", "description": "The node ID"},
                },
                "required": ["node_id"],
            },
        },
    },
]


# -----------------------------------------------------------------------------
# Mason Agent
# -----------------------------------------------------------------------------

class MasonAgent:
    def __init__(self, model: str = None, max_retries: int = 2):
        self.model = model or EFFECTIVE_MODEL_CODING
        self.max_retries = max_retries

    # -------------------------------------------------------------------------
    # Main API
    # -------------------------------------------------------------------------

    def generate_node_code(self, nodes: List[NodeTensor], brief: str = "", visual_palette: Dict = None) -> List[NodeTensor]:
        """Generate code for nodes using LLM (qwen2.5-coder or configured model)."""
        self._brief = brief
        self._visual_palette = visual_palette or {}
        self._all_nodes = nodes
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

                if not raw.strip():
                    # LLM returned empty (likely timeout) — skip to fallback
                    print(f"  [MASON] {node.id} LLM returned empty (attempt {attempt}), skipping to fallback")
                    break

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
                    # Semantic check: does code serve the design brief?
                    semantic_errors = self._semantic_check(node, code)
                    if semantic_errors:
                        errors = semantic_errors
                    else:
                        ok = True
                        final_code = code
                        last_errors = []
                        break

                if errors:
                    final_code = code
                    last_errors = errors
                    print(f"  [MASON] {node.id} attempt {attempt}/{self.max_retries} failed: {errors[0][:120]}")

            # FALLBACK: If primary model failed, retry with llama3.2
            if not ok:
                fallback_model = EFFECTIVE_MODEL_FALLBACK
                print(f"  [MASON] {node.id} primary failed — retrying with {fallback_model}")

                fb_template = ENGINE_TEMPLATES.get(engine, ENGINE_TEMPLATES.get("glsl", ""))
                if last_errors and final_code:
                    fb_prompt = self._build_repair_prompt(node, fb_template, final_code, last_errors)
                else:
                    fb_prompt = self._build_prompt(node, fb_template, None)

                raw = self._call_llm(fb_prompt, model=fallback_model)
                if not raw.strip():
                    print(f"  [MASON] {node.id} fallback LLM also empty")
                else:
                    code = self._extract_code(raw)
                    code = self._clean_llm_output(node, code)
                    code = self._wrap_code(node, code)
                    errors = self._node_validate(node, code)
                    if not errors:
                        errors = self._validate_input_usage(node, code)
                    if not errors:
                        semantic_errors = self._semantic_check(node, code)
                        if semantic_errors:
                            last_errors = semantic_errors
                        else:
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
        
        for model_idx, model in enumerate([EFFECTIVE_MODEL_CODING, EFFECTIVE_MODEL_FALLBACK]):
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
            # Passthrough GLSL: sample input texture, fallback to animated gradient
            code = """void main() {
    vec2 uv = v_uv;
    vec4 src = texture(u_input0, uv);
    if (src.a > 0.0) {
        fragColor = src;
    } else {
        float t = u_time * 0.3;
        fragColor = vec4(uv.x * 0.5 + 0.25, uv.y * 0.5 + 0.25, sin(t) * 0.5 + 0.5, 1.0);
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
        
        elif engine in ("p5", "p5js"):
            # Passthrough P5.js: simple animated gradient
            code = """import p5 from 'https://esm.sh/p5@1.9.0';

export default function init(canvas, width, height, params) {
    let currentTime = 0;
    let currentInputs = [];
    const sketch = new p5((s) => {
        s.setup = () => {
            s.createCanvas(width, height, s.P2D, canvas);
            s.pixelDensity(1);
            s.colorMode(s.HSB, 360, 100, 100, 100);
            s.noLoop();
        };
        s.draw = () => {
            try {
                s.clear();
                const t = currentTime;
                const inputs = currentInputs;
                if (inputs[0]) s.drawingContext.drawImage(inputs[0], 0, 0, width, height);
                const hue = (t * 30) % 360;
                s.noStroke();
                s.fill(hue, 60, 80, 50);
                s.rect(0, 0, width, height);
            } catch (_err) { /* mute draw errors */ }
        };
    });
    return (time, inputs) => { currentTime = time; currentInputs = inputs; sketch.redraw(); };
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
            # Passthrough Three.js: body-only format matching ThreeExecutor expectations
            # (THREE, scene, camera, params, inputs are provided by the executor)
            code = """// THREE, scene, camera, params, inputs provided by runtime
scene.add(new THREE.AmbientLight(0x404040));
scene.add(new THREE.DirectionalLight(0xffffff, 1));
camera.position.z = 5;

const mesh = new THREE.Mesh(
    new THREE.IcosahedronGeometry(1.5, 2),
    new THREE.MeshStandardMaterial({ color: 0x1e293b, wireframe: true })
);
scene.add(mesh);

return {
    update(time, inputs) {
        mesh.rotation.x = time * 0.3;
        mesh.rotation.y = time * 0.5;
    }
};"""
            return self._wrap_code(node, code)
        
        else:
            # Generic passthrough
            code = """// Passthrough: LLM generation failed
return (time, inputs) => { };"""
            return self._wrap_code(node, code)

    # -------------------------------------------------------------------------
    # BuildSheet-based generation (Chain of Influence)
    # -------------------------------------------------------------------------

    def generate_from_build_sheets(self, nodes: List[NodeTensor],
                                    sheet_map: Dict, visual_palette: Dict = None) -> List[NodeTensor]:
        """Generate code using per-node BuildSheets instead of a global brief."""
        self._brief = ""
        self._visual_palette = visual_palette or {}
        self._all_nodes = nodes
        updated: List[NodeTensor] = []

        for node in nodes:
            engine = (node.engine or "").strip()
            node.meta = node.meta or {}
            node.parameters = node.parameters or {}
            node.input_nodes = node.input_nodes or []

            if engine == "html_video":
                node.code_snippet = self._generate_html_video_snippet(node)
                node.mason_approved = True
                node.validation_errors = []
                updated.append(node)
                continue

            # Check predefined templates FIRST (tracking, audio, color, noise, particles, blend)
            category = _meta_attr(node.meta, "category", "unknown")
            predef_key = self._find_predefined_category(category)
            if predef_key:
                predef = PREDEFINED_CODE[predef_key]
                while predef.get("alias_of"):
                    alias_target = predef["alias_of"]
                    if alias_target in PREDEFINED_CODE:
                        predef = PREDEFINED_CODE[alias_target]
                    else:
                        break

                predef_code = predef.get("code", "")
                predef_engine = predef.get("engine", "glsl")
                node.engine = predef_engine
                engine = predef_engine

                # Ensure meta is a dict
                if not isinstance(node.meta, dict):
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

                # Apply predefined parameters
                if predef.get("parameters"):
                    for param_name, param_info in predef["parameters"].items():
                        if param_name not in node.parameters:
                            if isinstance(param_info, dict):
                                node.parameters[param_name] = param_info.get("default", 0)
                            else:
                                node.parameters[param_name] = param_info
                    node.meta["parameter_ui"] = predef["parameters"]

                node.meta["is_predefined"] = True
                node.meta["predefined_type"] = predef_key
                if predef.get("is_source"):
                    node.meta["is_source"] = True
                if predef.get("manifest"):
                    node.meta["manifest"] = predef["manifest"]

                # Non-GLSL predefined: skip validation
                if predef_engine in ("html_video", "canvas2d", "p5", "webaudio"):
                    node.code_snippet = predef_code
                    node.mason_approved = True
                    node.validation_errors = []
                    print(f"  [MASON] {node.id} (category='{category}'): PREDEFINED '{predef_key}' [BuildSheet]")
                    updated.append(node)
                    continue

                # GLSL predefined: wrap + validate
                wrapped_code = self._wrap_code(node, predef_code)
                errors = self._node_validate(node, wrapped_code)
                if not errors:
                    node.code_snippet = wrapped_code
                    node.mason_approved = True
                    node.validation_errors = []
                    print(f"  [MASON] {node.id} (category='{category}'): PREDEFINED '{predef_key}' [BuildSheet]")
                    updated.append(node)
                    continue
                else:
                    print(f"  [MASON] {node.id}: Predefined validation failed: {errors[:2]}, falling back to LLM")

            # LLM code generation from BuildSheet
            sheet = sheet_map.get(node.id)
            template = ENGINE_TEMPLATES.get(engine, ENGINE_TEMPLATES.get("glsl", ""))
            if sheet:
                self._brief = sheet.intent

            ok = False
            final_code = ""
            last_errors = []

            # --- Tool-calling path (GLSL/regl with BuildSheet) ---
            use_tools = engine in ("glsl", "regl") and sheet is not None

            if use_tools:
                handlers, handler_state = self._make_tool_handlers(node, sheet, template)
                system = self._build_tool_system_prompt(node, sheet, template)
                user_msg = (
                    f"You MUST use the update_node_code tool to submit your code. "
                    f"Do NOT write code as plain text.\n"
                    f"Step 1: Call update_node_code(\"{node.id}\", <your GLSL body code>)\n"
                    f"Step 2: Call compile_and_get_errors(\"{node.id}\")\n"
                    f"Step 3: If errors, call ponder() then fix and resubmit.\n"
                    f"Step 4: When compilation succeeds, say \"Done.\""
                )

                from phase2.aider_llm import get_aider_llm
                aider = get_aider_llm()
                tool_result = aider.call_with_tools(
                    system_prompt=system,
                    user_prompt=user_msg,
                    model_name=EFFECTIVE_MODEL_CODING,
                    tools=MASON_TOOLS,
                    tool_handlers=handlers,
                    max_turns=6,
                    code_node_id=node.id,
                )

                if handler_state["last_code"]:
                    final_code = handler_state["last_code"]
                    errors = self._node_validate(node, final_code)
                    if not errors:
                        errors = self._validate_input_usage(node, final_code)
                    if not errors:
                        ok = True
                        print(f"  [MASON] {node.id}: Tool-calling succeeded in {tool_result.turns_used} turns")
                    else:
                        last_errors = errors
                        print(f"  [MASON] {node.id}: Tool-calling finished but code has errors: {errors[0][:120]}")
                else:
                    print(f"  [MASON] {node.id}: Tool-calling produced no code, falling back to single-shot")

            # --- Single-shot fallback (canvas2d, no BuildSheet, or tool failure) ---
            if not ok:
                if sheet:
                    prompt = self._build_prompt_from_sheet(node, sheet, template)
                else:
                    prompt = self._build_prompt(node, template)

                primary_model = EFFECTIVE_MODEL_FALLBACK if engine in ("canvas2d",) else None

                for attempt in range(1, self.max_retries + 1):
                    raw = self._call_llm(prompt, model=primary_model)
                    if not raw.strip():
                        print(f"  [MASON] {node.id} LLM returned empty (attempt {attempt}), skipping to fallback")
                        break
                    code = self._extract_code(raw)
                    code = self._clean_llm_output(node, code)
                    code = self._wrap_code(node, code)
                    errors = self._node_validate(node, code)
                    if not errors:
                        errors = self._validate_input_usage(node, code)
                    if not errors:
                        ok = True
                        final_code = code
                        break
                    final_code = code
                    last_errors = errors
                    print(f"  [MASON] {node.id} attempt {attempt}/{self.max_retries}: {errors[0][:120]}")
                    prompt = self._build_repair_prompt(node, template, code, errors, sheet=sheet)

            # Fallback model (swap: canvas2d used llama primary, so fallback to coder; others use llama fallback)
            if not ok:
                fallback_model = EFFECTIVE_MODEL_CODING if engine in ("canvas2d",) else EFFECTIVE_MODEL_FALLBACK
                fb_prompt = self._build_repair_prompt(node, template, final_code, last_errors, sheet=sheet) if final_code else prompt
                raw = self._call_llm(fb_prompt, model=fallback_model)
                if not raw.strip():
                    print(f"  [MASON] {node.id} fallback LLM also empty")
                else:
                    code = self._extract_code(raw)
                    code = self._clean_llm_output(node, code)
                    code = self._wrap_code(node, code)
                    errors = self._node_validate(node, code)
                    if not errors:
                        ok = True
                        final_code = code

            # Passthrough fallback — keeps rendering alive but marks as NOT approved
            if not ok:
                passthrough = self._generate_passthrough_code(node, engine)
                if passthrough:
                    final_code = passthrough
                    last_errors = ["PASSTHROUGH_FALLBACK: LLM failed, using identity code"] + last_errors

            node.code_snippet = final_code
            node.mason_approved = ok  # passthrough stays False — not a real pass
            node.validation_errors = last_errors
            updated.append(node)

        return updated

    def _build_prompt_from_sheet(self, node: NodeTensor, sheet, template: str) -> str:
        """Build Mason prompt from a BuildSheet (Chain of Influence contract)."""
        from phase2.agents.influence_compiler import PROTOCOL_DESCRIPTIONS

        # Input section
        input_lines = []
        for inp in sheet.inputs:
            proto_desc = PROTOCOL_DESCRIPTIONS.get(inp.get("protocol", ""), inp.get("protocol", ""))
            meaning = inp.get("meaning", proto_desc)
            src_intent = inp.get("source_intent", "")
            input_lines.append(
                f"- {inp['name']}: receives {inp.get('protocol', 'COLOR_RGBA')} ({proto_desc})\n"
                f"  Meaning: {meaning}\n"
                f"  From: {src_intent}" if src_intent else
                f"- {inp['name']}: receives {inp.get('protocol', 'COLOR_RGBA')} ({proto_desc})\n"
                f"  Meaning: {meaning}"
            )
        inputs_section = "\n".join(input_lines) if input_lines else "No inputs (source node — generate your own content)"

        # Influence rules section
        rules_parts = []
        rules = sheet.influence_rules or {}
        if rules.get("must_use"):
            rules_parts.append("MUST USE:\n" + "\n".join(f"  - {r}" for r in rules["must_use"]))
        if rules.get("preserve"):
            rules_parts.append("PRESERVE:\n" + "\n".join(f"  - {r}" for r in rules["preserve"]))
        if rules.get("allow"):
            rules_parts.append("ALLOWED enhancements:\n" + "\n".join(f"  - {r}" for r in rules["allow"]))
        if rules.get("avoid"):
            rules_parts.append("AVOID:\n" + "\n".join(f"  - {r}" for r in rules["avoid"]))
        rules_section = "\n".join(rules_parts) if rules_parts else ""

        # Style anchor
        style_parts = []
        palette = sheet.style_anchor.get("palette", [])
        if palette and node.engine in ("glsl", "regl"):
            hex_list = ", ".join(str(c) for c in palette[:4])
            glsl_colors = []
            for h in palette[:3]:
                h = str(h).lstrip('#')
                if len(h) == 6:
                    r, g, b = int(h[0:2], 16)/255, int(h[2:4], 16)/255, int(h[4:6], 16)/255
                    glsl_colors.append(f"vec3({r:.3f}, {g:.3f}, {b:.3f})")
            style_parts.append(f"COLOR PALETTE: {hex_list}")
            if glsl_colors:
                style_parts.append(f"GLSL: {', '.join(glsl_colors)}")
        elif palette:
            style_parts.append(f"COLOR PALETTE: {', '.join(str(c) for c in palette[:4])}")

        motion = sheet.style_anchor.get("motion", [])
        if motion:
            style_parts.append(f"MOTION STYLE: {', '.join(str(m) for m in motion)}")
        style_section = "\n".join(style_parts)

        # Engine-specific guidance
        glsl_guidance = ""
        if node.engine in ("glsl", "regl"):
            glsl_guidance = """
GLSL RULES (CRITICAL — read carefully):

AVAILABLE UTILITY FUNCTIONS (already in shader prefix — do NOT redefine):
  float hash(float n)       // hash a single float
  float hash(vec2 p)        // hash a 2D point
  vec2  hash2(vec2 p)       // 2D -> 2D hash
  vec3  hash3(vec3 p)       // 3D -> 3D hash
  vec3  hash3(vec2 p)       // 2D -> 3D hash
  float noise(vec2 x)       // Perlin noise — ONLY takes vec2
  float snoise(vec2 p)      // signed noise [-1,1] — ONLY takes vec2
  float snoise(vec3 p)      // signed noise from vec3 (internally uses .xy+.z)
  float fbm(vec2 x)         // fractal Brownian motion — ONLY takes vec2
  float worley(vec2 p)      // cellular/Worley noise — ONLY takes vec2
  float simplex(vec2 p)     // simplex noise alias — ONLY takes vec2
  float simplex(vec3 p)     // simplex from vec3
  float perlin(vec2 p)      // Perlin alias — ONLY takes vec2
  float perlin(vec3 p)      // Perlin from vec3
  float voronoi(vec2 p)     // Voronoi distance — ONLY takes vec2

Color space conversions (available in shader prefix — use directly):
  vec3 rgb2hsv(vec3 c)   // RGB -> HSV
  vec3 hsv2rgb(vec3 c)   // HSV -> RGB
  vec3 hsl2rgb(vec3 c)   // HSL -> RGB

FUNCTIONS THAT DO NOT EXIST (never call these):
  noise(float), noise(vec3), fbm(vec3), worley(vec3), voronoi(vec3)
  ridgedFBM(), turbulence()
  s_vec2(), s_vec3(), s_vec4()

GLSL RESERVED WORDS — NEVER use as variable or function names:
  smooth, sample, input, output, filter, texture, common, partition, active,
  flat, noperspective, centroid, precise, coherent, volatile, restrict,
  buffer, shared, resource, patch, subroutine

CONSTRUCTOR RULES:
  vec2(float, float) or vec2(float)
  vec3(float, float, float) or vec3(vec2, float) or vec3(float)
  vec4(float, float, float, float) or vec4(vec3, float) or vec4(float)
  NEVER pass more arguments than the constructor accepts.
  WRONG: vec3(r, g, b, 1.0)  RIGHT: vec4(r, g, b, 1.0)
  WRONG: vec4(color, a, extra) RIGHT: vec4(color, a)

DECLARATIONS:
  Do NOT include: #version, precision, in/out, uniform, layout()
  Do NOT use: gl_FragColor, texture2D()
  DO use: texture(sampler, uv), write to fragColor
  Do NOT redeclare uniforms as locals (no "float u_time = u_time;")
"""
        elif node.engine == "canvas2d":
            glsl_guidance = """
CANVAS2D RULES (CRITICAL):
- Define a draw(ctx, w, h, t, inputs, params) function
- External libraries are NOT available (no SimplexNoise, no THREE, no p5)
- For noise, use inline math: Math.sin(x * 12.9898 + y * 78.233) * 43758.5453 % 1
- Populate ALL arrays/collections OUTSIDE draw(), immediately after declaration
- draw() MUST call at least one drawing method (ctx.arc, ctx.fillRect, ctx.stroke, etc.)
- Use params.xxx for controllable values, NOT hardcoded magic numbers
- Do NOT use requestAnimationFrame, imports, require, or DOM manipulation
"""
        elif node.engine in ("p5", "p5js", "js_module"):
            glsl_guidance = """
P5.js / JS MODULE RULES (CRITICAL):
- Use s.noLoop() in setup and s. prefix on all p5 calls
- The exported update function MUST accept THREE parameters: (time, inputs, inputDataMap)
- NEVER use `new p5.Graphics(...)` — use s.createGraphics(width, height) inside the sketch callback
- NEVER push() to arrays inside s.draw() — pre-allocate at fixed size and update in-place
- NEVER use nested for loops over pixel coordinates (for x < width inside for y < height):
    FORBIDDEN: for (let y = 0; y < height; y++) { for (let x = 0; x < width; x++) { s.rect(x,y,1,1); } }
    This creates 262,144+ draw calls per frame and freezes the browser for 15+ seconds.
    For backgrounds: use s.background(r, g, b) or a single s.rect(0, 0, width, height).
    For starfields/particles: place N random points with N ≤ 500 total per frame.
    Total draw calls per s.draw() frame MUST stay under 1000.
"""

        params_str = ", ".join(sheet.params) if sheet.params else "auto-detect from your code"

        # Z-axis semantic context
        z_context = ""
        if sheet.z_total > 1:
            z = sheet.grid_position[2]
            x = sheet.grid_position[0]
            role_desc = {
                "source": "SOURCE layer — you generate original content, no upstream dependency",
                "processor": "PROCESSOR layer — you transform upstream input, preserving its structure",
                "compositor": "COMPOSITOR layer — you combine/refine signals approaching final output",
                "output": "OUTPUT layer — you produce the final visual for display",
            }.get(sheet.z_role, f"layer role: {sheet.z_role}")
            z_context = f"GRID POSITION: Z={z}/{sheet.z_total-1}, X={x} — {role_desc}"
            # Add per-input Z-distance context
            for inp in sheet.inputs:
                src_z = inp.get("source_z")
                if src_z is not None:
                    delta = z - src_z
                    direction = "upstream" if delta > 0 else ("same layer" if delta == 0 else "downstream feedback")
                    z_context += f"\n  {inp['name']} comes from Z={src_z} ({direction}, {abs(delta)} layer{'s' if abs(delta)!=1 else ''} away)"

        # Spatial coherence hint — only for nodes with inputs
        spatial_hint = ""
        if sheet.inputs and node.engine in ("glsl", "regl"):
            spatial_hint = """
SPATIAL COHERENCE (CRITICAL):
Your visual output MUST align spatially with your input textures in UV space.
- Sample the input texture at the SAME UV coordinates where you draw your output.
- If the input has content centered at UV (0.5, 0.5), your effect should also center there.
- Do NOT generate shapes at arbitrary UV positions — always derive position from the input.
- For compositing: read input with texture(u_input0, uv), then layer your effect on top at the same uv.
"""

        return f"""You are a Senior Graphics Engineer writing code for a node-graph visual system.

NODE PURPOSE: "{sheet.intent}"
{z_context}

INPUTS:
{inputs_section}

OUTPUT: {sheet.output_protocol}

{f"INFLUENCE RULES (contractual — you MUST follow these):" if rules_section else ""}
{rules_section}
{spatial_hint}
{style_section}
{glsl_guidance}
PARAMETERS to expose: {params_str}

PARAMETER RULES (CRITICAL — bad params = failing review):
- Each parameter MUST produce a CLEARLY VISIBLE visual change when moved 25% from its default.
- NEVER use a parameter as `col *= u_param` with default 1.0 — this is identity and has no visible effect.
- Use MIX/BLEND patterns: `mix(base_effect, full_effect, u_param)` where 0.0=no effect, 1.0=full effect.
- For intensity params (glow, blur, grade): 0.0=none, 0.5=moderate, 1.0=intense. NOT multipliers.
- For speed/rate: use as a real multiplier on time — `u_time * u_speed` is correct (default 0.5-1.0).
- For scale/zoom: use as a real UV scale — `uv * u_scale` is correct (default 1.0 is fine here).
- ALWAYS use each exposed parameter in the code — do not expose unused params.

{template}

Return ONLY raw code. No markdown fences, no explanations."""

    def _build_tool_system_prompt(self, node: NodeTensor, sheet, template: str) -> str:
        """System prompt for tool-calling code generation loop.

        Reuses the same BuildSheet context as _build_prompt_from_sheet() but
        replaces 'Return ONLY raw code' with tool workflow instructions.
        """
        # Build the same context string — reuse _build_prompt_from_sheet and strip the tail
        base = self._build_prompt_from_sheet(node, sheet, template)
        # Remove the final instruction line
        base = base.replace("Return ONLY raw code. No markdown fences, no explanations.", "").rstrip()

        return f"""{base}

TOOLS — You MUST respond with a JSON tool call object, not plain text code.

Available tools:
- update_node_code(node_id, new_code): Submit your GLSL code.
- compile_and_get_errors(node_id): Compile and check for errors.
- ponder(thoughts): Reason about the problem before acting.
- read_node_contract(node_id): Review this node's requirements.

HOW TO CALL A TOOL — respond with ONLY this JSON format:
{{"name": "update_node_code", "arguments": {{"node_id": "{node.id}", "new_code": "void main() {{\\n    vec2 uv = v_uv;\\n    fragColor = vec4(uv, 0.5, 1.0);\\n}}"}}}}

CODE FORMAT — your new_code MUST:
- Include void main() {{ ... }} wrapper
- Use v_uv for UV coordinates, u_time for time
- Use texture(u_input0, uv) to sample inputs (NOT texture2D)
- Write final color to fragColor (NOT gl_FragColor)
- NOT include #version, precision, uniform, in/out, or layout declarations

WORKFLOW:
1. Call update_node_code("{node.id}", <your GLSL code>)
2. Call compile_and_get_errors("{node.id}")
3. If errors: call ponder() to identify the EXACT line, then fix ONLY that line
4. Repeat steps 1-3 until compilation succeeds
5. When done, respond with "Done."

CRITICAL: Do NOT write code as plain text. Always use the JSON tool call format above."""

    def _make_tool_handlers(self, node: NodeTensor, sheet, template: str):
        """Create tool handler functions bound to a specific node context.

        Returns (handlers_dict, state_dict). Handlers close over the node
        and sheet. state["last_code"] holds the cleaned/wrapped code.
        """
        state = {"last_code": None, "compile_pass": False}

        def update_node_code(node_id: str, new_code: str) -> str:
            code = self._extract_code(new_code)
            code = self._clean_llm_output(node, code)
            code = self._wrap_code(node, code)
            state["last_code"] = code
            state["compile_pass"] = False
            return "Code updated. Call compile_and_get_errors to check for issues."

        def compile_and_get_errors(node_id: str) -> str:
            code = state["last_code"]
            if not code:
                return "No code submitted yet. Call update_node_code first."
            errors = self._node_validate(node, code)
            if not errors:
                errors = self._validate_input_usage(node, code)
            if not errors:
                state["compile_pass"] = True
                return "Compilation successful. No errors."
            parsed = self._parse_compiler_diagnostics(errors)
            return "Errors found:\n" + "\n".join(f"- {d}" for d in parsed)

        def ponder(thoughts: str) -> str:
            print(f"  [MASON] {node.id} ponder: {thoughts[:150]}")
            return "Thought recorded. Now proceed with your next action."

        def read_node_contract(node_id: str) -> str:
            if not sheet:
                return "No BuildSheet available for this node."
            from phase2.agents.influence_compiler import PROTOCOL_DESCRIPTIONS
            lines = [f"Intent: {sheet.intent}"]
            lines.append(f"Engine: {sheet.engine}")
            lines.append(f"Output: {sheet.output_protocol}")
            if sheet.inputs:
                lines.append("Inputs:")
                for inp in sheet.inputs:
                    proto = inp.get("protocol", "COLOR_RGBA")
                    desc = PROTOCOL_DESCRIPTIONS.get(proto, proto)
                    src = inp.get("source_intent", "")
                    lines.append(f"  - {inp['name']}: {desc}")
                    if src:
                        lines.append(f"    From: {src}")
            rules = sheet.influence_rules or {}
            if rules.get("must_use"):
                lines.append("Must use: " + ", ".join(rules["must_use"]))
            if rules.get("preserve"):
                lines.append("Preserve: " + ", ".join(rules["preserve"]))
            if rules.get("allow"):
                lines.append("Allowed: " + ", ".join(rules["allow"]))
            if rules.get("avoid"):
                lines.append("Avoid: " + ", ".join(rules["avoid"]))
            if sheet.params:
                lines.append(f"Parameters: {', '.join(sheet.params)}")
            return "\n".join(lines)

        handlers = {
            "update_node_code": update_node_code,
            "compile_and_get_errors": compile_and_get_errors,
            "ponder": ponder,
            "read_node_contract": read_node_contract,
        }
        return handlers, state

    # Category aliases: SemanticReasoner may output different names than PREDEFINED_CODE keys
    CATEGORY_ALIASES = {
        # Tracking variants
        "tracking": "p5_tracking", "face_tracking": "p5_tracking",
        "hand_tracking": "p5_tracking", "body_tracking": "p5_tracking",
        "face_detection": "p5_tracking", "hand_detection": "p5_tracking",
        "body_detection": "p5_tracking", "pose_detection": "p5_tracking",
        "face_mesh": "p5_tracking", "hand_pose": "p5_tracking",
        "body_pose": "p5_tracking", "landmark": "p5_tracking",
        "landmarks": "p5_tracking", "ml5_tracking": "p5_tracking",
        "mediapipe": "p5_tracking",
        # Audio variants
        "audio": "audio_input", "microphone": "audio_input",
        "mic_input": "audio_input", "audio_source": "audio_input",
        "fft": "audio_input", "audio_fft": "audio_input",
        "audio_analyser": "audio_input", "audio_analyzer": "audio_input",
        # Video/image variants
        "video": "video_input", "camera": "webcam_input",
        "webcam": "webcam_input", "image": "image_input",
        # Noise variants
        "noise": "noise_generator", "perlin": "noise_perlin",
        "simplex": "noise_simplex", "worley": "noise_worley",
        "fbm": "noise_fbm", "perlin_noise": "noise_perlin",
        # Particle variants
        "particle": "particle_node", "particles": "particle_node",
        "particle_system": "particle_node",
        # Color variants
        "color": "color_node", "color_grade": "color_node",
        "color_adjustment": "color_node", "color_grading": "color_node",
        # Blend variants
        "blend": "blend_node", "layer_blend": "blend_node",
        "composite": "blend_node", "mix": "blend_node",
    }

    def _find_predefined_category(self, category: str) -> Optional[str]:
        """Find matching predefined template key for a category.

        Returns:
            The key in PREDEFINED_CODE to use, or None if no match.
        """
        if not category:
            return None

        category_lower = category.lower().strip()

        # Exact match in PREDEFINED_CODE keys
        if category in PREDEFINED_CODE:
            return category
        if category_lower in PREDEFINED_CODE:
            return category_lower

        # Check aliases
        alias = self.CATEGORY_ALIASES.get(category_lower)
        if alias and alias in PREDEFINED_CODE:
            return alias

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

        # Build input context with data transfer + affinity weight guidance
        has_inputs = bool(node.input_nodes)
        num_inputs = len(node.input_nodes) if node.input_nodes else 0
        has_affinity = any(k.startswith('affinity_') for k in (node.parameters or {}))
        input_hint = ""
        if has_inputs and node.engine in ("glsl", "regl"):
            affinity_hint = ""
            if has_affinity and num_inputs >= 2:
                # Build affinity weight description from params
                aff_parts = []
                for i in range(num_inputs):
                    w = (node.parameters or {}).get(f'affinity_{i}', 0)
                    if w:
                        in_id = node.input_nodes[i] if i < len(node.input_nodes) else f"input_{i}"
                        aff_parts.append(f"u_affinity_{i} = {w:.3f} (from {in_id})")
                affinity_hint = f"""
- AFFINITY BLEND WEIGHTS (transformer-style attention — use these to weight your inputs):
  {chr(10).join('  ' + p for p in aff_parts)}
  When blending multiple inputs, use: mix(colorA, colorB, u_affinity_1) or weighted sum:
  vec4 blended = texture(u_input0, uv) * u_affinity_0 + texture(u_<sideInput>, uv) * u_affinity_1;"""

            input_hint = f"""- This node receives input via texture(u_input0, uv). You MUST sample and use it.
- If upstream is an audio node: u_audio_level (0-1), u_audio_bass, u_audio_mid, u_audio_treble are available as uniforms.
- If upstream is a tracking node: u_track_x, u_track_y (normalized 0-1 position), u_track_count are available.{affinity_hint}"""
        elif has_inputs and node.engine == "canvas2d":
            affinity_hint_c2d = ""
            if has_affinity and num_inputs >= 2:
                affinity_hint_c2d = """
  - params.affinity_weights — object mapping input node IDs to blend weights (0-1, sum to 1.0)
    Use these weights to proportionally blend upstream contributions."""

            input_hint = f"""- This node receives upstream data via the `inputs` object:
  - inputs.data[nodeId].data.trackingData — array of face/hand/body detections with keypoints
  - inputs.data[nodeId].data.audioData — {{ level, bass, mid, treble }} (all 0-1 range)
  - inputs.readTexture(nodeId) — returns ImageData from an upstream texture{affinity_hint_c2d}
- Use these to make your visualization reactive to upstream inputs."""

        glsl_guidance = ""
        if node.engine in ("glsl", "regl"):
            glsl_guidance = """
IMPORTANT for GLSL:
- Use ONLY these constructor forms: vec2(x, y), vec3(r, g, b), vec4(r, g, b, a)
- Available noise functions: hash(), hash2(), hash3(), noise(), snoise(), fbm(), worley(), simplex(), perlin(), voronoi()
- Do NOT redefine these functions — they are already provided in the shader prefix
- Do NOT use: s_vec2(), s_vec3(), s_vec4() — only use standard vec constructors
- Arrays must be pre-populated in draw/init sections, not declared empty
"""

        canvas2d_guidance = ""
        if node.engine == "canvas2d":
            canvas2d_guidance = """
IMPORTANT for Canvas2D:
- Declare and populate arrays WITHIN the same function scope (before draw() uses them)
- Pre-allocate arrays at a fixed size: `const particles = Array.from({length: params.count || 60}, () => spawn());`
- In draw(), UPDATE elements in-place; if one dies, replace at the same index: `particles[i] = spawn()`
- NEVER push() to arrays inside draw() — unbounded growth causes 400ms+ frame stalls
- Use ctx.beginPath(), ctx.arc(), ctx.fill(), ctx.stroke() for drawing
- Arrays cannot be empty when draw() is called
"""
        elif node.engine in ("p5", "p5js"):
            canvas2d_guidance = """
IMPORTANT for P5.js:
- ALL p5 functions MUST use s. prefix: s.ellipse(), s.fill(), s.bezier(), s.push(), s.pop()
- Use s.clear() at the start of s.draw() for transparent layering
- Use s.P2D renderer: s.createCanvas(width, height, s.P2D, canvas)
- Use s.pixelDensity(1) and s.noLoop() in setup — driven mode: pipeline calls sketch.redraw() each tick
- The exported update function MUST accept THREE parameters: (time, inputs, inputDataMap)
  CORRECT:   export default function init(canvas, width, height) { ... return (time, inputs, inputDataMap) => { ... }; }
  WRONG:     return (time, inputs) => { ... }   <-- missing inputDataMap, audio will not work
- AUDIO REACTIVITY: Read from inputDataMap to get audio/tracking data each frame:
    const audioLevel  = (inputDataMap?.audio?.level  ?? 0);   // 0.0–1.0
    const audioBass   = (inputDataMap?.audio?.bass   ?? 0);
    const audioMid    = (inputDataMap?.audio?.mid    ?? 0);
    const audioTreble = (inputDataMap?.audio?.treble ?? 0);
  Use these to modulate particle count, speed, size, color, or any intensity parameter.
- ALWAYS declare `let currentInputs = []; let currentInputDataMap = {};` next to currentTime
- In the return function: `currentTime = time; currentInputs = inputs; currentInputDataMap = inputDataMap || {};`
- At the top of s.draw: `const inputs = currentInputs; const idm = currentInputDataMap;`
- Inputs are HTMLCanvasElement objects — composite with s.drawingContext.drawImage(inputs[0], 0, 0, width, height). NEVER use s.image() with raw canvas elements.
- ALWAYS wrap the ENTIRE s.draw body in try { ... } catch (_err) {} — uncaught s.draw errors crash Chrome by re-throwing every frame
- SPATIAL PLACEMENT: Place elements at semantically correct positions — NOT always at center. Planets orbit, backgrounds fill canvas, side elements use edges.
- CRITICAL PERFORMANCE — ARRAYS: NEVER push() to any array inside s.draw() or inside the return function. Pre-allocate ALL arrays in init() at a fixed size. In s.draw(), UPDATE elements in-place; if one dies, replace at the same index. Unbounded arrays cause 400ms+ frame stalls.
- CRITICAL PERFORMANCE — NO PIXEL LOOPS: NEVER use nested for loops over pixel coordinates. The following pattern is FORBIDDEN and will freeze the browser (262,144 draw calls/frame at 512×512):
    WRONG: for (let y = 0; y < height; y++) { for (let x = 0; x < width; x++) { s.rect(x, y, 1, 1); } }
    WRONG: for (let i = 0; i < width; i++) { for (let j = 0; j < height; j++) { s.point(x, y); } }
  For backgrounds/fills: use s.background(r, g, b) or a single s.rect(0, 0, width, height). For starfields/particles: place N random points with N ≤ 500. Total draw calls per s.draw() frame MUST stay under 1000.
- GRAPHICS OBJECTS: NEVER use `new p5.Graphics(...)` — this is invalid. To create an offscreen buffer use `s.createGraphics(width, height)` inside the p5 sketch callback (setup or init), NOT outside it.
"""

        # Design brief + scoped semantic purpose
        brief_section = ""
        brief = getattr(self, '_brief', '')
        purpose = getattr(node, 'semantic_purpose', '') or ''
        if purpose:
            # Scoped purpose from SemanticDecomposer — more focused than full brief
            brief_section = f"""
DESIGN BRIEF: "{brief[:120]}"

YOUR NODE'S SPECIFIC PURPOSE: "{purpose}"
Your code must fulfill THIS node's specific purpose above. Don't try to implement
the entire brief — focus on what THIS node contributes to the pipeline.
"""
        elif brief:
            brief_section = f"""
DESIGN BRIEF: "{brief}"
Your code must visually contribute to this design. Don't write generic effects —
make the output specifically serve the brief above.
"""

        # Visual palette from reference images — concrete color/shape/motion constraints
        palette_section = ""
        vp = getattr(self, '_visual_palette', {})
        if vp:
            parts_vp = []
            primary = vp.get('primary_colors', [])
            accent = vp.get('accent_colors', [])
            shapes = vp.get('shapes', [])
            motion = vp.get('motion_words', [])

            if primary:
                hex_list = ', '.join(primary[:4])
                # Convert hex to GLSL/JS-friendly format
                glsl_colors = []
                js_colors = []
                for h in primary[:3]:
                    h = h.lstrip('#')
                    if len(h) == 6:
                        r, g, b = int(h[0:2], 16)/255, int(h[2:4], 16)/255, int(h[4:6], 16)/255
                        glsl_colors.append(f"vec3({r:.3f}, {g:.3f}, {b:.3f})")
                        js_colors.append(f"'#{h}'")

                if node.engine in ("glsl", "regl"):
                    parts_vp.append(f"- PRIMARY COLORS (use these, not random): {hex_list}")
                    parts_vp.append(f"  GLSL: {', '.join(glsl_colors)}")
                elif node.engine == "canvas2d":
                    parts_vp.append(f"- PRIMARY COLORS (use these, not random): {hex_list}")
                    parts_vp.append(f"  JS: [{', '.join(js_colors)}]")
                else:
                    parts_vp.append(f"- PRIMARY COLORS: {hex_list}")

            if accent:
                parts_vp.append(f"- ACCENT COLORS (for highlights/details): {', '.join(accent[:3])}")

            if shapes:
                parts_vp.append(f"- SHAPES to use: {', '.join(shapes)}")

            if motion:
                parts_vp.append(f"- MOTION style: {', '.join(motion)}")

            if parts_vp:
                palette_section = "VISUAL PALETTE (from reference images — you MUST use these colors):\n" + "\n".join(parts_vp)

        # Graph position — what feeds into this node and what it feeds
        graph_section = ""
        all_nodes = getattr(self, '_all_nodes', [])
        if all_nodes:
            upstream = []
            for nid in (node.input_nodes or []):
                src = next((n for n in all_nodes if n.id == nid), None)
                if src:
                    src_label = _meta_attr(src.meta, "label", src.id)
                    src_cat = _meta_attr(src.meta, "category", src.id)
                    upstream.append(f"{src_label} ({src_cat})")
            downstream = []
            for n in all_nodes:
                if node.id in (n.input_nodes or []):
                    d_label = _meta_attr(n.meta, "label", n.id)
                    d_cat = _meta_attr(n.meta, "category", n.id)
                    downstream.append(f"{d_label} ({d_cat})")

            parts = []
            if upstream:
                parts.append(f"- Receives input from: {', '.join(upstream)}")
            if downstream:
                parts.append(f"- Feeds into: {', '.join(downstream)}")
            if not upstream and not downstream:
                parts.append("- Standalone generator (no inputs, or final output)")
            graph_section = "GRAPH POSITION:\n" + "\n".join(parts)

        return f"""You are a Senior Graphics Engineer writing code for a node-graph visual system.
{brief_section}
{palette_section}
Task: Write SELF-CONTAINED code for a node named "{label}" (category: {category}).

{graph_section}

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

    def _parse_compiler_diagnostics(self, errors: List[str]) -> List[str]:
        """Parse validation errors into structured diagnostics for LLM repair.

        Translates cryptic glslangValidator messages into plain English with fix hints.
        The LLM uses these descriptions to understand and fix the actual issue —
        no hardcoded fixes, just better diagnostic context (like IDE error tooltips).
        """
        diagnostics = []

        for err in errors:
            if "GLSL compile failed" not in err:
                diagnostics.append(err)
                continue

            for line in err.split('\n'):
                line = line.strip()
                m = re.match(r"ERROR:\s*\d+:(\d+):\s*'([^']*)'\s*:\s*(.*)", line)
                if not m:
                    continue

                err_line = m.group(1)
                context = m.group(2)
                message = m.group(3).strip()
                msg_lower = message.lower()

                if "scalar swizzle" in msg_lower:
                    diagnostics.append(
                        f"Line ~{err_line}: SCALAR SWIZZLE — Applying .x/.y/.z/.w to a float value. "
                        f"Functions like noise(), hash(), length(), dot(), sin(), cos() return float, not vec. "
                        f"Remove the swizzle (.x, .xy, etc.) from the scalar expression."
                    )
                elif "wrong operand types" in msg_lower:
                    diagnostics.append(
                        f"Line ~{err_line}: TYPE MISMATCH — {message}. "
                        f"Fixes: float() to cast int→float, mod(a,b) instead of a%b, "
                        f"ensure arithmetic operands match (float*float, vec3*float, etc.)."
                    )
                elif "undeclared identifier" in msg_lower:
                    diagnostics.append(
                        f"Line ~{err_line}: UNDECLARED '{context}' — not defined. "
                        f"Available functions: noise(vec2), fbm(vec2), worley(vec2), voronoi(vec2), "
                        f"simplex(vec2/vec3), perlin(vec2/vec3), hash(float/vec2), hash2(vec2), "
                        f"hash3(vec3/vec2), rgb2hsv(vec3), hsv2rgb(vec3), hsl2rgb(vec3), snoise(vec2/vec3). "
                        f"If '{context}' is a custom function, define it before use."
                    )
                elif "reserved word" in msg_lower or "reserved" in msg_lower:
                    diagnostics.append(
                        f"Line ~{err_line}: RESERVED WORD '{context}' — cannot use as identifier. "
                        f"Rename to '{context}Val' or similar. Reserved: smooth, sample, input, output, "
                        f"filter, common, flat, buffer, shared, texture."
                    )
                elif "redefinition" in msg_lower or "already defined" in msg_lower:
                    diagnostics.append(
                        f"Line ~{err_line}: REDEFINITION of '{context}' — already in shader prefix. "
                        f"Remove your definition. Runtime provides: noise, fbm, worley, voronoi, "
                        f"simplex, perlin, hash, hash2, hash3, snoise, rgb2hsv, hsv2rgb, hsl2rgb."
                    )
                elif "missing entry point" in msg_lower:
                    diagnostics.append(
                        "MISSING MAIN — Code must define void main(). "
                        "Wrap your code in void main() { ... } and assign result to fragColor."
                    )
                elif "constructor" in msg_lower and "argument" in msg_lower:
                    diagnostics.append(
                        f"Line ~{err_line}: CONSTRUCTOR — {message}. "
                        f"vec2=2 args, vec3=3 args, vec4=4 args. Also: vec4(vec3, float), vec4(vec2, vec2)."
                    )
                elif "assign" in msg_lower and "convert" in msg_lower:
                    diagnostics.append(
                        f"Line ~{err_line}: TYPE CONVERSION — {message}. "
                        f"Cannot assign between types. fragColor needs vec4: fragColor = vec4(myVec3, 1.0)."
                    )
                elif "no matching overloaded function" in msg_lower:
                    diagnostics.append(
                        f"Line ~{err_line}: WRONG ARGS for '{context}' — {message}. "
                        f"Check the function signature. noise/fbm/worley/voronoi/simplex/perlin take vec2, "
                        f"not vec3 or float. hash takes float or vec2."
                    )
                else:
                    diagnostics.append(f"Line ~{err_line}: {context}: {message}")

        return diagnostics

    def _build_repair_prompt(self, node: NodeTensor, template: str, last_code: str, errors: List[str], sheet=None) -> str:
        category = _meta_attr(node.meta, "category", "unknown")

        # Parse compiler errors into clear diagnostics for the LLM
        if any("GLSL compile failed" in e for e in errors):
            parsed = self._parse_compiler_diagnostics(errors)
            err_text = "\n".join(f"- {d}" for d in parsed)
        else:
            err_text = "\n".join(f"- {e}" for e in (errors or ["Unknown error"]))

        engine_guidance = ""
        if node.engine in ("glsl", "regl"):
            engine_guidance = """
GLSL RULES:
- noise/fbm/worley/voronoi/simplex/perlin: ONLY take vec2
- rgb2hsv/hsv2rgb/hsl2rgb: already in shader prefix, do NOT redefine
- Do NOT redeclare uniforms, #version, precision, in/out, layout()
- Do NOT use s_vec2/s_vec3/s_vec4 — standard GLSL vec constructors only
- Reserved words (never use as var names): smooth, sample, input, output, filter, texture
"""
        elif node.engine == "canvas2d":
            engine_guidance = """
Canvas2D RULES:
- Populate ALL arrays OUTSIDE draw(), not inside it
- draw() MUST call at least one drawing method (ctx.arc, ctx.fillRect, etc.)
- No external libraries (SimplexNoise, THREE, p5). Use inline math for noise.
"""
        elif node.engine in ("p5", "p5js"):
            engine_guidance = """
P5.js RULES:
- The exported update function MUST accept THREE parameters: (time, inputs, inputDataMap)
  CORRECT: return (time, inputs, inputDataMap) => { ... }
  WRONG:   return (time, inputs) => { ... }
- NEVER use `new p5.Graphics(...)` — use `s.createGraphics(width, height)` inside the sketch callback instead
- NEVER push() to arrays inside s.draw() — pre-allocate at fixed size and update in-place
- ALL p5 calls must use s. prefix
- NEVER use nested for loops over pixel coordinates (for x < width inside for y < height or vice versa). This creates 262,144+ draw calls per frame and freezes the browser. For backgrounds use s.background(). For starfields use N random points with N ≤ 500. Total draw calls per s.draw() MUST stay under 1000.
"""

        brief_hint = ""
        brief = getattr(self, '_brief', '')
        if brief:
            brief_hint = f'\nDesign Brief: "{brief}"\nFix the code while keeping it visually relevant to this design.\n'

        # Include color palette in repair prompt too
        palette_hint = ""
        vp = getattr(self, '_visual_palette', {})
        primary = vp.get('primary_colors', []) if vp else []
        if primary:
            palette_hint = f"\nCOLOR PALETTE (use these colors, not random): {', '.join(primary[:4])}\n"

        # BuildSheet context — keeps the LLM aware of node purpose during repair
        sheet_hint = ""
        if sheet:
            sheet_hint = f'\nNODE PURPOSE: "{sheet.intent}"'
            if sheet.inputs:
                input_names = [f"{inp['name']} ({inp.get('protocol', 'COLOR_RGBA')})" for inp in sheet.inputs]
                sheet_hint += f"\nINPUTS: {', '.join(input_names)}"
            rules = sheet.influence_rules or {}
            if rules.get("must_use"):
                sheet_hint += f"\nMUST USE: {', '.join(rules['must_use'])}"
            if rules.get("avoid"):
                sheet_hint += f"\nAVOID: {', '.join(rules['avoid'])}"
            sheet_hint += f"\nOUTPUT: {sheet.output_protocol}\n"

        return f"""You are Senior Graphics Engineer writing code for a node-graph visual system. Your previous output did not validate. Fix the errors.
{brief_hint}{palette_hint}{sheet_hint}
Node:
- id: {node.id}
- engine: {node.engine}
- category: {category}
- parameters: {json.dumps(node.parameters, ensure_ascii=False)}

ENGINE TEMPLATE:
{template}

DIAGNOSTICS:
{err_text}

YOUR CODE (line-numbered for reference):
{self._numbered_code(last_code)}

{engine_guidance}

FIX INSTRUCTIONS:
- Read each diagnostic — it tells you the exact line and what went wrong
- Fix ONLY the broken line(s) — do NOT rewrite unrelated code
- Return the complete fixed code. No explanations.
"""

    @staticmethod
    def _numbered_code(code: str) -> str:
        """Add line numbers to code for LLM repair context."""
        lines = code.splitlines()
        return "\n".join(f"{i+1:3d}| {line}" for i, line in enumerate(lines))

    # ---------- LLM Call (via Aider) ----------

    def _call_llm(self, prompt: str, model: str = None) -> str:
        """Call LLM via Aider wrapper with thinking mode for code generation."""
        model_to_use = model or self.model

        try:
            aider = get_aider_llm()
            text = aider.call(prompt, model_to_use, think_tokens="8k")
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

            # Strip ponder comment from repair output (// FIX: Line N — ...)
            if t.startswith("// FIX:"):
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

                # Strip runtime-injected declarations (uniform, in/out, layout)
                # The runtime ALWAYS provides these — LLM copies cause validation errors
                if re.match(r'^\s*uniform\s+(?:float|int|vec[234]|mat[234]|sampler2D|samplerCube|bool)\s+', t):
                    continue
                if re.match(r'^\s*in\s+vec2\s+v_uv\s*;', t):
                    continue
                if re.match(r'^\s*out\s+vec4\s+fragColor\s*;', t):
                    continue
                if re.match(r'^\s*layout\s*\(', t):
                    continue

                # WebGL1 fallbacks
                line = line.replace("gl_FragColor", "fragColor")
                line = re.sub(r"texture2D\s*\(", "texture(", line)

            cleaned.append(line)

        out = "\n".join(cleaned).strip()

        # GLSL: Remove stray escape characters — LLM tool calls sometimes produce
        # literal \n, \t, \r in code strings which are illegal in GLSL
        if engine in {"glsl", "regl"} and out:
            out = out.replace("\\n", "\n").replace("\\t", "\t").replace("\\r", "")

        # GLSL ES: fix { } style array initializers → type[]() constructor syntax
        # e.g. "vec3[] colors = {vec3(1), vec3(2)}" → "vec3[] colors = vec3[](vec3(1), vec3(2))"
        # e.g. "float[] arr = {1.0, 2.0}" → "float[] arr = float[](1.0, 2.0)"
        if engine in {"glsl", "regl"} and out:
            def _fix_array_initializer(m):
                base_type = m.group(1)  # e.g. "vec3", "float", "int"
                var_name = m.group(2)
                contents = m.group(3)
                return f"{base_type}[] {var_name} = {base_type}[]({contents})"
            out = re.sub(
                r'\b(vec[234]|float|int|mat[234])\s*\[\s*\]\s+(\w+)\s*=\s*\{([^}]+)\}',
                _fix_array_initializer, out
            )
            # Also handle sized arrays: vec3[3] colors = {v1, v2, v3} → vec3[3] colors = vec3[](v1, v2, v3)
            def _fix_sized_array_initializer(m):
                base_type = m.group(1)
                size = m.group(2)
                var_name = m.group(3)
                contents = m.group(4)
                return f"{base_type}[{size}] {var_name} = {base_type}[]({contents})"
            out = re.sub(
                r'\b(vec[234]|float|int|mat[234])\s*\[\s*(\d+)\s*\]\s+(\w+)\s*=\s*\{([^}]+)\}',
                _fix_sized_array_initializer, out
            )

        # GLSL: rewrite bare param identifiers to u_<param>
        if engine in {"glsl", "regl"} and out:
            out = self._rewrite_glsl_param_idents(out, node)

        # GLSL: Sanitize any s_vec* calls back to standard constructors (GLSL ES doesn't support overloading)
        if engine in {"glsl", "regl"} and out:
            # Convert s_vec*() back to vec*() since GLSL doesn't support function overloading
            out = re.sub(r'\bs_vec4\s*\(', 'vec4(', out)
            out = re.sub(r'\bs_vec3\s*\(', 'vec3(', out)
            out = re.sub(r'\bs_vec2\s*\(', 'vec2(', out)

            # Rewrite common LLM mistakes: GLSL reserved words used as identifiers
            out = re.sub(r'\bcolor\b', 'col', out)
            # 'input' is a GLSL reserved word — LLMs frequently use it as a var name
            out = re.sub(r'\b(float|vec[234]|int|sampler2D)\s+input\b', r'\1 inp', out)
            out = re.sub(r'\binput\b(?!\s*\()', 'inp', out)  # standalone 'input' → 'inp' (not function calls)

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

                # Pattern 3: Bare GLSL statements (no function wrappers at all)
                # LLMs sometimes output raw expressions/assignments without void main()
                if "void main" not in out:
                    has_func = re.search(r'\b(?:void|float|vec[234]|int|mat[234]|bool)\s+\w+\s*\([^)]*\)\s*\{', out)
                    if not has_func:
                        if 'fragColor' in out:
                            out = f"void main() {{\n{out}\n}}"
                        else:
                            vec4_vars = re.findall(r'\b(\w+)\s*=\s*vec4\s*\(', out)
                            if vec4_vars:
                                out = f"void main() {{\n{out}\nfragColor = {vec4_vars[-1]};\n}}"
                            else:
                                out = f"void main() {{\nvec2 uv = v_uv;\n{out}\nfragColor = vec4(uv, 0.5 + 0.5 * sin(u_time), 1.0);\n}}"

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

    # ── Keyword → code-signature mapping for deterministic semantic check ──
    # Each entry: keyword stem → set of code patterns (any match = pass)
    # Patterns are checked case-insensitively against the generated code.
    _KEYWORD_SIGNATURES: Dict[str, List[str]] = {
        # Noise / procedural generation
        "noise":    ["noise", "fbm", "perlin", "simplex", "hash", "random", "fract(sin"],
        "perlin":   ["perlin", "fbm", "noise", "hash"],
        "simplex":  ["simplex", "fbm", "noise", "snoise"],
        "worley":   ["worley", "voronoi", "cellular", "mindist"],
        "voronoi":  ["voronoi", "worley", "cellular"],
        "fractal":  ["fbm", "fractal", "octave", "lacunarity"],
        "fbm":      ["fbm", "noise", "octave"],
        # Spatial transforms
        "blur":     ["blur", "gaussian", "kernel", "smooth", "convolv"],
        "smooth":   ["smooth", "blur", "lerp", "mix"],
        "distort":  ["distort", "warp", "displace", "offset", "perturb"],
        "displace": ["displace", "displacement", "offset", "warp", "distort", "texture"],
        "warp":     ["warp", "distort", "displace", "twist", "bend"],
        "rotate":   ["rotate", "cos", "sin", "mat2", "angle"],
        "scale":    ["scale", "zoom", "*", "multiply"],
        "transform":["transform", "matrix", "translate", "rotate", "scale"],
        # Color / visual
        "color":    ["color", "rgb", "hsl", "mix", "vec3", "vec4", "palette"],
        "gradient": ["gradient", "mix", "lerp", "ramp", "step", "smoothstep"],
        "blend":    ["blend", "mix", "alpha", "composite", "lerp"],
        "composite":["composite", "blend", "mix", "alpha", "layer"],
        "edge":     ["edge", "sobel", "laplacian", "gradient", "detect"],
        "glow":     ["glow", "bloom", "bright", "emissive", "pow"],
        "shadow":   ["shadow", "dark", "ambient", "occlusion"],
        "light":    ["light", "diffuse", "specular", "phong", "normal", "dot("],
        # Motion / animation
        "wave":     ["sin", "cos", "wave", "oscillat", "frequency"],
        "oscillat": ["sin", "cos", "oscillat", "wave", "period"],
        "particle": ["particle", "velocity", "position", "emit", "point", "life", "arc", "circle", "draw", "sprite"],
        "flow":     ["flow", "velocity", "advect", "direction", "stream", "angle", "sin(", "cos(", "move", "update"],
        "physics":  ["velocity", "accel", "force", "gravity", "mass"],
        # Shapes / geometry
        "circle":   ["circle", "arc", "radius", "distance", "length("],
        "sphere":   ["sphere", "radius", "distance", "length("],
        "rect":     ["rect", "box", "square", "min(", "max(", "abs("],
        "grid":     ["grid", "cell", "floor", "fract", "mod"],
        "pattern":  ["pattern", "repeat", "tile", "mod", "fract", "floor"],
        "mesh":     ["mesh", "vertex", "triangle", "geometry", "buffer"],
        # Data / input
        "audio":    ["audio", "frequency", "fft", "bass", "level", "analyser"],
        "track":    ["track", "landmark", "keypoint", "detect", "pose", "face", "hand"],
        "video":    ["video", "camera", "webcam", "capture", "stream"],
        "midi":     ["midi", "note", "velocity", "channel"],
        # Processing
        "feedback": ["feedback", "previous", "history", "accumulate", "decay", "mix"],
        "tile":     ["tile", "repeat", "fract", "mod", "floor", "grid"],
        "kaleidoscope": ["kaleidoscope", "reflect", "mirror", "atan", "angle", "polar", "mod"],
        "zoom":     ["zoom", "scale", "*", "uv", "center", "distance"],
        "clamp":    ["clamp", "min(", "max(", "saturate", "limit", "threshold"],
        "threshold":["threshold", "step", "clamp", "cutoff"],
        "mask":     ["mask", "alpha", "stencil", "clip", "cutout"],
        "invert":   ["invert", "1.0 -", "1. -", "negate"],
    }

    def _semantic_check(self, node: NodeTensor, code: str) -> List[str]:
        """Deterministic structural check: verify code contains patterns
        expected from the node's keywords and semantic purpose.

        Replaces the LLM-based review with keyword→code-signature matching:
        1. Collect keywords from node.keywords + extracted from semantic_purpose
        2. For each keyword, check if ANY expected code pattern appears
        3. Fail only if NONE of the expected patterns are found for a keyword
           AND the keyword has high confidence (appears in node.keywords, not
           just purpose text where the LLM may use synonyms)

        Design: lenient. Only catches obvious mismatches (e.g., a "noise" node
        with no noise functions). False negatives are better than false positives
        because the prompt already embeds the purpose, and a slightly off-topic
        but functional node is better than retry/failure.

        Returns list of issues (empty = pass).
        """
        if getattr(node, 'is_passthrough', False):
            return []
        meta = node.meta if isinstance(node.meta, dict) else {}
        if meta.get("is_predefined"):
            return []

        code_lower = code.lower()

        # Collect keywords: node.keywords are high-confidence (from SemanticReasoner),
        # purpose-extracted keywords are lower-confidence hints
        kw_from_node = set(k.lower().strip() for k in (getattr(node, 'keywords', []) or []))

        # Extract keyword stems from semantic_purpose text
        # Use word-boundary matching to avoid false positives like "flow" in "flower"
        # Skip generic verbs that the decomposer uses in every purpose string
        _GENERIC_STEMS = {"transform", "scale", "blend", "smooth", "color", "pattern", "grid"}
        purpose = getattr(node, 'semantic_purpose', '') or ''
        kw_from_purpose = set()
        if purpose:
            purpose_lower = purpose.lower()
            for stem in self._KEYWORD_SIGNATURES:
                if stem in _GENERIC_STEMS:
                    continue  # Too generic — appears in most purpose texts
                # Word boundary check: stem must appear as a standalone word,
                # not as a substring of another word (e.g., "flow" in "flower")
                if re.search(r'\b' + re.escape(stem) + r'\b', purpose_lower):
                    kw_from_purpose.add(stem)

        # Check high-confidence keywords (from node.keywords)
        # These MUST have at least one matching code pattern
        errors = []
        missing_high = []
        for kw in kw_from_node:
            # Find the best matching signature entry
            sigs = self._KEYWORD_SIGNATURES.get(kw)
            if not sigs:
                # Try partial match (e.g., "perlin_noise" matches "perlin")
                for stem, patterns in self._KEYWORD_SIGNATURES.items():
                    if stem in kw or kw in stem:
                        sigs = patterns
                        break
            if not sigs:
                continue  # Unknown keyword, skip (don't penalize)

            if not any(pat in code_lower for pat in sigs):
                missing_high.append(kw)

        # Only fail if majority of high-confidence keywords are missing
        # (allows utility nodes with 1 keyword that got satisfied differently)
        if kw_from_node and missing_high:
            match_rate = 1.0 - len(missing_high) / max(len(kw_from_node), 1)
            if match_rate < 0.5 and len(missing_high) >= 2:
                errors.append(
                    f"STRUCTURAL: Code missing patterns for keywords: "
                    f"{', '.join(sorted(missing_high))}. "
                    f"Expected at least some of: "
                    f"{'; '.join(str(self._KEYWORD_SIGNATURES.get(k, [])) for k in missing_high[:3])}"
                )

        # Check purpose-extracted keywords (lower bar — informational only)
        # Only flag if ALL purpose keywords miss AND there are no node keywords
        if not kw_from_node and kw_from_purpose:
            missing_purpose = []
            for kw in kw_from_purpose:
                sigs = self._KEYWORD_SIGNATURES.get(kw, [])
                if sigs and not any(pat in code_lower for pat in sigs):
                    missing_purpose.append(kw)
            if missing_purpose and len(missing_purpose) == len(kw_from_purpose):
                errors.append(
                    f"STRUCTURAL (purpose): Code has no patterns matching purpose keywords: "
                    f"{', '.join(sorted(missing_purpose))}"
                )

        # Minimum complexity check: code should do SOMETHING beyond just pass-through
        # (at least one mathematical/drawing operation)
        engine = (node.engine or "").strip()
        if engine in ("glsl", "regl"):
            # GLSL should have at least one operation beyond gl_FragColor assignment
            math_ops = ["sin", "cos", "pow", "mix", "smoothstep", "clamp",
                        "noise", "fbm", "texture", "dot(", "length(", "normalize",
                        "*", "+", "-", "/", "step", "fract", "floor", "mod",
                        "abs", "min(", "max(", "exp", "sqrt", "distance"]
            if not any(op in code_lower for op in math_ops):
                errors.append("STRUCTURAL: GLSL code contains no mathematical operations")
        elif engine == "canvas2d":
            draw_ops = ["bindimage", "bindgroup", "bindvertex",
                        "bindindex", "bindtexture",
                        "bindsampler", "bindlayout", "fill", "stroke", "arc",
                        "rect", "draw", "putimage", "moveto", "lineto",
                        "beginpath", "closepath", "drawimage",
                        "fillrect", "strokerect", "filltext",
                        "clearrect", "createlineargradient",
                        "createradialgradient", "setpixel", "createimagedata",
                        "putimagedata", "getimagedata"]
            if not any(op in code_lower.replace("_", "") for op in draw_ops):
                errors.append("STRUCTURAL: Canvas2D code contains no drawing operations")
        elif engine in ("p5", "p5js"):
            p5_ops = ["ellipse", "rect", "triangle", "bezier", "curve",
                      "fill", "stroke", "beginshape", "vertex",
                      "endshape", "arc", "line", "quad", "point",
                      "background", "clear", "push", "pop",
                      "translate", "rotate", "scale", "noise"]
            if not any(op in code_lower for op in p5_ops):
                errors.append("STRUCTURAL: P5 code contains no drawing operations (need s.ellipse, s.fill, etc.)")

        return errors

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

// Color space conversions (LLMs call these constantly)
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

        # Check for common texture() misuse patterns that glslangValidator may miss
        # but WebGL2 rejects at runtime
        texture_calls = re.findall(r'texture\s*\(([^)]*)\)', body)
        for call_args in texture_calls:
            args = [a.strip() for a in call_args.split(',')]
            if len(args) != 2:
                errors.append(f"texture() requires exactly 2 arguments (sampler, vec2), got {len(args)}: texture({call_args.strip()})")
            elif args[0] and not re.match(r'^u_\w+$', args[0]):
                # First arg should be a uniform sampler — warn if it's an expression
                if '(' in args[0] or '+' in args[0] or '-' in args[0]:
                    errors.append(f"texture() first argument must be a sampler2D uniform (like u_input0), not an expression: {args[0]}")

        # --- WebGL2-specific static checks (catch what glslangValidator misses) ---

        # 1. Scalar swizzle: someFloat.r / someFloat.x — only valid on vec types
        #    Heuristic: identifier that looks like a float var (declared as float) followed by .rgb etc
        float_decls = set(re.findall(r'\bfloat\s+(\w+)', body))
        for var in float_decls:
            if re.search(rf'\b{re.escape(var)}\s*\.[rgbaxyzw]{{1,4}}\b', body):
                errors.append(
                    f"Scalar swizzle: `{var}` is declared as float but accessed with a swizzle "
                    f"(e.g. {var}.r). Swizzles are only valid on vec2/vec3/vec4."
                )

        # 2. Function redeclaration of GLSL_UTILS builtins injected by runtime
        _glsl_utils_fns = {
            "hash", "hash2", "hash3", "noise", "snoise", "fbm",
            "worley", "simplex", "perlin", "voronoi",
            "rgb2hsv", "hsv2rgb", "hsl2rgb",
        }
        for fn in _glsl_utils_fns:
            if re.search(rf'\b{fn}\s*\(', body) and re.search(rf'\b(?:float|vec\d|bool)\s+{fn}\s*\(', body):
                errors.append(
                    f"Function `{fn}` is already declared in the shader prefix. "
                    f"Do NOT redefine it — call it directly."
                )

        # 3. C-style array initializer: float arr[] = {...} — not valid in GLSL ES 3.00
        if re.search(r'\b\w+\s*\[\s*\d*\s*\]\s*=\s*\{', body):
            errors.append(
                "C-style array initializer `{...}` is not valid in GLSL ES 3.00. "
                "Declare the array then assign each element: arr[0]=...; arr[1]=...;"
            )

        # 4. gl_FragColor is GLSL ES 1.00 / WebGL1 only
        if "gl_FragColor" in body:
            errors.append("Use `fragColor` (out variable), not `gl_FragColor` — this is WebGL2/GLSL ES 3.00.")

        # 5. Struct initializer with braces: MyStruct s = MyStruct(...) is OK but
        #    MyStruct s = {field: val} is not GLSL syntax.
        if re.search(r'=\s*\{[^;]*\}', body):
            errors.append(
                "Brace initializer `= {...}` is not valid GLSL. "
                "Use constructors like vec3(r, g, b) or assign fields individually."
            )

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
            # Prefer bare -S frag first — the #version 300 es in the shader
            # already tells glslangValidator to use GLSL ES 3.00 (matches WebGL2).
            # The --client opengl100 flag uses desktop GL which is more permissive
            # and can mask errors that WebGL2 will reject at runtime.
            for flags in [
                [glslang, "-S", "frag", str(src)],
                [glslang, "--client", "opengl100", "-S", "frag", str(src)],
                [glslang, "--target-env", "opengl", "-S", "frag", str(src)],
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

    def _node_harness_p5(self, body: str) -> str:
        """Syntax-only validation for P5 ESM modules.

        P5 code can't run in Node.js (needs browser + p5 library), so we
        strip imports/exports and check the remaining JS parses correctly.
        """
        # Strip import/export lines so Node can parse the function body
        lines = body.split('\n')
        stripped = []
        for line in lines:
            t = line.strip()
            if t.startswith('import ') or t.startswith('export '):
                continue
            stripped.append(line)
        clean_body = '\n'.join(stripped)
        payload = json.dumps(clean_body)
        return f"""
const code = {payload};
try {{
  new Function(code);
}} catch (e) {{
  console.error('p5: ' + e.message);
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
        engine = (node.engine or "").strip()
        errors: List[str] = []
        if not code.strip():
            return ["js_module: snippet is empty"]

        if "export default" not in code:
            errors.append("js_module: missing `export default` function.")
        # Accept both create(ctx) pattern (js_module) and init(canvas,...) pattern (three_js)
        has_create = "function create" in code
        has_init = "function init" in code
        has_export_fn = "export default" in code and ("function" in code or "async function" in code)
        if not has_create and not has_init and not has_export_fn:
            errors.append("js_module: expected `export default function init(...)` or `export default async function create(ctx) { ... }`.")

        # Detect hallucinated placeholder asset paths — LLMs often generate fake file paths
        _fake_url_patterns = [
            r"""['"`]path/to/""",
            r"""['"`]/path/to/""",
            r"""load\(['"`][^'"`)]*\.(?:jpg|png|jpeg|gif|webp|svg|mp3|wav|ogg)['"`]\)""",
            r"""TextureLoader\(\)\.load\(['"`][^'"`)]*['"`]\)""",
        ]
        for pat in _fake_url_patterns:
            if re.search(pat, code):
                errors.append(
                    "Hallucinated asset path detected (e.g. 'path/to/texture.jpg'). "
                    "NEVER reference external image/audio files — all content must be PROCEDURALLY GENERATED. "
                    "Remove all TextureLoader/loadImage/fetch calls and generate visuals with math/noise instead."
                )
                break

        # P5-specific contract checks
        if engine in ("p5", "p5js"):
            if ".setup" not in code and "s.setup" not in code:
                errors.append("p5: missing s.setup. P5 nodes must define s.setup = () => { ... }")
            if ".draw" not in code and "s.draw" not in code:
                errors.append("p5: missing s.draw. P5 nodes must define s.draw = () => { ... }")
            if "p5" not in code.lower():
                errors.append("p5: missing p5 import. Add: import p5 from 'https://esm.sh/p5@1.9.0'")
            if "sketch.redraw()" not in code:
                errors.append("p5: missing sketch.redraw() in return function. Pipeline uses driven mode — add sketch.redraw() after updating state.")

        # ── Performance checks (p5 and js_module) ─────────────────────────────
        # 1. Nested pixel loops with step ≤ 2 over canvas dimensions.
        #    x++ or x+=1 or x+=2 over width/height = 65k–262k iterations/frame.
        _pixel_fwd = re.compile(
            r'for\s*\([^)]*\bwidth\b[^;]*;\s*(?:\w+\+\+|\w+\s*\+=\s*[12]\b)[^)]*\)'
            r'[\s\S]{0,400}'
            r'for\s*\([^)]*\bheight\b[^;]*;\s*(?:\w+\+\+|\w+\s*\+=\s*[12]\b)[^)]*\)'
        )
        _pixel_rev = re.compile(
            r'for\s*\([^)]*\bheight\b[^;]*;\s*(?:\w+\+\+|\w+\s*\+=\s*[12]\b)[^)]*\)'
            r'[\s\S]{0,400}'
            r'for\s*\([^)]*\bwidth\b[^;]*;\s*(?:\w+\+\+|\w+\s*\+=\s*[12]\b)[^)]*\)'
        )
        if _pixel_fwd.search(code) or _pixel_rev.search(code):
            errors.append(
                "PERFORMANCE: nested pixel loop detected (for x<width; x++ or x+=2 inside for y<height). "
                "Even stride-2 loops iterate 65,536+ times per frame and stall the browser. "
                "REPLACE with: s.background(r,g,b) for fills; pre-allocate an array of ≤500 random "
                "positions ONCE in init and iterate it in s.draw(). Total draw calls per frame must stay under 1000."
            )

        # 2. getImageData inside s.draw — implies per-pixel CPU work every frame.
        #    Only flag when paired with a nested loop (the expensive combination).
        if re.search(r'getImageData', code) and (
            re.search(r's\.draw\s*=[\s\S]{0,3000}getImageData', code) and
            re.search(r'for\s*\(', code)
        ):
            errors.append(
                "PERFORMANCE: getImageData() inside s.draw() with a for-loop forces CPU pixel readback every frame. "
                "This is extremely slow on hardware-accelerated canvases. "
                "REPLACE with GPU-side effects: use s.drawingContext.globalCompositeOperation, s.tint(), "
                "or GLSL engine for per-pixel color operations instead."
            )

        # 3. createGraphics inside s.draw — allocates a new GPU canvas every frame (memory leak).
        _draw_body = re.search(r's\.draw\s*=\s*\(\s*\)\s*=>\s*\{([\s\S]*)', code)
        if _draw_body and 'createGraphics' in _draw_body.group(1):
            errors.append(
                "MEMORY LEAK: createGraphics() called inside s.draw(). "
                "This allocates a new off-screen canvas every frame. After 300 frames, 300 canvases consume all GPU RAM and crash the browser. "
                "FIX: declare 'let img;' at module scope, assign 'img = s.createGraphics(width, height)' "
                "inside s.setup() ONCE, then call img.clear() at the top of s.draw() to reuse it."
            )

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
