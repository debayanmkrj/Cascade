"""
LowKCDR Web Application
Hybrid Python backend + JavaScript frontend

Features:
- Phase 1: Image search, CLIP clustering, brand extraction, RAG, design assistant
- Phase 2: Architect/Mason agents for node generation
- Node execution: JS-based nodes in browser (replaces Taichi)

No npm dependencies - pure Python backend serving static JS frontend
"""

import os
import sys
import json
import traceback
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid
import requests

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import Phase 1 components
from config import (
    OLLAMA_URL, MODEL_NAME, MODEL_NAME_REASONING, MODEL_NAME_CODING,
    PEXELS_API_KEY, EMOTION_LABELS, BRAND_ATTRIBUTE_LABELS,
    DEFAULT_LEVEL_WEIGHTS, MODE_FACTORS
)
from data_types import RawInputBundle, ImageRef

# Lazy loading for heavy components
_phase1_pipeline = None
_visual_clusterer = None

# ---------------------------------------------------------------------------
# Copilot async task store
# ---------------------------------------------------------------------------
_copilot_tasks: Dict[str, Dict] = {}  # task_id → {status, result}
_copilot_tasks_lock = threading.Lock()
_image_searcher = None
_design_copilot = None
_phase2_pipeline = None

def get_phase1_pipeline():
    global _phase1_pipeline
    if _phase1_pipeline is None:
        print("[INIT] Loading Phase 1 Pipeline (this may take a moment)...")
        from phase1.phase1_core import Phase1Pipeline
        _phase1_pipeline = Phase1Pipeline()
    return _phase1_pipeline

def get_visual_clusterer():
    global _visual_clusterer
    if _visual_clusterer is None:
        print("[INIT] Loading Visual Clusterer...")
        from phase1.visual_clustering import VisualClusterer
        _visual_clusterer = VisualClusterer()
    return _visual_clusterer

def get_image_searcher():
    global _image_searcher
    if _image_searcher is None:
        print("[INIT] Loading Image Searcher...")
        from phase1.image_search import ImageSearcher
        _image_searcher = ImageSearcher()
    return _image_searcher

def get_design_copilot():
    global _design_copilot
    if _design_copilot is None:
        print("[INIT] Loading Design Copilot...")
        from phase2.design_copilot import DesignCopilot
        _design_copilot = DesignCopilot()
    return _design_copilot

def get_phase2_pipeline():
    global _phase2_pipeline
    if _phase2_pipeline is None:
        print("[INIT] Loading Phase 2 Pipeline...")
        from phase2.pipeline import Phase2Pipeline
        _phase2_pipeline = Phase2Pipeline()
    return _phase2_pipeline


# Flask app setup
app = Flask(__name__,
            template_folder='templates',
            static_folder='static')
CORS(app)

# Config
UPLOAD_FOLDER = Path(__file__).parent / 'uploads'
UPLOAD_FOLDER.mkdir(exist_ok=True)
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max

# Session storage
sessions: Dict[str, Dict] = {}


def _reconstruct_session_from_project(project: dict, session_id: str) -> dict:
    """Synthesize a minimal session JSON from an existing output project.

    Used when the original session file is no longer on disk but the frontend
    still has the project JSON (e.g. loaded from outputs/).  Allows the copilot
    and Phase 2 to operate on a loaded project without "Run Phase 1 first" errors.
    """
    brief = project.get("design_brief", {})
    nodes = project.get("nodes", [])
    archetypes = [
        {
            "id": n["id"],
            "category": n.get("category", ""),
            "role": n.get("role", "process"),
            "engine": n.get("engine", "glsl"),
        }
        for n in nodes
    ]
    return {
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(),
        "input": {"prompt_text": brief.get("prompt_text", "")},
        "brief": {
            "visual_palette": brief.get("visual_palette", {}),
            "node_archetypes": archetypes,
        },
        "phase2_context": {
            "essence": brief.get("essence", brief.get("prompt_text", "")),
            "node_archetypes": archetypes,
            "visual_palette": brief.get("visual_palette", {}),
        },
    }


# ============================================================================
# ROUTES - Static Files
# ============================================================================

@app.route('/')
def index():
    """Serve main application"""
    try:
        from phase2.agents.mason import PREDEFINED_CODE
        predefined_codes = {k: v.get('code', '') for k, v in PREDEFINED_CODE.items()}
    except Exception:
        predefined_codes = {}
    return render_template('index.html', predefined_codes=predefined_codes)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# ============================================================================
# API - Health & Status
# ============================================================================

@app.route('/api/health')
def health():
    """Health check endpoint"""
    ollama_ok = False
    try:
        resp = requests.get('http://localhost:11434/api/tags', timeout=2)
        ollama_ok = resp.ok
    except:
        pass

    return jsonify({
        'status': 'ok',
        'ollama': ollama_ok,
        'models': {
            'general': MODEL_NAME,
            'reasoning': MODEL_NAME_REASONING,
            'coding': MODEL_NAME_CODING
        }
    })


# ============================================================================
# API - Image Search & Clustering (Phase 1)
# ============================================================================

@app.route('/api/search-images', methods=['POST'])
def search_images():
    """Search for images using Pexels API"""
    try:
        data = request.json
        query = data.get('query', '')
        num_results = data.get('num_results', 20)

        if not query:
            return jsonify({'success': False, 'error': 'No query provided'})

        searcher = get_image_searcher()
        images = searcher.search(query, num_results=num_results)

        # Format ImageRef objects for frontend
        formatted = []
        for img in images:
            # ImageRef has: id, url, thumbnail_url, source, mood, metadata
            photographer = ''
            if hasattr(img, 'metadata') and img.metadata:
                photographer = img.metadata.get('photographer', '')

            formatted.append({
                'id': getattr(img, 'id', str(uuid.uuid4())),
                'url': getattr(img, 'url', ''),
                'thumbnail_url': getattr(img, 'thumbnail_url', getattr(img, 'url', '')),
                'photographer': photographer,
                'mood': getattr(img, 'mood', 'neutral')
            })

        return jsonify({'success': True, 'images': formatted})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/upload-images', methods=['POST'])
def upload_images():
    """Upload local images"""
    try:
        if 'images' not in request.files:
            return jsonify({'success': False, 'error': 'No images in request'})

        files = request.files.getlist('images')
        uploaded = []

        for file in files:
            if file and file.filename:
                filename = secure_filename(file.filename)
                unique_name = f"{uuid.uuid4().hex[:8]}_{filename}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
                file.save(filepath)

                uploaded.append({
                    'id': unique_name,
                    'url': f'/uploads/{unique_name}',
                    'thumbnail_url': f'/uploads/{unique_name}',
                    'photographer': 'Local Upload',
                    'mood': 'neutral'
                })

        return jsonify({'success': True, 'images': uploaded})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/cluster-images', methods=['POST'])
def cluster_images():
    """Cluster images using CLIP embeddings"""
    try:
        data = request.json
        images = data.get('images', [])
        n_clusters = data.get('n_clusters', 5)

        if not images:
            return jsonify({'success': False, 'error': 'No images provided'})

        # Convert to ImageRef format
        image_refs = []
        for img in images:
            url = img.get('url', img.get('thumbnail_url', ''))
            thumb = img.get('thumbnail_url', url)
            if url.startswith('/uploads/'):
                url = str(UPLOAD_FOLDER / url.replace('/uploads/', ''))
                thumb = url
            image_refs.append(ImageRef(
                id=img.get('id', str(uuid.uuid4())),
                url=url,
                thumbnail_url=thumb,
                source='upload' if '/uploads/' in img.get('url', '') else 'reference',
                mood=img.get('mood', 'neutral'),
                metadata={'photographer': img.get('photographer', '')}
            ))

        # Cluster
        clusterer = get_visual_clusterer()
        result = clusterer.cluster_images(image_refs, n_clusters=min(n_clusters, len(images)))

        # Format response
        clusters = []
        colors = ['#f97316', '#22c55e', '#3b82f6', '#a855f7', '#ec4899', '#eab308']

        for i, cluster in enumerate(result.get('clusters', [])):
            clusters.append({
                'name': cluster.get('theme', f'Cluster {i+1}'),
                'count': len(cluster.get('images', [])),
                'color': colors[i % len(colors)],
                'images': cluster.get('images', [])
            })

        # Add cluster info to images
        formatted_images = []
        for img in images:
            img_copy = dict(img)
            # Find which cluster this image belongs to
            for ci, cluster in enumerate(clusters):
                if any(ci_img.get('id') == img.get('id') for ci_img in cluster.get('images', [])):
                    img_copy['cluster'] = ci
                    break
            formatted_images.append(img_copy)

        return jsonify({
            'success': True,
            'clusters': clusters,
            'images': formatted_images,
            'mood': result.get('mood', 'neutral'),
            'palette': result.get('palette', [])
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


# ============================================================================
# API - Brief Generation (Phase 1 Full Pipeline)
# ============================================================================

@app.route('/api/generate-full', methods=['POST'])
def generate_full():
    """Run full Phase 1 pipeline to generate design brief"""
    try:
        data = request.json
        prompt = data.get('prompt', '')
        images = data.get('images', [])
        tension_value = data.get('tension_value', 0.5)
        llm_seed = data.get('llm_seed', 42)

        if not prompt:
            return jsonify({'success': False, 'error': 'No prompt provided'})

        print(f"\n[GENERATE] Starting pipeline with prompt: {prompt[:50]}...")

        # Convert images to ImageRef
        image_refs = []
        for img in images:
            url = img.get('url', '')
            thumb = img.get('thumbnail_url', url)
            if url.startswith('/uploads/'):
                url = str(UPLOAD_FOLDER / url.replace('/uploads/', ''))
                thumb = url
            image_refs.append(ImageRef(
                id=img.get('id', str(uuid.uuid4())),
                url=url,
                thumbnail_url=thumb,
                source='upload' if '/uploads/' in img.get('url', '') else 'reference',
                mood=img.get('mood', 'neutral'),
                metadata={'photographer': img.get('photographer', '')}
            ))

        # Create input bundle
        input_bundle = RawInputBundle(
            prompt_text=prompt,
            D_global=tension_value,
            user_mode='aesthetic' if tension_value > 0.5 else 'functional',
            reference_images=image_refs,
            custom_images=[],
            seed=llm_seed
        )

        # Run Phase 1 pipeline
        pipeline = get_phase1_pipeline()
        node_brief = pipeline.execute(input_bundle)

        # Convert to frontend format
        brief = _convert_brief_to_frontend(node_brief)

        # Generate visualization data
        visualization = _generate_visualization(node_brief, images)

        # Save session JSON for Phase 2
        session_path = pipeline.save_session_json(input_bundle, node_brief)

        # Store in memory
        session_id = str(uuid.uuid4())
        sessions[session_id] = {
            'brief': brief,
            'node_brief': node_brief,
            'session_path': session_path,
            'created': datetime.now().isoformat()
        }

        return jsonify({
            'success': True,
            'session_id': session_id,
            'brief': brief,
            'visualization': visualization,
            'stats': {
                'concepts': len(node_brief.node_archetypes) if hasattr(node_brief, 'node_archetypes') else 0,
                'nodes': len(node_brief.ui_controls) if hasattr(node_brief, 'ui_controls') else 0
            }
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


def _convert_brief_to_frontend(node_brief) -> Dict:
    """Convert NodeBrief to frontend-compatible format"""
    brief = {
        'essence': {
            'statement': getattr(node_brief, 'essence', 'Visual design workflow')
        },
        'pompelli': {
            'project_name': 'Generated Workflow',
            'node_workflow': {
                'input_nodes': [],
                'processing_nodes': [],
                'output_nodes': []
            },
            'visual_identity': {
                'color_palette': [],
                'typography': {'primary': 'Inter', 'secondary': 'Roboto Mono'}
            },
            'design_principles': [],
            'shape_language': {
                'primary': 'geometric',
                'characteristics': [],
                'rationale': ''
            }
        }
    }

    # Extract nodes from archetypes
    if hasattr(node_brief, 'node_archetypes'):
        for arch in node_brief.node_archetypes:
            role = getattr(arch, 'role', 'process')
            node_entry = {
                'id': getattr(arch, 'id', getattr(arch, 'category', 'node')),
                'engine': getattr(arch, 'engine', '')
            }

            if role == 'input':
                brief['pompelli']['node_workflow']['input_nodes'].append(node_entry)
            elif role == 'output':
                brief['pompelli']['node_workflow']['output_nodes'].append(node_entry)
            else:
                brief['pompelli']['node_workflow']['processing_nodes'].append(node_entry)

    # Extract colors
    if hasattr(node_brief, 'visual_palette'):
        palette = node_brief.visual_palette
        if hasattr(palette, 'primary_colors'):
            brief['pompelli']['visual_identity']['color_palette'].extend(palette.primary_colors)
        if hasattr(palette, 'accent_colors'):
            brief['pompelli']['visual_identity']['color_palette'].extend(palette.accent_colors)

    # Extract shapes
    if hasattr(node_brief, 'visual_palette'):
        palette = node_brief.visual_palette
        if hasattr(palette, 'shapes'):
            brief['pompelli']['shape_language']['characteristics'] = palette.shapes

    return brief


def _generate_visualization(node_brief, images: List) -> Dict:
    """Generate visualization data for funnel diagram"""
    stages = []

    # Input stage
    input_nodes = []
    for img in images[:8]:
        input_nodes.append({'label': 'Image', 'type': 'image'})
    input_nodes.append({'label': 'Prompt', 'type': 'text'})
    stages.append({'name': 'inputs', 'nodes': input_nodes})

    # Abstraction stage
    abs_nodes = []
    if hasattr(node_brief, 'creative_levels'):
        levels = node_brief.creative_levels
        abs_nodes.append({'label': 'Surface', 'level': 'visceral', 'weight': getattr(levels, 'surface', 0.4)})
        abs_nodes.append({'label': 'Flow', 'level': 'behavioral', 'weight': getattr(levels, 'flow', 0.3)})
        abs_nodes.append({'label': 'Narrative', 'level': 'reflective', 'weight': getattr(levels, 'narrative', 0.3)})
    stages.append({'name': 'abstraction', 'nodes': abs_nodes})

    # Tensions stage
    tension_nodes = [
        {'label': 'T1', 'conflict': 'form-function'},
        {'label': 'T2', 'conflict': 'complexity-simplicity'}
    ]
    stages.append({'name': 'tensions', 'nodes': tension_nodes})

    # Synthesis stage
    synth_nodes = [{'label': 'Design Brief', 'is_center': True}]
    stages.append({'name': 'synthesis', 'nodes': synth_nodes})

    return {'stages': stages, 'flows': []}


# ============================================================================
# API - Chat / Design Assistant
# ============================================================================

def _resolve_session_json(session_id: Optional[str], project: Optional[Dict]) -> Optional[Dict]:
    """Load session JSON from disk, or reconstruct and persist from project if not found."""
    session_json = None
    if session_id:
        sessions_dir = Path(__file__).parent / 'sessions'
        for f in sorted(sessions_dir.glob('*.json'), reverse=True):
            try:
                sj = json.loads(f.read_text(encoding='utf-8'))
                if sj.get('session_id') == session_id:
                    session_json = sj
                    break
            except Exception:
                pass

    if not session_json and session_id and project:
        session_json = _reconstruct_session_from_project(project, session_id)
        sessions_dir = Path(__file__).parent / 'sessions'
        ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        reconstructed_path = sessions_dir / f"{ts}_session_{session_id[:8]}.json"
        try:
            reconstructed_path.write_text(json.dumps(session_json, indent=2), encoding='utf-8')
            print(f"[chat] Reconstructed session from project → {reconstructed_path.name}")
        except Exception as e:
            print(f"[chat] Could not save reconstructed session: {e}")

    return session_json


def _save_session_if_needed(session_id: Optional[str], result: Dict) -> None:
    """Write updated session JSON back to disk when pipeline re-run is requested."""
    if not (session_id and result.get('session_json') and result.get('pipeline_needed')):
        return
    sessions_dir = Path(__file__).parent / 'sessions'
    for f in sorted(sessions_dir.glob('*.json'), reverse=True):
        try:
            sj = json.loads(f.read_text(encoding='utf-8'))
            if sj.get('session_id') == session_id:
                f.write_text(json.dumps(result['session_json'], indent=2), encoding='utf-8')
                break
        except Exception:
            pass


def _run_copilot_task(task_id: str, message: str, session_json: Optional[Dict],
                      project: Optional[Dict], selected_node_id: Optional[str],
                      session_id: Optional[str]) -> None:
    """Background worker: runs the copilot and stores the result in _copilot_tasks."""
    try:
        copilot = get_design_copilot()
        result = copilot.process(message, session_json, project, selected_node_id)
        _save_session_if_needed(session_id, result)
        payload = {
            'status': 'done',
            'success': True,
            'response': result['response'],
            'project': result.get('project'),
            'pipeline_needed': result.get('pipeline_needed', False),
        }
    except Exception as e:
        traceback.print_exc()
        payload = {'status': 'done', 'success': False, 'error': str(e)}

    with _copilot_tasks_lock:
        _copilot_tasks[task_id] = payload


@app.route('/api/chat', methods=['POST'])
def chat():
    """Design Assistant Copilot — starts a background task and returns immediately.

    The frontend should poll /api/chat-result/<task_id> until status == 'done'.
    Returns: {task_id, status: 'processing', response: 'Working on it...'}
    """
    try:
        data = request.json
        message = data.get('message', '')
        session_id = data.get('session_id')
        selected_node_id = data.get('selected_node_id')
        project = data.get('project')

        if not message:
            return jsonify({'success': False, 'error': 'No message provided'})

        session_json = _resolve_session_json(session_id, project)

        task_id = str(uuid.uuid4())
        with _copilot_tasks_lock:
            _copilot_tasks[task_id] = {'status': 'processing'}

        t = threading.Thread(
            target=_run_copilot_task,
            args=(task_id, message, session_json, project, selected_node_id, session_id),
            daemon=True,
        )
        t.start()

        return jsonify({'success': True, 'task_id': task_id, 'status': 'processing',
                        'response': 'Working on it...'})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/chat-result/<task_id>', methods=['GET'])
def chat_result(task_id: str):
    """Poll for the result of a /api/chat task.

    Returns:
      {status: 'processing'} while still running.
      {status: 'done', success, response, project, pipeline_needed} when complete.
      {status: 'not_found'} if task_id is unknown.
    """
    with _copilot_tasks_lock:
        task = _copilot_tasks.get(task_id)

    if task is None:
        return jsonify({'status': 'not_found'})

    if task.get('status') == 'processing':
        return jsonify({'status': 'processing'})

    # Done — return result and clean up
    with _copilot_tasks_lock:
        _copilot_tasks.pop(task_id, None)

    return jsonify(task)


@app.route('/api/save-project', methods=['POST'])
def save_project():
    """Save current project JSON — called after manual UI edits (position, param, delete).

    Writes in-place to the same file on every call (no new timestamps).
    The client passes back `output_path` from the previous save response so we know
    which file to overwrite.  On the very first save the client passes null and we
    derive a stable session-keyed filename: project_{sid8}.json.
    """
    try:
        data = request.json
        project = data.get('project')
        session_id = data.get('session_id', 'manual')
        output_path = data.get('output_path')  # path returned by a previous save

        if not project:
            return jsonify({'success': False, 'error': 'No project provided'})

        output_dir = Path(__file__).parent / 'outputs'
        output_dir.mkdir(exist_ok=True)

        if output_path:
            target = Path(output_path)
            # Safety: must remain inside outputs/
            try:
                target.resolve().relative_to(output_dir.resolve())
            except ValueError:
                target = output_dir / target.name
        else:
            # First save: stable filename keyed to session (no timestamp)
            sid_short = str(session_id)[:8] if session_id else 'manual'
            target = output_dir / f"project_{sid_short}.json"

        target.write_text(json.dumps(project, indent=2), encoding='utf-8')
        return jsonify({'success': True, 'path': str(target)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


# ============================================================================
# API - Phase 2 Node Generation
# ============================================================================

@app.route('/api/generate-nodes', methods=['POST'])
def generate_nodes():
    """Generate executable JS nodes from brief using Phase 2 agents"""
    try:
        data = request.json
        brief = data.get('brief', {})
        session_id = data.get('session_id', '')

        # Get node workflow from brief
        workflow = brief.get('pompelli', {}).get('node_workflow', {})
        all_nodes = (
            workflow.get('input_nodes', []) +
            workflow.get('processing_nodes', []) +
            workflow.get('output_nodes', [])
        )

        if not all_nodes:
            return jsonify({'success': False, 'error': 'No nodes in workflow'})

        # Generate JS node specifications
        nodes = []
        for i, node_name in enumerate(all_nodes):
            node_spec = _generate_js_node_spec(node_name, i)
            nodes.append(node_spec)

        # Generate connections (linear for now)
        connections = []
        for i in range(len(nodes) - 1):
            connections.append({
                'from': nodes[i]['id'],
                'from_output': 0,
                'to': nodes[i + 1]['id'],
                'to_input': 0
            })

        return jsonify({
            'success': True,
            'nodes': nodes,
            'connections': connections
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


def _generate_js_node_spec(node_name: str, index: int) -> Dict:
    """Generate JS node specification"""
    node_type = _classify_node_type(node_name)

    return {
        'id': f'node_{index}',
        'name': node_name,
        'type': node_type,
        'position': {'x': 100 + (index % 4) * 200, 'y': 100 + (index // 4) * 150},
        'inputs': _get_node_inputs(node_type),
        'outputs': _get_node_outputs(node_type),
        'params': _get_node_params(node_type, node_name)
    }


def _classify_node_type(name: str) -> str:
    """Classify node into type category"""
    name_lower = name.lower()

    # Input nodes
    if any(x in name_lower for x in ['input', 'webcam', 'camera', 'video', 'image', 'audio', 'mic']):
        return 'input'

    # Output nodes
    if any(x in name_lower for x in ['output', 'render', 'display', 'export', 'record']):
        return 'output'

    # Generator nodes
    if any(x in name_lower for x in ['noise', 'gradient', 'pattern', 'oscillator', 'generator', 'emitter']):
        return 'generator'

    # Effect nodes
    if any(x in name_lower for x in ['blur', 'glow', 'bloom', 'distort', 'chromatic', 'feedback', 'delay']):
        return 'effect'

    # Color nodes
    if any(x in name_lower for x in ['color', 'hue', 'saturation', 'brightness', 'contrast', 'tint']):
        return 'color'

    # Math nodes
    if any(x in name_lower for x in ['math', 'add', 'multiply', 'mix', 'blend', 'lerp']):
        return 'math'

    return 'process'


def _get_node_inputs(node_type: str) -> List[Dict]:
    """Get input ports for node type"""
    if node_type == 'input':
        return []
    elif node_type == 'generator':
        return [{'name': 'time', 'type': 'number'}]
    elif node_type == 'output':
        return [{'name': 'texture', 'type': 'texture'}]
    else:
        return [{'name': 'input', 'type': 'texture'}]


def _get_node_outputs(node_type: str) -> List[Dict]:
    """Get output ports for node type"""
    if node_type == 'output':
        return []
    else:
        return [{'name': 'output', 'type': 'texture'}]


def _get_node_params(node_type: str, name: str) -> List[Dict]:
    """Get parameters for node type"""
    params = []

    if node_type == 'generator':
        params = [
            {'name': 'scale', 'type': 'float', 'default': 1.0, 'min': 0.1, 'max': 10.0},
            {'name': 'speed', 'type': 'float', 'default': 1.0, 'min': 0.0, 'max': 5.0}
        ]
    elif node_type == 'effect':
        params = [
            {'name': 'intensity', 'type': 'float', 'default': 1.0, 'min': 0.0, 'max': 2.0},
            {'name': 'radius', 'type': 'float', 'default': 5.0, 'min': 0.0, 'max': 50.0}
        ]
    elif node_type == 'color':
        params = [
            {'name': 'amount', 'type': 'float', 'default': 1.0, 'min': 0.0, 'max': 2.0}
        ]

    return params


# ============================================================================
# API - Phase 2 Polyglot Pipeline
# ============================================================================

@app.route('/api/phase2/execute', methods=['POST'])
def phase2_execute():
    """Run Phase 2 pipeline: Architect + Mason on a Phase 1 session."""
    try:
        data = request.json
        session_id = data.get('session_id', '')
        session_path = data.get('session_path', '')

        # Try loading from file, in-memory session, or disk scan by session_id
        session_json = None
        if session_path and Path(session_path).exists():
            with open(session_path, 'r') as f:
                session_json = json.load(f)
        elif session_id:
            # 1. Check disk first (covers loaded output files whose session is on disk)
            sessions_dir = Path(__file__).parent / 'sessions'
            for f in sorted(sessions_dir.glob('*.json'), reverse=True):
                try:
                    sj = json.loads(f.read_text(encoding='utf-8'))
                    if sj.get('session_id') == session_id:
                        session_json = sj
                        break
                except Exception:
                    pass

        if not session_json and session_id and session_id in sessions:
            # 2. Fall back to in-memory session (active pipeline run)
            sess = sessions[session_id]
            node_brief = sess.get('node_brief')
            if node_brief and hasattr(node_brief, 'to_dict'):
                session_json = {
                    'session_id': session_id,
                    'phase2_context': {
                        'essence': node_brief.essence,
                        'node_archetypes': [
                            n.to_dict() if hasattr(n, 'to_dict') else {'id': n.id, 'meta': n.meta}
                            for n in node_brief.node_archetypes
                        ],
                        'ui_controls': [
                            {
                                'id': c.id, 'type': c.type, 'label': c.label,
                                'parameters': c.parameters, 'bindings': c.bindings,
                                'creative_level': c.creative_level
                            } for c in node_brief.ui_controls
                        ],
                        'visual_palette': {
                            'primary_colors': node_brief.visual_palette.primary_colors,
                            'accent_colors': node_brief.visual_palette.accent_colors,
                            'shapes': node_brief.visual_palette.shapes,
                            'motion_words': node_brief.visual_palette.motion_words
                        },
                        'creative_levels': {
                            'surface': node_brief.creative_levels.surface,
                            'flow': node_brief.creative_levels.flow,
                            'narrative': node_brief.creative_levels.narrative
                        },
                        'divergence': {
                            'D_surface': node_brief.divergence.D_surface,
                            'D_flow': node_brief.divergence.D_flow,
                            'D_narrative': node_brief.divergence.D_narrative
                        }
                    }
                }

        if not session_json:
            return jsonify({'success': False, 'error': 'No valid session found. Run Phase 1 first.'})

        # Run Phase 2
        pipeline = get_phase2_pipeline()
        project_json = pipeline.execute(session_json)

        # Save output
        output_path = pipeline.save_project(project_json, session_id or 'unnamed')

        return jsonify({
            'success': True,
            'project': project_json,
            'output_path': output_path
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/phase2/sessions', methods=['GET'])
def list_sessions():
    """List available Phase 1 session files for Phase 2."""
    try:
        from config import SESSIONS_DIR
        session_files = list(Path(SESSIONS_DIR).glob('*_session_*.json')) + list(Path(SESSIONS_DIR).glob('session_*.json'))
        result = []
        for sf in sorted(session_files, reverse=True)[:20]:
            with open(sf, 'r') as f:
                data = json.load(f)
            result.append({
                'path': str(sf),
                'session_id': data.get('session_id', ''),
                'timestamp': data.get('timestamp', ''),
                'prompt': data.get('input', {}).get('prompt_text', '')[:80]
            })
        return jsonify({'success': True, 'sessions': result})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/phase2/add-node', methods=['POST'])
def add_predefined_node_to_session():
    """Add a predefined node to the session for Phase 2 processing.

    This endpoint allows the frontend to add drag-and-drop predefined nodes
    (Task 4 types) to the current session so they can be processed by Mason.
    """
    try:
        data = request.json
        session_id = data.get('session_id', '')
        node_data = data.get('node', {})

        if not session_id or session_id not in sessions:
            return jsonify({'success': False, 'error': 'Invalid session_id'})

        if not node_data:
            return jsonify({'success': False, 'error': 'No node data provided'})

        sess = sessions[session_id]
        node_brief = sess.get('node_brief')

        if not node_brief:
            return jsonify({'success': False, 'error': 'No node_brief in session'})

        # Import NodeTensor for creating the archetype
        from phase2.data_types import NodeTensor

        # Create a NodeTensor from the predefined node data
        meta = node_data.get('meta', {})
        new_node = NodeTensor(
            id=node_data.get('id', f'n{len(node_brief.node_archetypes)}'),
            meta=meta,
            grid_position=tuple(node_data.get('grid_position', [0, 0, 0])),
            grid_size=tuple(node_data.get('grid_size', [1, 1])),
            engine=node_data.get('engine', 'glsl'),
            code_snippet=node_data.get('code_snippet', ''),
            parameters=node_data.get('parameters', {}),
            input_nodes=node_data.get('input_nodes', []),
            keywords=meta.get('keywords', [meta.get('category', '')]),
            architect_approved=True,
            mason_approved=False
        )

        # Add to the session's node archetypes
        node_brief.node_archetypes.append(new_node)

        print(f"[API] Added predefined node {new_node.id} (category={meta.get('category')}) to session {session_id}")

        return jsonify({
            'success': True,
            'node_id': new_node.id,
            'message': f'Added {meta.get("label", new_node.id)} to session'
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/phase2/add-predefined', methods=['POST'])
def add_predefined_node_quick():
    """Add a predefined node - TRUE plug-and-play, NO Mason regeneration.

    Predefined nodes have pre-validated code templates that should NOT be regenerated.
    This endpoint:
    1. Places the node at optimal Z-layer based on role
    2. Uses the predefined template code directly (NO Mason)
    3. Returns the node ready to execute
    """
    try:
        data = request.json
        session_id = data.get('session_id', '')
        node_data = data.get('node', {})
        current_project = data.get('current_project', {})

        if not session_id:
            return jsonify({'success': False, 'error': 'No session_id provided'})

        # Get existing nodes from current project
        existing_nodes = current_project.get('nodes', [])

        # Determine optimal Z-layer based on node role
        meta = node_data.get('meta', {})
        role = meta.get('role', 'process')
        is_source = meta.get('is_source', False)
        category = meta.get('category', '')

        # Calculate Z-layer placement
        max_z = max([n.get('grid_position', [0, 0, 0])[2] for n in existing_nodes], default=0)

        if role == 'input' or is_source:
            target_z = 0  # Input nodes at Z=0
        elif role == 'output':
            target_z = max_z + 1  # Output at top
        else:
            target_z = max(1, max_z)  # Process nodes in middle/top

        # Update node position
        grid_pos = node_data.get('grid_position', [0, 0, 0])
        grid_pos[2] = target_z
        node_data['grid_position'] = grid_pos

        # Get predefined template code - NO Mason regeneration
        from phase2.agents.mason import PREDEFINED_CODE

        predefined = PREDEFINED_CODE.get(category, {})
        code_snippet = predefined.get('code', '')

        # Merge predefined default parameters with any user-provided ones
        params = {}
        if predefined.get("parameters"):
            for param_name, param_info in predefined["parameters"].items():
                if isinstance(param_info, dict):
                    params[param_name] = param_info.get("default", 0)
                else:
                    params[param_name] = param_info
        # Override with user-provided params
        params.update(node_data.get('parameters', {}))

        # Build result node directly - plug and play
        result_node = {
            'id': node_data.get('id'),
            'name': node_data.get('id'),
            'engine': node_data.get('engine', predefined.get('engine', 'glsl')),
            'meta': meta,
            'grid_position': grid_pos,
            'grid_size': node_data.get('grid_size', [1, 1]),
            'code_snippet': code_snippet,
            'parameters': params,
            'input_nodes': node_data.get('input_nodes', []),
            'keywords': meta.get('keywords', [category]),
            'enabled': True,
            'architect_approved': True,
            'mason_approved': True,  # Predefined = already validated
            'validation_errors': []
        }

        print(f"[API] Added predefined node {result_node['id']} (category={category}) at Z={target_z} - PLUG & PLAY (no Mason)")

        return jsonify({
            'success': True,
            'node': result_node,
            'message': f'Added {meta.get("label", result_node["id"])} at Z-layer {target_z}'
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


# ============================================================================
# API - LLM Direct Access
# ============================================================================

@app.route('/api/llm/generate', methods=['POST'])
def llm_generate():
    """Direct LLM generation endpoint"""
    try:
        data = request.json
        prompt = data.get('prompt', '')
        model = data.get('model', MODEL_NAME)

        if not prompt:
            return jsonify({'success': False, 'error': 'No prompt provided'})

        response = requests.post(
            OLLAMA_URL,
            json={
                'model': model,
                'prompt': prompt,
                'stream': False,
                'options': {'temperature': 0.7, 'num_predict': 2000}
            },
            timeout=120
        )

        if response.ok:
            result = response.json()
            return jsonify({
                'success': True,
                'response': result.get('response', '')
            })
        else:
            return jsonify({'success': False, 'error': f'LLM error: {response.status_code}'})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


# ============================================================================
# API - Runtime Error Inspector (LLM-based fixes)
# ============================================================================

# Lazy load runtime inspector
_runtime_inspector = None

def get_runtime_inspector():
    global _runtime_inspector
    if _runtime_inspector is None:
        from phase2.agents.runtime_inspector import RuntimeInspector
        _runtime_inspector = RuntimeInspector()
    return _runtime_inspector


@app.route('/api/runtime-errors', methods=['POST'])
def report_runtime_errors():
    """
    Receive runtime errors from browser and attempt LLM-based fixes.
    Also handles passthrough node retries and fallback node generation.

    Expected payload:
    {
        "session_id": "...",
        "errors": [
            {
                "node_id": "node_4_n4",
                "category": "color_grade",
                "engine": "glsl",
                "error_message": "'u_amplitude' : undeclared identifier",
                "code_snippet": "...",
                "parameters": {"hue_shift": 0, ...},
                "input_nodes": ["node_3_n3"],
                "is_passthrough": false,  # Optional - indicates if this was a passthrough node
                "keywords": ["color", "grade"],  # Optional - semantic keywords
                "timestamp": "..."
            }
        ]
    }

    Returns:
    {
        "success": true,
        "received": 2,
        "fixes": [...],
        "retries": [...],  # Nodes that need passthrough retry in Mason
        "fallbacks": [...]  # Nodes that need semantic replacement
    }
    """
    try:
        data = request.json
        session_id = data.get('session_id', 'unknown')
        errors = data.get('errors', [])

        if not errors:
            return jsonify({'success': True, 'received': 0, 'fixes': [], 'retries': [], 'fallbacks': []})

        print(f"[RuntimeErrors] Received {len(errors)} errors from session {session_id}")

        # Enrich errors with node metadata from session if available
        if session_id in sessions:
            session = sessions[session_id]
            # Try to enrich from various node sources if available
            # (This depends on how nodes are stored per session)
            # For now, we assume the browser sends the necessary fields
        
        # Store errors for debugging (in temp folder)
        temp_dir = Path(__file__).parent / 'temp'
        temp_dir.mkdir(exist_ok=True)
        error_file = temp_dir / f"errors_{session_id}.json"
        with open(error_file, 'w') as f:
            json.dump({
                'session_id': session_id,
                'timestamp': datetime.now().isoformat(),
                'errors': errors
            }, f, indent=2)

        # Attempt LLM-based fixes with new action-based handling
        inspector = get_runtime_inspector()
        result = inspector.batch_fix(errors)

        fixes = result.get('fixes', [])
        retries = result.get('retries', [])
        fallbacks = result.get('fallbacks', [])
        
        total_actions = len(fixes) + len(retries) + len(fallbacks)
        print(f"[RuntimeErrors] Generated {len(fixes)} fixes, {len(retries)} passthrough retries, {len(fallbacks)} fallbacks")

        # Store detailed results for debugging
        if total_actions > 0:
            fixes_file = temp_dir / f"fixes_{session_id}.json"
            with open(fixes_file, 'w') as f:
                json.dump({
                    'session_id': session_id,
                    'timestamp': datetime.now().isoformat(),
                    'fixes': fixes,
                    'retries': retries,
                    'fallbacks': fallbacks
                }, f, indent=2)

        # If there are passthrough retries, signal Mason to regenerate those nodes
        if retries:
            print(f"[RuntimeErrors] {len(retries)} nodes need passthrough retry from Mason")
            # The frontend should send these back to the pipeline for regeneration
        
        # If there are fallbacks, the frontend should request semantic node replacement
        if fallbacks:
            print(f"[RuntimeErrors] {len(fallbacks)} nodes need fallback semantic replacement")

        return jsonify({
            'success': True,
            'received': len(errors),
            'fixes': fixes,
            'retries': retries,
            'fallbacks': fallbacks
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/fix-node', methods=['POST'])
def fix_single_node():
    """
    Request LLM fix for a specific node error.

    Expected payload:
    {
        "session_id": "...",
        "error": {
            "node_id": "...",
            "category": "...",
            "engine": "...",
            "error_message": "...",
            "code_snippet": "...",
            "parameters": {...},
            "input_nodes": [...]
        }
    }

    Returns:
    {
        "success": true,
        "fixed_code": "...",
        "fixed_parameters": {...},
        "analysis": "..."
    }
    """
    try:
        data = request.json
        session_id = data.get('session_id', 'unknown')
        error_info = data.get('error', {})

        if not error_info:
            return jsonify({'success': False, 'error': 'No error info provided'})

        node_id = error_info.get('node_id', 'unknown')
        print(f"[FixNode] Fixing node {node_id} from session {session_id}")

        inspector = get_runtime_inspector()
        result = inspector.analyze_and_fix(error_info)

        if result['success']:
            print(f"[FixNode] Successfully fixed {node_id}: {result['analysis']}")
            return jsonify({
                'success': True,
                'fixed_code': result['fixed_code'],
                'fixed_parameters': result['fixed_parameters'],
                'analysis': result['analysis']
            })
        else:
            print(f"[FixNode] Could not fix {node_id}")
            return jsonify({
                'success': False,
                'error': 'Could not determine fix',
                'analysis': result.get('analysis', '')
            })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/inspector/history', methods=['GET'])
def get_inspector_history():
    """Get the history of fixes applied by the runtime inspector."""
    try:
        inspector = get_runtime_inspector()
        return jsonify({
            'success': True,
            'history': inspector.get_fix_history()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                    LowKCDR - Visual Node Editor                           ║
╠═══════════════════════════════════════════════════════════════════════════╣
║  Server:  http://localhost:5000                                           ║
║  Backend: Python/Flask (Phase 1 + Phase 2)                                ║
║  Frontend: JavaScript (Neurospiral-style + Node Editor)                   ║
║  Nodes: JS-based (replaces Taichi kernels)                               ║
╚═══════════════════════════════════════════════════════════════════════════╝
    """)

    # Pre-warm critical components in background
    import threading
    def warmup():
        try:
            get_visual_clusterer()
            get_image_searcher()
        except:
            pass
    threading.Thread(target=warmup, daemon=True).start()

    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
