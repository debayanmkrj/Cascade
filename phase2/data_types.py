"""Phase 2 Data Types - Polyglot Triple Filter Architecture"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Literal
from enum import Enum


# Runtime engine types for polyglot nodes
RuntimeEngine = Literal[
    "three_js", "webgpu", "glsl", "pixi_js", "regl",
    "onnx", "tensorflow_js", "webaudio", "midi",
    "webcodecs", "html_video", "events", "python"
]

Modality = Literal["field", "particles", "mesh", "audio", "event", "texture", "data"]
Domain = Literal["visual", "physics", "audio", "logic", "control"]
NodeRole = Literal["input", "process", "output", "control", "utility"]
CreativeLevel = Literal["surface", "flow", "narrative"]


@dataclass
class NodeMeta:
    """Metadata for a NodeTensor"""
    concept_id: str
    label: str
    level: CreativeLevel
    modality: Modality
    domain: Domain
    description: str = ""
    category: str = "process"
    role: NodeRole = "process"


@dataclass
class TextureHandle:
    """GPU texture reference for Z-layer compositing"""
    node_id: str
    width: int = 512
    height: int = 512
    format: str = "rgba8"
    z_layer: int = 0


@dataclass
class NodeTensor:
    """Polyglot node in the volumetric grid.
    Python generates the spec, JS hydrates and executes."""
    id: str
    meta: NodeMeta
    grid_position: Tuple[int, int, int]  # (x, y, z)
    grid_size: Tuple[int, int]           # (w, h) in grid cells
    engine: RuntimeEngine
    code_snippet: str                     # JS/GLSL/WGSL code
    parameters: Dict[str, Any] = field(default_factory=dict)
    input_nodes: List[str] = field(default_factory=list)
    output_texture: Optional[TextureHandle] = None
    # Semantic keywords from SemanticReasoner (for Mason tag matching)
    keywords: List[str] = field(default_factory=list)
    # Semantic purpose: what this specific node should contribute to the brief
    # Set by SemanticDecomposer — scoped validation target for Mason
    semantic_purpose: str = ""
    # Validation state
    architect_approved: bool = False
    mason_approved: bool = False
    validation_errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        # Handle meta as either NodeMeta object or plain dict
        if isinstance(self.meta, dict):
            meta_dict = self.meta
        else:
            meta_dict = {
                'concept_id': self.meta.concept_id,
                'label': self.meta.label,
                'level': self.meta.level,
                'modality': self.meta.modality,
                'domain': self.meta.domain,
                'description': self.meta.description,
                'category': self.meta.category,
                'role': self.meta.role
            }

        return {
            'id': self.id,
            'name': self.id,  # For backwards compatibility, use id as name
            'category': meta_dict.get('category', 'process'),
            'role': meta_dict.get('role', 'process'),
            'meta': meta_dict,
            'grid_position': list(self.grid_position),
            'grid_size': list(self.grid_size),
            'engine': self.engine,
            'code_snippet': self.code_snippet,
            'parameters': self.parameters,
            'input_nodes': self.input_nodes,
            'output_texture': (
                self.output_texture if isinstance(self.output_texture, dict) else {
                    'node_id': self.output_texture.node_id,
                    'width': self.output_texture.width,
                    'height': self.output_texture.height,
                    'format': self.output_texture.format,
                    'z_layer': self.output_texture.z_layer
                }
            ) if self.output_texture else None,
            'keywords': self.keywords,
            'semantic_purpose': self.semantic_purpose,
            'architect_approved': self.architect_approved,
            'mason_approved': self.mason_approved,
            'validation_errors': self.validation_errors
        }


@dataclass
class Connection:
    """Edge between two nodes"""
    from_node: str
    from_output: int
    to_node: str
    to_input: int


@dataclass
class VolumetricGrid:
    """The volumetric grid holding all NodeTensors.
    Z-axis = compositing layers, X/Y = spatial layout."""
    dimensions: Tuple[int, int, int]  # (X, Y, Z)
    nodes: List[NodeTensor] = field(default_factory=list)
    connections: List[Connection] = field(default_factory=list)
    runtime_hints: Dict[str, Any] = field(default_factory=dict)

    def to_project_json(self) -> Dict:
        """Export as project JSON for JS runtime"""
        return {
            'grid': list(self.dimensions),
            'nodes': [n.to_dict() for n in self.nodes],
            'connections': [
                {
                    'from_node': c.from_node,
                    'from_output': c.from_output,
                    'to_node': c.to_node,
                    'to_input': c.to_input
                } for c in self.connections
            ],
            'runtime_hints': self.runtime_hints
        }

    def get_nodes_at_z(self, z: int) -> List[NodeTensor]:
        """Get all nodes at a given Z layer"""
        return [n for n in self.nodes if n.grid_position[2] == z]

    def get_z_layers(self) -> int:
        """Get number of Z layers in use"""
        if not self.nodes:
            return 0
        return max(n.grid_position[2] for n in self.nodes) + 1


@dataclass
class ArchitectPlan:
    """Output of the Architect agent - graph topology + grid placement"""
    grid: VolumetricGrid
    reasoning: str = ""
    topology_valid: bool = False
    validation_log: List[str] = field(default_factory=list)


@dataclass
class MasonResult:
    """Output of the Mason agent - code snippets for each node"""
    node_id: str
    engine: RuntimeEngine
    code_snippet: str
    syntax_valid: bool = False
    validation_log: List[str] = field(default_factory=list)


# ============================================================
# Chain of Influence types
# ============================================================

ChannelProtocol = Literal[
    "DENSITY_RGBA",   # R=density, G=age/trail, B=noise, A=mask
    "COLOR_RGBA",     # RGB=color, A=alpha
    "IMAGE_RGBA",     # Camera/image feed
    "AUDIO_FFT",      # Frequency bin data
    "LANDMARKS",      # Tracking keypoints
    "DATA_JSON",      # Structured events/control
]

InfluenceType = Literal[
    "mask_and_emit",
    "warp_only",
    "color_grade",
    "composite",
    "feedback_trails",
    "data_drive",
]


@dataclass
class InfluenceContract:
    """What one node must do with another node's output."""
    influence_type: str  # InfluenceType
    must_use: List[str] = field(default_factory=list)
    preserve: List[str] = field(default_factory=list)
    allow: List[str] = field(default_factory=list)
    avoid: List[str] = field(default_factory=list)


@dataclass
class InfluenceEdge:
    """Typed edge in the Influence Graph IR."""
    from_node: str
    to_node: str
    protocol: str  # ChannelProtocol
    influence: InfluenceContract

    def to_legacy_connection(self, input_index: int = 0) -> Connection:
        return Connection(from_node=self.from_node, from_output=0,
                          to_node=self.to_node, to_input=input_index)


@dataclass
class InfluenceNode:
    """Node in the Influence Graph IR."""
    id: str
    engine: str  # RuntimeEngine
    role: str  # NodeRole
    intent: str
    output_protocol: str  # ChannelProtocol
    category: str = ""
    keywords: List[str] = field(default_factory=list)
    suggested_params: List[str] = field(default_factory=list)
    style_anchor: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InfluenceGraphIR:
    """Complete Influence Graph intermediate representation."""
    global_context: Dict[str, Any]
    nodes: List[InfluenceNode] = field(default_factory=list)
    edges: List[InfluenceEdge] = field(default_factory=list)
    reasoning: str = ""


@dataclass
class BuildSheet:
    """Per-node instruction sheet for Mason — the local contract."""
    node_id: str
    engine: str
    intent: str
    inputs: List[Dict[str, Any]] = field(default_factory=list)
    influence_rules: Dict[str, List[str]] = field(default_factory=dict)
    output_protocol: str = "COLOR_RGBA"
    style_anchor: Dict[str, Any] = field(default_factory=dict)
    params: List[str] = field(default_factory=list)
    # Z-axis semantic context
    grid_position: Tuple[int, int, int] = (0, 0, 0)
    z_total: int = 1
    z_role: str = ""  # "source", "processor", "compositor", "output"
