"""Data Types for Low-k-cdr Phase 1"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal
from datetime import datetime
import hashlib
import uuid


def stable_id(prefix: str, *components) -> str:
    """Generate stable, deterministic ID using SHA256"""
    content = "_".join(str(c) for c in components)
    hash_hex = hashlib.sha256(content.encode('utf-8')).hexdigest()[:12]
    return f"{prefix}_{hash_hex}"


# Input Data Types
@dataclass
class ImageRef:
    """Reference or uploaded image"""
    id: str
    url: str
    thumbnail_url: str
    source: Literal['reference', 'custom', 'upload']
    metadata: Dict = field(default_factory=dict)
    cluster_id: Optional[int] = None  # CLIP cluster assignment
    colors: List[str] = field(default_factory=list)  # Dominant colors
    mood: Optional[str] = None  # CLIP mood classification


@dataclass
class RawInputBundle:
    """User input bundle with all raw data"""
    prompt_text: str
    reference_images: List[ImageRef]
    custom_images: List[ImageRef]
    D_global: float  # Divergence dial [0, 1]
    user_mode: Literal['functional', 'aesthetic', 'flow']  # Updated from spec
    platform_preference: Optional[Literal['touchdesigner', 'max_msp', 'generic']] = 'generic'
    seed: int = 42  # Random seed for reproducibility
    timestamp: datetime = field(default_factory=datetime.now)


# Brand/Emotion Extraction
@dataclass
class BrandValues:
    """Extracted brand values and emotions from prompt + images"""
    emotions: Dict[str, float]  # emotion_name -> strength [0, 1]
    brand_attributes: Dict[str, float]  # attribute -> strength [0, 1]
    visual_mood: str  # dominant mood from images
    color_palette: List[str]  # hex colors
    confidence: float


# Creative Levels
@dataclass
class CreativeLevelSpec:
    """Weight for each creative level"""
    surface: float  # [0, 1]
    flow: float     # [0, 1]
    narrative: float  # [0, 1]

    def get_dominant_level(self) -> str:
        """Get the level with highest weight"""
        levels = {'surface': self.surface, 'flow': self.flow, 'narrative': self.narrative}
        return max(levels, key=levels.get)


@dataclass
class DivergenceValues:
    """Divergence values for each level"""
    D_surface: float
    D_flow: float
    D_narrative: float


# UI Generation
@dataclass
class UIControl:
    """Single UI control with properties"""
    id: str  # Stable ID
    type: str  # slider, toggle, dropdown, xy_pad, color, etc.
    label: str
    parameters: Dict  # type-specific normalized parameters
    grounding_score: float
    grounding_source: str  # which RAG chunk
    confidence: float
    targets: List[str] = field(default_factory=list)  # which level (surface/flow/narrative)
    bindings: List[Dict] = field(default_factory=list)  # [{node_id: str, param: str}]
    creative_level: str = "surface"  # Which level this belongs to


@dataclass
class UILayoutNode:
    """UI layout with grouped controls"""
    id: str
    type: Literal['group', 'control']
    label: Optional[str] = None
    controls: List[UIControl] = field(default_factory=list)
    children: List['UILayoutNode'] = field(default_factory=list)
    layout_hint: Optional[str] = None  # 'horizontal', 'vertical', 'grid'


# Node Archetypes
@dataclass
class NodeArchetype:
    """TouchDesigner/Max MSP node archetype"""
    id: str  # Stable ID
    name: str
    category: str  # Validated against NODE_CATEGORIES
    role: str  # 'input', 'process', 'output', 'control', 'utility'
    creative_level: str  # 'surface', 'flow', 'narrative'
    description: str
    parameters: List[Dict]  # list of parameter specs
    engine_hints: Dict[str, str] = field(default_factory=dict)  # platform-specific hints
    grounding_score: float = 0.5
    grounding_source: str = ""
    confidence: float = 0.5


# Visual Palette (from spec)
@dataclass
class VisualPalette:
    """Visual design palette extracted from images"""
    primary_colors: List[str]  # hex colors
    accent_colors: List[str]
    shapes: List[str]  # "circles", "grid", "tunnel", "particles"
    motion_words: List[str]  # "pulsing", "glitchy", "smooth", "jittery"


# Final Output
@dataclass
class NodeBrief:
    """Complete node-based workflow blueprint"""
    essence: str  # Core concept statement
    brand_values: BrandValues
    creative_levels: CreativeLevelSpec
    divergence: DivergenceValues

    visual_palette: VisualPalette
    ui_layout: UILayoutNode
    ui_controls: List[UIControl]

    node_archetypes: List[NodeArchetype]
    node_count: int
    node_workflow_description: str

    control_bindings: List[Dict] = field(default_factory=list)  # UI -> Node bindings
    grounding_report: Dict = field(default_factory=dict)  # sources and confidence scores
    timestamp: datetime = field(default_factory=datetime.now)
    seed: int = 42  # Random seed used
    platform: str = "generic"  # Target platform

    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict"""
        return {
            'essence': self.essence,
            'brand_values': {
                'emotions': self.brand_values.emotions,
                'brand_attributes': self.brand_values.brand_attributes,
                'visual_mood': self.brand_values.visual_mood,
                'color_palette': self.brand_values.color_palette,
                'confidence': self.brand_values.confidence
            },
            'creative_levels': {
                'surface': self.creative_levels.surface,
                'flow': self.creative_levels.flow,
                'narrative': self.creative_levels.narrative
            },
            'divergence': {
                'D_surface': self.divergence.D_surface,
                'D_flow': self.divergence.D_flow,
                'D_narrative': self.divergence.D_narrative
            },
            'visual_palette': {
                'primary_colors': self.visual_palette.primary_colors,
                'accent_colors': self.visual_palette.accent_colors,
                'shapes': self.visual_palette.shapes,
                'motion_words': self.visual_palette.motion_words
            },
            'ui_layout': self._layout_to_dict(self.ui_layout),
            'ui_controls': [
                {
                    'id': c.id,
                    'type': c.type,
                    'label': c.label,
                    'parameters': c.parameters,
                    'grounding_score': c.grounding_score,
                    'grounding_source': c.grounding_source,
                    'confidence': c.confidence,
                    'targets': c.targets,
                    'bindings': c.bindings,
                    'creative_level': c.creative_level
                } for c in self.ui_controls
            ],
            'node_archetypes': [
                n.to_dict() if hasattr(n, 'to_dict') else {
                    'id': n.id,
                    'meta': n.meta if isinstance(n.meta, dict) else n.meta.__dict__,
                    'parameters': n.parameters
                } for n in self.node_archetypes
            ],
            'node_count': self.node_count,
            'node_workflow_description': self.node_workflow_description,
            'control_bindings': self.control_bindings,
            'grounding_report': self.grounding_report,
            'timestamp': self.timestamp.isoformat(),
            'seed': self.seed,
            'platform': self.platform
        }

    def _layout_to_dict(self, layout: UILayoutNode) -> Dict:
        """Recursively convert UI layout to dict"""
        if layout is None:
            return None
        result = {
            'id': layout.id,
            'type': layout.type,
            'label': layout.label,
            'layout_hint': layout.layout_hint
        }
        if layout.controls:
            result['controls'] = [
                {
                    'id': c.id,
                    'type': c.type,
                    'label': c.label,
                    'parameters': c.parameters
                } for c in layout.controls
            ]
        if layout.children:
            result['children'] = [self._layout_to_dict(child) for child in layout.children]
        return result


# Design Assistant (from spec section 8)
@dataclass
class EditRequest:
    """User edit request to modify NodeBrief"""
    brief_id: str
    edit_type: Literal['explain', 'edit', 'reiterate']
    target: str  # what to edit (e.g., 'ui_controls', 'node_archetypes')
    instruction: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ExplanationResponse:
    """Explanation of a NodeBrief component"""
    component: str
    explanation: str
    grounding_sources: List[str]
    confidence: float


# Session State (from spec section 8.1)
@dataclass
class SessionState:
    """Complete session state for DesignAssistant"""
    session_id: str
    seed: int
    input_bundle: RawInputBundle
    node_brief: Optional[NodeBrief]
    history: List[Dict] = field(default_factory=list)
    phase1_meta: Dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        """Convert session to JSON"""
        return {
            'session_id': self.session_id,
            'seed': self.seed,
            'input_bundle': {
                'prompt_text': self.input_bundle.prompt_text,
                'reference_images': [
                    {
                        'id': img.id,
                        'url': img.url,
                        'source': img.source,
                        'cluster_id': img.cluster_id,
                        'mood': img.mood
                    } for img in self.input_bundle.reference_images
                ],
                'custom_images': [
                    {
                        'id': img.id,
                        'url': img.url,
                        'source': img.source,
                        'cluster_id': img.cluster_id,
                        'mood': img.mood
                    } for img in self.input_bundle.custom_images
                ],
                'D_global': self.input_bundle.D_global,
                'user_mode': self.input_bundle.user_mode,
                'platform_preference': self.input_bundle.platform_preference,
                'seed': self.input_bundle.seed
            },
            'node_brief': self.node_brief.to_dict() if self.node_brief else None,
            'history': self.history,
            'phase1_meta': self.phase1_meta,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }
