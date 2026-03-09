"""Visual Palette Generation - CLIP-based Color and Style Extraction"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from typing import List, Tuple
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import colorsys
import os
import requests
from io import BytesIO

from config import CLIP_MODEL
from data_types import ImageRef, VisualPalette, BrandValues


def _ensure_tensor(features):
    """Extract tensor from CLIP output (handles both raw tensors and BaseModelOutputWithPooling)."""
    if hasattr(features, 'pooler_output'):
        return features.pooler_output
    if hasattr(features, 'last_hidden_state'):
        return features.last_hidden_state[:, 0]
    return features


class VisualPaletteGenerator:
    """Generate visual palette using CLIP embeddings for colors, shapes, and motion"""

    def __init__(self, clip_model=None, clip_processor=None):
        if clip_model and clip_processor:
            self.model = clip_model
            self.processor = clip_processor
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
        else:
            print("Loading CLIP for visual palette generation...")
            try:
                self.model = CLIPModel.from_pretrained(CLIP_MODEL)
                self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
            except Exception as e:
                print(f"  Online CLIP load failed ({e}), trying local cache...")
                self.model = CLIPModel.from_pretrained(CLIP_MODEL, local_files_only=True)
                self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL, local_files_only=True)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)

        # CLIP color labels for zero-shot classification
        self.color_labels = [
            "deep red color", "bright red color", "orange color", "golden yellow color",
            "lime green color", "forest green color", "cyan color", "sky blue color",
            "deep blue color", "purple color", "magenta color", "pink color",
            "white color", "black color", "gray color", "brown color",
            "neon cyan color", "neon magenta color", "neon green color"
        ]

        self.color_hex_map = {
            "deep red color": "#8B0000", "bright red color": "#FF0000",
            "orange color": "#FF8800", "golden yellow color": "#FFD700",
            "lime green color": "#32CD32", "forest green color": "#228B22",
            "cyan color": "#00FFFF", "sky blue color": "#87CEEB",
            "deep blue color": "#00008B", "purple color": "#800080",
            "magenta color": "#FF00FF", "pink color": "#FF69B4",
            "white color": "#FFFFFF", "black color": "#000000",
            "gray color": "#808080", "brown color": "#8B4513",
            "neon cyan color": "#00FFFF", "neon magenta color": "#FF00FF",
            "neon green color": "#39FF14"
        }

        # Mood to color mapping
        self.mood_color_map = {
            "excitement": ["#FF0000", "#FF8800", "#FFD700"],
            "calmness": ["#87CEEB", "#00CED1", "#E0FFFF"],
            "joy": ["#FFD700", "#FF69B4", "#FFA500"],
            "melancholy": ["#4169E1", "#483D8B", "#708090"],
            "energy": ["#FF0000", "#39FF14", "#FF00FF"],
            "serenity": ["#E0FFFF", "#98FB98", "#AFEEEE"],
            "playfulness": ["#FF69B4", "#00FFFF", "#FFD700"],
            "warmth": ["#FF8800", "#FFD700", "#FF6347"],
            "coldness": ["#00CED1", "#4682B4", "#E0FFFF"],
            "chaos": ["#FF00FF", "#00FFFF", "#FF0000"],
            "luxury": ["#FFD700", "#800080", "#000000"],
            "futurism": ["#00FFFF", "#FF00FF", "#0000FF"]
        }

    def generate_palette(self, images: List[ImageRef], brand_values: BrandValues) -> VisualPalette:
        """
        Generate visual palette from multiple sources:
        1. CLIP embeddings of selected images (primary source)
        2. Text color references from BrandValues
        3. Text mood to infer colors
        """
        all_colors = []

        # SOURCE 1: CLIP-based color extraction from images
        if images:
            clip_colors = self._extract_colors_via_clip(images)
            all_colors.extend(clip_colors)
            print(f"  VisualPalette - CLIP extracted {len(clip_colors)} colors from images")

        # SOURCE 2: Colors from text (already in BrandValues from brand_extraction)
        if brand_values.color_palette:
            all_colors.extend(brand_values.color_palette)
            print(f"  VisualPalette - {len(brand_values.color_palette)} colors from text")

        # SOURCE 3: Mood-based color inference
        if brand_values.visual_mood:
            mood_colors = self._colors_from_mood(brand_values.visual_mood)
            all_colors.extend(mood_colors)
            print(f"  VisualPalette - {len(mood_colors)} colors from mood '{brand_values.visual_mood}'")

        # Deduplicate and organize
        unique_colors = self._deduplicate_colors(all_colors)

        # Split into primary (first 3) and accent (next 3)
        primary_colors = unique_colors[:3] if len(unique_colors) >= 3 else unique_colors
        accent_colors = unique_colors[3:6] if len(unique_colors) > 3 else []

        # Ensure minimums
        if len(primary_colors) < 3:
            primary_colors.extend(["#1e293b", "#64748b", "#e2e8f0"][:3 - len(primary_colors)])
        if len(accent_colors) < 3:
            accent_colors.extend(["#f59e0b", "#06b6d4", "#a855f7"][:3 - len(accent_colors)])

        # Extract shapes and motion from images
        shapes = self._classify_shapes(images)
        motion_words = self._classify_motion(images)

        return VisualPalette(
            primary_colors=primary_colors,
            accent_colors=accent_colors,
            shapes=shapes,
            motion_words=motion_words
        )

    def _extract_colors_via_clip(self, images: List[ImageRef]) -> List[str]:
        """Extract dominant colors from images using CLIP zero-shot classification"""
        color_scores = {label: 0.0 for label in self.color_labels}
        valid_count = 0

        for img_ref in images:
            try:
                pil_img = self._load_image(img_ref.url)
                if pil_img is None:
                    continue

                text_inputs = self.processor(
                    text=self.color_labels,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)

                image_inputs = self.processor(
                    images=pil_img,
                    return_tensors="pt"
                ).to(self.device)

                with torch.no_grad():
                    image_features = _ensure_tensor(self.model.get_image_features(**image_inputs))
                    text_features = _ensure_tensor(self.model.get_text_features(**text_inputs))
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    similarities = (image_features @ text_features.T).squeeze().cpu().numpy()

                exp_sims = np.exp(similarities - np.max(similarities))
                probs = exp_sims / exp_sims.sum()

                for label, prob in zip(self.color_labels, probs):
                    color_scores[label] += prob

                valid_count += 1

            except Exception as e:
                print(f"  CLIP color extraction error for {img_ref.id}: {e}")

        if valid_count == 0:
            return []

        color_scores = {k: v / valid_count for k, v in color_scores.items()}
        sorted_colors = sorted(color_scores.items(), key=lambda x: x[1], reverse=True)

        result = []
        for label, score in sorted_colors[:5]:
            if score > 0.05:
                result.append(self.color_hex_map.get(label, "#808080"))

        return result

    def _colors_from_mood(self, mood: str) -> List[str]:
        """Infer colors from visual mood"""
        mood_lower = mood.lower()
        for mood_key, colors in self.mood_color_map.items():
            if mood_key in mood_lower:
                return colors
        return []

    def _deduplicate_colors(self, colors: List[str]) -> List[str]:
        """Deduplicate colors, merging similar ones"""
        seen = set()
        unique = []

        for color in colors:
            color_upper = color.upper()
            if color_upper not in seen:
                is_similar = False
                for existing in seen:
                    if self._colors_similar(color_upper, existing):
                        is_similar = True
                        break

                if not is_similar:
                    seen.add(color_upper)
                    unique.append(color_upper)

        return unique

    def _colors_similar(self, hex1: str, hex2: str, threshold: float = 0.15) -> bool:
        """Check if two colors are similar using HSV distance"""
        try:
            rgb1 = self._hex_to_rgb(hex1)
            rgb2 = self._hex_to_rgb(hex2)

            hsv1 = colorsys.rgb_to_hsv(rgb1[0]/255, rgb1[1]/255, rgb1[2]/255)
            hsv2 = colorsys.rgb_to_hsv(rgb2[0]/255, rgb2[1]/255, rgb2[2]/255)

            h_diff = min(abs(hsv1[0] - hsv2[0]), 1 - abs(hsv1[0] - hsv2[0]))
            s_diff = abs(hsv1[1] - hsv2[1])
            v_diff = abs(hsv1[2] - hsv2[2])

            distance = (h_diff * 2 + s_diff + v_diff) / 4
            return distance < threshold
        except:
            return False

    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """Convert hex to RGB tuple"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def _load_image(self, url: str) -> Image.Image:
        """Load image from URL or local file"""
        try:
            if url.startswith('/uploads/') or url.startswith('./uploads/'):
                filepath = '.' + url if url.startswith('/') else url
                if os.path.exists(filepath):
                    return Image.open(filepath).convert('RGB')
                return None
            else:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                return Image.open(BytesIO(response.content)).convert('RGB')
        except Exception as e:
            print(f"  Error loading image from {url}: {e}")
            return None

    def _classify_shapes(self, images: List[ImageRef]) -> List[str]:
        """Classify visual shapes using CLIP zero-shot"""
        shape_labels = [
            "circles", "grid", "tunnel", "particles", "waves",
            "fractals", "geometric shapes", "organic shapes", "lines",
            "dots", "polygons", "spheres", "cubes", "spirals"
        ]

        shape_scores = {label: 0.0 for label in shape_labels}
        valid_count = 0

        for img_ref in images:
            try:
                pil_img = self._load_image(img_ref.url)
                if pil_img is None:
                    continue

                text_inputs = self.processor(text=shape_labels, return_tensors="pt", padding=True).to(self.device)
                image_inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)

                with torch.no_grad():
                    image_features = _ensure_tensor(self.model.get_image_features(**image_inputs))
                    text_features = _ensure_tensor(self.model.get_text_features(**text_inputs))
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    similarities = (image_features @ text_features.T).squeeze().cpu().numpy()

                exp_sims = np.exp(similarities - np.max(similarities))
                probs = exp_sims / exp_sims.sum()

                for label, prob in zip(shape_labels, probs):
                    shape_scores[label] += prob

                valid_count += 1
            except Exception as e:
                print(f"  Shape classification error for {img_ref.id}: {e}")

        if valid_count > 0:
            shape_scores = {k: v / valid_count for k, v in shape_scores.items()}

        top_shapes = sorted(shape_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        return [shape for shape, score in top_shapes]

    def _classify_motion(self, images: List[ImageRef]) -> List[str]:
        """Classify motion qualities using CLIP zero-shot"""
        motion_labels = [
            "pulsing motion", "glitchy movement", "smooth flow", "jittery motion",
            "flowing movement", "chaotic motion", "rhythmic movement", "organic movement",
            "static", "dynamic", "fast motion", "slow motion", "vibrating"
        ]

        motion_scores = {label: 0.0 for label in motion_labels}
        valid_count = 0

        for img_ref in images:
            try:
                pil_img = self._load_image(img_ref.url)
                if pil_img is None:
                    continue

                text_inputs = self.processor(text=motion_labels, return_tensors="pt", padding=True).to(self.device)
                image_inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)

                with torch.no_grad():
                    image_features = _ensure_tensor(self.model.get_image_features(**image_inputs))
                    text_features = _ensure_tensor(self.model.get_text_features(**text_inputs))
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    similarities = (image_features @ text_features.T).squeeze().cpu().numpy()

                exp_sims = np.exp(similarities - np.max(similarities))
                probs = exp_sims / exp_sims.sum()

                for label, prob in zip(motion_labels, probs):
                    motion_scores[label] += prob

                valid_count += 1
            except Exception as e:
                print(f"  Motion classification error for {img_ref.id}: {e}")

        if valid_count > 0:
            motion_scores = {k: v / valid_count for k, v in motion_scores.items()}

        top_motions = sorted(motion_scores.items(), key=lambda x: x[1], reverse=True)[:3]

        clean_motions = []
        for motion, score in top_motions:
            clean = motion.replace(" motion", "").replace(" movement", "").replace(" flow", "")
            clean_motions.append(clean)

        return clean_motions
