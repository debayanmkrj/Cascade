"""Brand and Emotion Extraction (Phase 1 - Section 7.1.1)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from typing import List, Dict
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
from io import BytesIO
import os

from config import CLIP_MODEL, EMOTION_LABELS, BRAND_ATTRIBUTE_LABELS, OLLAMA_URL, MODEL_NAME
from data_types import BrandValues, ImageRef


class BrandExtractor:
    """Extract brand values and emotions from prompt + images using CLIP"""

    def __init__(self):
        print("Loading CLIP for brand extraction...")
        self.model = CLIPModel.from_pretrained(CLIP_MODEL)
        self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def extract_brand_values(self, prompt_text: str, images: List[ImageRef]) -> BrandValues:
        """
        Extract brand values from prompt text and images.

        Algorithm (from Phase 1 spec):
        1. Extract colors from prompt text (hex codes, color names)
        2. CLIP zero-shot classify images for emotions
        3. Aggregate emotion scores across images
        4. Extract brand attributes from prompt using LLM
        5. Combine image emotions + text attributes
        6. Extract color palette from images
        7. Merge text colors + image colors

        Args:
            prompt_text: User's text prompt
            images: List of reference/custom images

        Returns:
            BrandValues with emotions, attributes, mood, palette
        """
        # Extract colors from prompt text FIRST
        text_colors = self._extract_colors_from_text(prompt_text)

        # Extract emotions from images
        image_emotions = {}
        color_palette = text_colors.copy()  # Start with text colors
        valid_image_count = 0

        if images:
            for img_ref in images:
                try:
                    pil_img = self._load_image(img_ref.url)
                    if pil_img is None:
                        continue

                    # Classify emotions
                    emotions = self._classify_emotions(pil_img)
                    for emotion, score in emotions.items():
                        image_emotions[emotion] = image_emotions.get(emotion, 0.0) + score

                    # Extract colors
                    colors = self._extract_colors(pil_img)
                    color_palette.extend(colors)

                    valid_image_count += 1
                except Exception as e:
                    print(f"Error processing image {img_ref.id}: {e}")

            # Average emotions across images
            if valid_image_count > 0:
                image_emotions = {k: v / valid_image_count for k, v in image_emotions.items()}

        # Extract brand attributes from prompt text
        text_attributes = self._extract_text_attributes(prompt_text)

        # Determine dominant mood
        visual_mood = "neutral"
        if image_emotions:
            visual_mood = max(image_emotions.items(), key=lambda x: x[1])[0]

        # Deduplicate and limit color palette
        unique_colors = list(dict.fromkeys(color_palette))[:8]
        if not unique_colors:
            unique_colors = ["#1e293b", "#64748b", "#e2e8f0"]  # Default neutral palette

        # Confidence based on data availability
        confidence = 0.5
        if valid_image_count > 0:
            confidence += 0.3
        if prompt_text.strip():
            confidence += 0.2

        return BrandValues(
            emotions=image_emotions or {e: 0.5 for e in EMOTION_LABELS[:4]},
            brand_attributes=text_attributes,
            visual_mood=visual_mood,
            color_palette=unique_colors,
            confidence=min(1.0, confidence)
        )

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
                response.raise_for_status()  # Raise on HTTP errors
                return Image.open(BytesIO(response.content)).convert('RGB')
        except requests.exceptions.RequestException as e:
            print(f"Network error loading image from {url}: {e}")
            return None
        except Exception as e:
            print(f"Error loading image from {url}: {e}")
            return None

    def _classify_emotions(self, image: Image.Image) -> Dict[str, float]:
        """Classify image emotions using CLIP zero-shot"""
        try:
            text_inputs = self.processor(text=EMOTION_LABELS, return_tensors="pt", padding=True).to(self.device)
            image_inputs = self.processor(images=image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                image_features = self.model.get_image_features(**image_inputs)
                text_features = self.model.get_text_features(**text_inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                similarities = (image_features @ text_features.T).squeeze().cpu().numpy()

            # Softmax to get probabilities
            exp_sims = np.exp(similarities - np.max(similarities))
            probs = exp_sims / exp_sims.sum()

            return {label: float(prob) for label, prob in zip(EMOTION_LABELS, probs)}
        except Exception as e:
            print(f"CLIP emotion classification error: {e}")
            # Return uniform distribution as fallback
            return {label: 1.0 / len(EMOTION_LABELS) for label in EMOTION_LABELS}

    def _extract_text_attributes(self, prompt_text: str) -> Dict[str, float]:
        """Extract brand attributes from text using LLM"""
        if not prompt_text.strip():
            return {attr: 0.5 for attr in BRAND_ATTRIBUTE_LABELS[:4]}

        extraction_prompt = f"""Analyze this design prompt and rate how strongly each brand attribute applies (0.0-1.0).
Return ONLY valid JSON with no explanation.

PROMPT: "{prompt_text}"

BRAND ATTRIBUTES: {', '.join(BRAND_ATTRIBUTE_LABELS)}

Output format:
{{"trustworthy": 0.8, "innovative": 0.6, ...}}

JSON OUTPUT:"""

        try:
            import requests
            response = requests.post(
                OLLAMA_URL,
                json={'model': MODEL_NAME, 'prompt': extraction_prompt, 'stream': False, 'format': 'json'},
                timeout=60
            )
            response.raise_for_status()  # Raise on HTTP errors
            result = response.json().get('response', '{}')

            import re
            import json
            json_match = re.search(r'\{[^}]+\}', result, re.DOTALL)
            if json_match:
                values = json.loads(json_match.group())
                return {k: max(0.0, min(1.0, float(v))) for k, v in values.items() if k in BRAND_ATTRIBUTE_LABELS}
        except requests.exceptions.Timeout:
            print("LLM request timed out (60s)")
        except requests.exceptions.RequestException as e:
            print(f"Network error in text attribute extraction: {e}")
        except Exception as e:
            print(f"Text attribute extraction error: {e}")

        return {attr: 0.5 for attr in BRAND_ATTRIBUTE_LABELS[:4]}

    def _extract_colors(self, image: Image.Image, n_colors: int = 3) -> List[str]:
        """Extract dominant colors from image using k-means"""
        try:
            from sklearn.cluster import KMeans

            img = image.resize((50, 50))
            pixels = np.array(img).reshape(-1, 3)

            kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
            kmeans.fit(pixels)

            colors = []
            for center in kmeans.cluster_centers_:
                hex_color = '#{:02x}{:02x}{:02x}'.format(int(center[0]), int(center[1]), int(center[2]))
                colors.append(hex_color)
            return colors
        except Exception as e:
            print(f"Color extraction error: {e}")
            return ["#64748b", "#94a3b8", "#cbd5e1"]  # Default neutral colors

    def _extract_colors_from_text(self, prompt_text: str) -> List[str]:
        """
        Extract color hex codes and color names from prompt text.

        Handles:
        - Hex codes: #00FFFF, #FF00CC
        - Color names: cyan, magenta, neon cyan, etc.
        """
        import re

        colors = []

        # Extract hex codes (e.g., #00FFFF, #FF00CC)
        hex_pattern = r'#[0-9A-Fa-f]{6}'
        hex_matches = re.findall(hex_pattern, prompt_text)
        colors.extend(hex_matches)

        # Color name to hex mapping
        color_map = {
            'cyan': '#00FFFF',
            'magenta': '#FF00FF',
            'yellow': '#FFFF00',
            'red': '#FF0000',
            'green': '#00FF00',
            'blue': '#0000FF',
            'orange': '#FF8800',
            'purple': '#8800FF',
            'pink': '#FF0088',
            'lime': '#88FF00',
            'teal': '#00FF88',
            'violet': '#8800FF',
            'indigo': '#4B0082',
            'turquoise': '#40E0D0',
            'coral': '#FF7F50',
            'salmon': '#FA8072',
            'gold': '#FFD700',
            'silver': '#C0C0C0',
            'black': '#000000',
            'white': '#FFFFFF',
            'gray': '#808080',
            'grey': '#808080',
            'dark': '#1a1a1a',
            'neon cyan': '#00FFFF',
            'neon magenta': '#FF00FF',
            'neon pink': '#FF10F0',
            'neon green': '#39FF14',
            'neon blue': '#0080FF',
            'neon yellow': '#FFFF00',
            'electric blue': '#0080FF',
            'hot pink': '#FF69B4',
            'lime green': '#32CD32',
            'sky blue': '#87CEEB',
            'forest green': '#228B22',
            'royal blue': '#4169E1',
            'crimson': '#DC143C',
            'navy': '#000080',
            'olive': '#808000',
            'maroon': '#800000',
            'aqua': '#00FFFF',
            'fuchsia': '#FF00FF'
        }

        # Search for color names in prompt (case insensitive)
        prompt_lower = prompt_text.lower()
        for color_name, hex_code in color_map.items():
            if color_name in prompt_lower:
                colors.append(hex_code)

        # Remove duplicates while preserving order
        seen = set()
        unique_colors = []
        for color in colors:
            if color.upper() not in seen:
                seen.add(color.upper())
                unique_colors.append(color.upper())

        return unique_colors
