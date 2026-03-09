"""Visual Clustering - CLIP-based image clustering (from neurospiral pattern)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from typing import List, Dict
from PIL import Image
import requests
from io import BytesIO
import torch
from transformers import CLIPProcessor, CLIPModel
from sklearn.cluster import KMeans
from collections import Counter
import os

from config import CLIP_MODEL, VISUAL_MOOD_LABELS
from data_types import ImageRef


def _ensure_tensor(features):
    """Extract tensor from CLIP output (handles both raw tensors and BaseModelOutputWithPooling)."""
    if hasattr(features, 'pooler_output'):
        return features.pooler_output
    if hasattr(features, 'last_hidden_state'):
        return features.last_hidden_state[:, 0]
    return features


class VisualClusterer:
    """Cluster images using CLIP and extract semantic properties"""

    def __init__(self):
        print("Loading CLIP for clustering...")
        try:
            self.model = CLIPModel.from_pretrained(CLIP_MODEL)
            self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
        except Exception as e:
            print(f"  Online CLIP load failed ({e}), trying local cache...")
            self.model = CLIPModel.from_pretrained(CLIP_MODEL, local_files_only=True)
            self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL, local_files_only=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.mood_labels = VISUAL_MOOD_LABELS

    def cluster_images(self, images: List[ImageRef], n_clusters: int = 6) -> Dict:
        """
        Cluster images semantically and extract visual properties.

        Args:
            images: List of ImageRef objects
            n_clusters: Number of clusters

        Returns:
            Dict with clusters, palette, mood, and updated images
        """
        if not images:
            return {'clusters': [], 'palette': [], 'mood': 'neutral', 'images': []}

        embeddings = []
        valid_images = []
        image_data = []

        # Get embeddings for each image
        for img_ref in images:
            try:
                pil_img = self._load_image(img_ref.url)
                if pil_img is None:
                    continue

                emb = self._get_embedding(pil_img)
                embeddings.append(emb)
                valid_images.append(img_ref)

                # Extract colors and mood
                colors = self._extract_colors(pil_img)
                mood = self._classify_mood(pil_img)

                # Update ImageRef with clustering info
                img_ref.colors = colors
                img_ref.mood = mood

                image_data.append({
                    'id': img_ref.id,
                    'url': img_ref.url,
                    'thumbnail_url': img_ref.thumbnail_url,
                    'colors': colors,
                    'mood': mood,
                    'cluster_id': None  # Will be set after clustering
                })
            except Exception as e:
                print(f"Error processing image {img_ref.id}: {e}")
                continue

        if len(embeddings) < 2:
            # Not enough images to cluster
            if image_data:
                image_data[0]['cluster_id'] = 0
                valid_images[0].cluster_id = 0

            return {
                'clusters': [{
                    'id': 0,
                    'name': 'All',
                    'images': image_data,
                    'color': '#22c55e',
                    'count': len(image_data)
                }],
                'palette': self._aggregate_palette(image_data),
                'mood': image_data[0]['mood'] if image_data else 'neutral',
                'images': image_data
            }

        # Cluster embeddings
        embeddings_array = np.array(embeddings)
        n_clusters = min(n_clusters, len(embeddings))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings_array)

        # Assign cluster IDs
        for i, label in enumerate(labels):
            image_data[i]['cluster_id'] = int(label)
            valid_images[i].cluster_id = int(label)

        # Build cluster info
        cluster_colors = ['#ef4444', '#f97316', '#22c55e', '#38bdf8', '#a855f7', '#ec4899']
        clusters = []
        for c in range(n_clusters):
            cluster_images = [img for i, img in enumerate(image_data) if labels[i] == c]
            if cluster_images:
                # Determine cluster mood by majority vote
                moods = [img['mood'] for img in cluster_images]
                dominant_mood = Counter(moods).most_common(1)[0][0]
                clusters.append({
                    'id': int(c),
                    'name': dominant_mood.title(),
                    'images': cluster_images,
                    'color': cluster_colors[c % len(cluster_colors)],
                    'count': len(cluster_images)
                })

        # Aggregate palette
        palette = self._aggregate_palette(image_data)

        # Overall mood
        all_moods = [img['mood'] for img in image_data]
        overall_mood = Counter(all_moods).most_common(1)[0][0] if all_moods else 'neutral'

        return {
            'clusters': clusters,
            'palette': palette,
            'mood': overall_mood,
            'images': image_data
        }

    def _load_image(self, url: str) -> Image.Image:
        """Load image from URL or local file path"""
        try:
            # Check if it's a local file path
            if url.startswith('/uploads/') or url.startswith('./uploads/') or url.startswith('uploads/'):
                # Convert to absolute path
                if url.startswith('/uploads/'):
                    filepath = '.' + url
                elif url.startswith('./'):
                    filepath = url
                else:
                    filepath = './' + url

                if os.path.exists(filepath):
                    return Image.open(filepath).convert('RGB')
                else:
                    print(f"Local file not found: {filepath}")
                    return None
            else:
                # It's a URL, fetch from network
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                return Image.open(BytesIO(response.content)).convert('RGB')
        except Exception as e:
            print(f"Error loading image from {url}: {e}")
            return None

    def _get_embedding(self, image: Image.Image) -> np.ndarray:
        """Get CLIP embedding for image"""
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            features = _ensure_tensor(self.model.get_image_features(**inputs))
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().flatten()

    def _classify_mood(self, image: Image.Image) -> str:
        """Classify image mood using CLIP zero-shot"""
        try:
            text_inputs = self.processor(text=self.mood_labels, return_tensors="pt", padding=True).to(self.device)
            image_inputs = self.processor(images=image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                image_features = _ensure_tensor(self.model.get_image_features(**image_inputs))
                text_features = _ensure_tensor(self.model.get_text_features(**text_inputs))
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                sims = (image_features @ text_features.T).squeeze().cpu().numpy()

            return self.mood_labels[np.argmax(sims)]
        except Exception as e:
            print(f"Mood classification error: {e}")
            return "neutral"

    def _extract_colors(self, image: Image.Image, n_colors: int = 3) -> List[str]:
        """Extract dominant colors from image"""
        try:
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
            return ['#808080', '#a0a0a0', '#606060']

    def _aggregate_palette(self, image_data: List[Dict]) -> List[str]:
        """Aggregate color palette from all images"""
        all_colors = []
        for img in image_data:
            all_colors.extend(img.get('colors', []))

        if not all_colors:
            return ['#22c55e', '#38bdf8', '#f97316']

        # Count and return most common
        color_counts = Counter(all_colors)
        return [c for c, _ in color_counts.most_common(8)]
