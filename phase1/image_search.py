"""Image Search - Pexels API integration (from neurospiral pattern)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
from typing import List, Dict
from config import PEXELS_API_KEY
from data_types import ImageRef


class ImageSearcher:
    """Search images from Pexels API"""

    def __init__(self):
        self.pexels_key = PEXELS_API_KEY

    def search(self, query: str, num_results: int = 30) -> List[ImageRef]:
        """
        Search images using Pexels API.

        Args:
            query: Search query
            num_results: Number of images to return

        Returns:
            List of ImageRef objects
        """
        images = []

        # Try Pexels
        pexels_images = self._search_pexels(query, num_results)
        images.extend(pexels_images)

        # Fallback if Pexels fails
        if not images:
            images = self._fallback_images(query, num_results)

        return images[:num_results]

    def _search_pexels(self, query: str, num: int) -> List[ImageRef]:
        """Search using Pexels API"""
        images = []
        if num <= 0:
            return images

        per_page = min(num, 80)
        try:
            headers = {"Authorization": self.pexels_key}
            response = requests.get(
                f"https://api.pexels.com/v1/search?query={query}&per_page={per_page}",
                headers=headers,
                timeout=15
            )
            response.raise_for_status()

            if response.status_code == 200:
                data = response.json()
                for photo in data.get('photos', []):
                    images.append(ImageRef(
                        id=f"pexels_{photo['id']}",
                        url=photo['src']['large'],
                        thumbnail_url=photo['src']['medium'],
                        source='reference',
                        metadata={'photographer': photo.get('photographer', ''), 'alt': photo.get('alt', '')}
                    ))
        except Exception as e:
            print(f"Pexels search error: {e}")

        # Try related terms if we need more
        if len(images) < num:
            related_terms = [f"{query} design", f"{query} aesthetic", f"{query} art"]
            for term in related_terms:
                if len(images) >= num:
                    break
                try:
                    response = requests.get(
                        f"https://api.pexels.com/v1/search?query={term}&per_page=15",
                        headers={"Authorization": self.pexels_key},
                        timeout=10
                    )
                    response.raise_for_status()

                    if response.status_code == 200:
                        data = response.json()
                        for photo in data.get('photos', []):
                            img_id = f"pexels_{photo['id']}"
                            if not any(i.id == img_id for i in images):
                                images.append(ImageRef(
                                    id=img_id,
                                    url=photo['src']['large'],
                                    thumbnail_url=photo['src']['medium'],
                                    source='reference',
                                    metadata={'photographer': photo.get('photographer', ''), 'alt': photo.get('alt', '')}
                                ))
                except:
                    pass

        return images[:num]

    def _fallback_images(self, query: str, num: int) -> List[ImageRef]:
        """Generate Lorem Picsum fallback images"""
        return [
            ImageRef(
                id=f"picsum_{i}",
                url=f"https://picsum.photos/seed/{query}{i}/800/600",
                thumbnail_url=f"https://picsum.photos/seed/{query}{i}/400/300",
                source='reference',
                metadata={'query': query}
            ) for i in range(num)
        ]
