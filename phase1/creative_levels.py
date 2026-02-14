"""Creative Level Estimation (Phase 1 - Section 3)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from typing import Dict

from config import DEFAULT_LEVEL_WEIGHTS, MODE_FACTORS
from data_types import BrandValues, CreativeLevelSpec, DivergenceValues


class CreativeLevelEstimator:
    """Estimate creative level weights and compute divergence per level"""

    def estimate_level_weights(self, brand_values: BrandValues, user_mode: str) -> CreativeLevelSpec:
        """
        Estimate weights for surface/flow/narrative based on brand values and user mode.

        Algorithm (from Phase 1 spec - section 3.2):
        1. Start with default weights
        2. Adjust based on user_mode (functional/aesthetic/flow)
        3. Adjust based on brand emotions
        4. Normalize to sum to 1.0

        Args:
            brand_values: Extracted brand values
            user_mode: User's selected mode

        Returns:
            CreativeLevelSpec with normalized weights
        """
        # Start with defaults
        weights = DEFAULT_LEVEL_WEIGHTS.copy()

        # Adjust based on user mode (from spec section 3.3)
        if user_mode == 'functional':
            weights['flow'] += 0.2
            weights['narrative'] -= 0.1
            weights['surface'] -= 0.1
        elif user_mode == 'aesthetic':
            weights['surface'] += 0.2
            weights['narrative'] += 0.1
            weights['flow'] -= 0.3
        elif user_mode == 'flow':
            weights['flow'] += 0.2
            weights['surface'] -= 0.1
            weights['narrative'] -= 0.1

        # Adjust based on brand emotions
        emotions = brand_values.emotions

        # High energy/intensity -> boost surface (visual impact)
        if emotions.get('energy', 0) > 0.6 or emotions.get('excitement', 0) > 0.6 or emotions.get('intensity', 0) > 0.6:
            weights['surface'] += 0.1

        # High playfulness -> boost flow (interaction)
        if emotions.get('playfulness', 0) > 0.6:
            weights['flow'] += 0.1

        # High melancholy/serenity -> boost narrative (meaning)
        if emotions.get('melancholy', 0) > 0.6 or emotions.get('serenity', 0) > 0.6:
            weights['narrative'] += 0.1

        # High boldness/chaos -> boost surface
        if emotions.get('bold', 0) > 0.6 or emotions.get('chaos', 0) > 0.6:
            weights['surface'] += 0.05

        # High calmness/order -> reduce surface, boost flow
        if emotions.get('calmness', 0) > 0.6 or emotions.get('order', 0) > 0.6:
            weights['surface'] -= 0.05
            weights['flow'] += 0.05

        # High surprise -> boost narrative
        if emotions.get('surprise', 0) > 0.6:
            weights['narrative'] += 0.1

        # Normalize to sum to 1.0
        total = sum(weights.values())
        if total > 0:
            weights = {k: max(0.0, v / total) for k, v in weights.items()}
        else:
            weights = DEFAULT_LEVEL_WEIGHTS.copy()

        return CreativeLevelSpec(
            surface=weights['surface'],
            flow=weights['flow'],
            narrative=weights['narrative']
        )

    def compute_divergence_per_level(self, D_global: float, user_mode: str, level_weights: CreativeLevelSpec) -> DivergenceValues:
        """
        Compute divergence for each creative level.

        Algorithm (from Phase 1 spec - section 3.3):
        D[lvl] = clamp(D_global * f_mode + 0.2 * w, 0.0, 1.0)

        Where:
        - f_mode = mode_factors[user_mode][lvl]
        - w = level_weights[lvl]

        Mode factors (from spec):
        - functional: conservative across all levels
        - aesthetic: high surface/narrative, moderate flow
        - flow: high flow, moderate surface/narrative

        Args:
            D_global: Global divergence dial [0, 1]
            user_mode: User mode (functional/aesthetic/flow)
            level_weights: Creative level weights

        Returns:
            DivergenceValues for each level
        """
        # Get mode factors
        if user_mode not in MODE_FACTORS:
            user_mode = 'aesthetic'  # Default fallback

        factors = MODE_FACTORS[user_mode]

        # Compute divergence per level
        D = {}
        for lvl in ["surface", "flow", "narrative"]:
            f_mode = factors[lvl]
            w = getattr(level_weights, lvl)

            # Apply formula from spec
            D[lvl] = np.clip(D_global * f_mode + 0.2 * w, 0.0, 1.0)

        return DivergenceValues(
            D_surface=D['surface'],
            D_flow=D['flow'],
            D_narrative=D['narrative']
        )
