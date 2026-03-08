import math
from typing import Iterable

import numpy as np

from ..PolygonOps import PolygonOps

class CapacityTargets:
    """
    Convert hierarchy logic into per-child target areas within a region.

    Current rule (preserved from your code)
    ---------------------------------------
    Each key is an iterable (e.g., a set/tuple of labels). For each key k, we
    count how many elements are *unique to k* among its siblings:

        unique(k) = k \\ union(all other sibling keys)

    Target area proportions are set proportional to |unique(k)|, then normalized
    to the region area. If all uniques are 0, we fallback to uniform proportions.

    If you later want a different targeting scheme, change only this class and
    the rest of the solver remains unchanged.
    """

    def __init__(self, poly_ops: PolygonOps):
        self.poly_ops = poly_ops

    def target_areas(self, node_label, keys: list[Iterable], region: np.ndarray) -> np.ndarray:
        """
        Compute target areas for each child in `keys`.

        Parameters
        ----------
        node_label:
            Label for the current node (unused in this default rule, but kept so
            you can use it for custom weighting in the future).
        keys:
            Sibling keys at this node.
        region:
            Parent region polygon (convex).

        Returns
        -------
        np.ndarray:
            Target areas (K,) summing to the parent region area.
        """
        sizes = []
        for k in keys:
            others = set().union(*(l for l in keys if l is not k))
            unique_values = set(k) - others
            sizes.append(len(unique_values))

        portions = np.array(sizes, dtype=float)
        if portions.sum() <= 0:
            portions = np.ones(len(keys), dtype=float) / max(len(keys), 1)
        else:
            portions /= portions.sum()

        A = self.poly_ops.area_abs(region)
        return portions * A