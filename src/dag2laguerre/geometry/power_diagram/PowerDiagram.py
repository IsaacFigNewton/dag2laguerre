import math
from typing import Iterable

import numpy as np

from ..HalfplaneClipper import HalfPlaneClipper

class PowerDiagram:
    """
    Construct Laguerre / power diagram cells by half-plane intersection.

    For sites s_i with weights w_i, the Laguerre cell of i is:

      C_i = { x : ||x - s_i||^2 - w_i <= ||x - s_j||^2 - w_j  for all j }

    Each inequality yields a half-plane in x, so the cell is an intersection of
    half-planes, clipped to the parent region.
    """

    def __init__(self, clipper: HalfPlaneClipper):
        self.clipper = clipper

    def cell(self, i: int, sites: np.ndarray, weights: np.ndarray, clip: np.ndarray) -> np.ndarray:
        """
        Compute cell i of the power diagram clipped to `clip` polygon.

        Parameters
        ----------
        i:
            Index of the site.
        sites:
            Array (K,2) of site positions.
        weights:
            Array (K,) of Laguerre weights.
        clip:
            Convex polygon region used as a bounding domain.

        Returns
        -------
        np.ndarray:
            Polygon vertices for cell i. Empty/degenerate polygons possible.
        """
        sites = np.asarray(sites, dtype=float)
        weights = np.asarray(weights, dtype=float)
        clip = np.asarray(clip, dtype=float)

        si, wi = sites[i], weights[i]
        si2 = float(si @ si)
        poly = clip.copy()

        for j in range(len(sites)):
            if j == i:
                continue

            sj, wj = sites[j], weights[j]

            # Derivation:
            # ||x - si||^2 - wi <= ||x - sj||^2 - wj
            # expands to: 2(sj - si)·x <= (||sj||^2 - wj) - (||si||^2 - wi)
            a, b = 2.0 * (sj - si)
            c = (float(sj @ sj) - float(wj)) - (si2 - float(wi))

            poly = self.clipper.clip(poly, float(a), float(b), float(c))
            if len(poly) < 3:
                return np.empty((0, 2), dtype=float)

        return poly

    def cells(self, sites: np.ndarray, weights: np.ndarray, clip: np.ndarray) -> list[np.ndarray]:
        """Compute all power diagram cells, one per site."""
        return [self.cell(i, sites, weights, clip) for i in range(len(sites))]