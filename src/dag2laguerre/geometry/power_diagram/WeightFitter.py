import math
from typing import Iterable

import numpy as np

from ..PolygonOps import PolygonOps
from .PowerDiagram import PowerDiagram

class WeightFitter:
    """
    Fit Laguerre weights so power diagram cell areas match targets.

    Approach
    --------
    We use a simple iterative controller:

        w <- w + lr * (target_area - current_area) / region_area

    and re-center weights each iteration (since adding a constant to all weights
    does not change the diagram).

    This is not the most sophisticated optimal transport solver, but it is
    lightweight, robust for demos, and works well when combined with Lloyd
    relaxation.
    """

    def __init__(self, poly_ops: PolygonOps, diagram: PowerDiagram):
        self.poly_ops = poly_ops
        self.diagram = diagram

    def fit(
        self,
        sites: np.ndarray,
        target_areas: np.ndarray,
        region: np.ndarray,
        *,
        max_iter: int = 60,
        tol_rel: float = 2e-2,
        lr: float | None = None,
    ) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
        """
        Fit weights for fixed sites.

        Parameters
        ----------
        sites:
            Array (K,2) of site positions.
        target_areas:
            Desired cell areas (K,). Will be normalized to the region area.
        region:
            Parent region polygon (convex).
        max_iter:
            Maximum iterations.
        tol_rel:
            Stop when max relative error <= tol_rel.
        lr:
            Learning rate (if None, chosen from region scale).

        Returns
        -------
        weights:
            Array (K,) of fitted weights (mean-zero).
        cells:
            List of polygons (one per site).
        areas:
            Array (K,) of final cell areas.
        """
        sites = np.asarray(sites, dtype=float)
        region = np.asarray(region, dtype=float)

        K = len(sites)
        weights = np.zeros(K, dtype=float)

        # Heuristic lr from region diameter squared
        bbox = region.max(axis=0) - region.min(axis=0)
        d2 = float(bbox @ bbox) + self.poly_ops.EPS
        if lr is None:
            lr = 0.35 * d2

        region_area = self.poly_ops.area_abs(region) + self.poly_ops.EPS
        target_areas = np.asarray(target_areas, dtype=float)

        # Normalize targets to sum to region area
        s = float(target_areas.sum())
        if s <= 0:
            target_areas = np.ones(K, dtype=float) * (region_area / max(K, 1))
        else:
            target_areas = target_areas * (region_area / s)

        for _ in range(max_iter):
            cells = self.diagram.cells(sites, weights, region)
            areas = np.array(
                [self.poly_ops.area_abs(c) if len(c) >= 3 else 0.0 for c in cells],
                dtype=float,
            )

            err = target_areas - areas
            rel = float(np.max(np.abs(err) / (target_areas + 1e-12)))
            if rel <= tol_rel:
                return weights, cells, areas

            # Update weights: bigger-than-target cells should shrink => decrease w
            # (this sign convention works well empirically with the cell constructor above)
            weights += lr * (err / region_area)

            # Global shift doesn't matter; re-center to avoid drift
            weights -= weights.mean()

        # One final evaluation on exit
        cells = self.diagram.cells(sites, weights, region)
        areas = np.array(
            [self.poly_ops.area_abs(c) if len(c) >= 3 else 0.0 for c in cells],
            dtype=float,
        )
        return weights, cells, areas