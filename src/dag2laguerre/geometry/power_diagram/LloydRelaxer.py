import math
from typing import Iterable

import numpy as np

from ..PolygonOps import PolygonOps
from .PowerDiagram import PowerDiagram
from .WeightFitter import WeightFitter

class LloydRelaxer:
    """
    Relax sites while maintaining area constraints.

    Each Lloyd iteration:
      1) Fit weights to match target areas (for the current sites).
      2) Move each site toward its cell centroid (damped).
      3) If any cell degenerates, resample that site's position in the parent region.

    The combination tends to produce nicer (more "round") cells without losing
    the capacity/area proportions.
    """

    def __init__(self, poly_ops: PolygonOps, diagram: PowerDiagram, fitter: WeightFitter):
        self.poly_ops = poly_ops
        self.diagram = diagram
        self.fitter = fitter

    def relax(
        self,
        sites: np.ndarray,
        target_areas: np.ndarray,
        region: np.ndarray,
        lloyd_iters: int = 8,
        site_step: float = 0.7,
        fit_iters: int = 60,
        fit_tol_rel: float = 2e-2,
    ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
        """
        Run damped Lloyd relaxation with repeated weight fitting.

        Parameters
        ----------
        sites:
            Initial site positions (K,2).
        target_areas:
            Desired cell areas (K,).
        region:
            Parent region polygon (convex).
        lloyd_iters:
            Number of Lloyd iterations.
        site_step:
            Damping factor in [0,1]. 1.0 jumps directly to centroids; smaller is steadier.
        fit_iters:
            Weight fitting iterations per Lloyd step.
        fit_tol_rel:
            Relative tolerance for weight fitting.

        Returns
        -------
        sites:
            Relaxed site positions (K,2).
        weights:
            Final fitted weights (K,).
        cells:
            Final cell polygons (list length K).
        """
        sites = np.asarray(sites, dtype=float).copy()
        region = np.asarray(region, dtype=float)

        K = len(sites)
        weights = np.zeros(K, dtype=float)
        cells: list[np.ndarray] = [np.empty((0, 2), dtype=float) for _ in range(K)]

        for _ in range(lloyd_iters):
            weights, cells, _areas = self.fitter.fit(
                sites, target_areas, region, max_iter=fit_iters, tol_rel=fit_tol_rel
            )

            # Prevent collapse: if a cell is empty/degenerate, move its site somewhere valid
            for i, c in enumerate(cells):
                if len(c) < 3:
                    sites[i] = self.poly_ops.sample_in(region)

            # Lloyd step: move sites toward their cell centroids
            new_sites = sites.copy()
            for i, c in enumerate(cells):
                if len(c) < 3:
                    continue

                cen = self.poly_ops.centroid(c)

                # Rare fallback: centroid outside parent region due to numeric issues
                if not self.poly_ops.point_in_poly(cen, region):
                    cen = self.poly_ops.sample_in(c)

                new_sites[i] = (1.0 - site_step) * sites[i] + site_step * cen

            sites = new_sites

        # Final tighter fit
        weights, cells, _areas = self.fitter.fit(
            sites,
            target_areas,
            region,
            max_iter=max(2 * fit_iters, 120),
            tol_rel=min(fit_tol_rel, 1.5e-2),
        )
        return sites, weights, cells