import math
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt

from ..geometry.PolygonOps import PolygonOps
from .CentroidColoring import CentroidColoring

class HierarchicalFieldColorer:
    """
    Hierarchy-aware adapter for centroid field coloring.

    The core centroid->RGB mapping (`CentroidColoring`) depends on the bbox used
    to normalize centroid coordinates. If you used a *global* bbox for an entire
    plot, colors inside a small subcell can become nearly constant (because the
    subcell occupies only a tiny fraction of the global range).

    This adapter fixes that by using the current **parent region bbox** as the
    normalization frame. Result: all siblings inside the same parent region
    share a stable, high-contrast palette that doesn't depend on the rest of the
    diagram.

    Usage
    -----
    - Call `colors_for_cells(cells, parent_region)` right before drawing the
      siblings for a node.
    - Pass the same parent region you used to solve those cells.

    Notes
    -----
    - For empty/degenerate cells we fall back to the parent region centroid.
    """

    def __init__(self, poly_ops: PolygonOps, *, angles: tuple[float, float, float] = (0.9, 0.6, 0.2)):
        self.poly_ops = poly_ops
        self.angles = angles

    def colors_for_cells(self, cells: list[np.ndarray], region: np.ndarray) -> np.ndarray:
        """
        Compute per-cell RGB colors for a node's children.

        Parameters
        ----------
        cells:
            List of polygons (each (Ni,2)). Cells may be empty/degenerate.
        region:
            The parent polygon (convex) in which these `cells` live. Its bbox is
            used as the normalization domain to keep sibling colors consistent.

        Returns
        -------
        np.ndarray:
            Array (K, 3) of RGB values in [0, 1] where K = len(cells).
        """
        region = np.asarray(region, dtype=float)

        xmin, ymin = region.min(axis=0)
        xmax, ymax = region.max(axis=0)
        bbox = (float(xmin), float(xmax), float(ymin), float(ymax))

        parent_center = region.mean(axis=0)
        cents: list[np.ndarray] = []
        for c in cells:
            if len(c) >= 3:
                cents.append(self.poly_ops.centroid(c))
            else:
                cents.append(parent_center)

        return CentroidColoring.centroids_to_rgb(np.asarray(cents, float), bbox, angles=self.angles)