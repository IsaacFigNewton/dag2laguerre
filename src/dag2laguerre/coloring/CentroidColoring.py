import math
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt


class CentroidColoring:
    """
    Map 2D centroids to RGB by lifting to 3D, rotating, and normalizing.

    This is a lightweight "field coloring" that produces visually distinct but
    spatially coherent colors (nearby centroids have related colors).

    Pipeline
    --------
    1) Normalize centroid (x, y) into [-1, 1] using a provided domain bbox.
    2) Lift to 3D by defining a simple z = f(x, y). (Here: z = 0.5*(x+y))
    3) Apply a 3D rotation (acts like choosing a projection).
    4) Normalize rotated coordinates to [0, 1] to obtain an RGB triplet.

    Parameters that matter
    ----------------------
    - `domain_bbox`: controls how normalization behaves. If you pass the parent
      region bbox (hierarchical use case), siblings share a consistent palette.
    - `angles`: rotation angles (radians) about x/y/z. Different angles yield
      different "palettes" while preserving coherence.

    Returned values
    ---------------
    - RGB values in [0, 1], shape (K, 3).
    """

    @staticmethod
    def _rotation_matrix(ax: float, ay: float, az: float) -> np.ndarray:
        """Create a 3D rotation matrix from Euler angles (x then y then z)."""
        cx, sx = math.cos(ax), math.sin(ax)
        cy, sy = math.cos(ay), math.sin(ay)
        cz, sz = math.cos(az), math.sin(az)

        Rx = np.array([[1, 0, 0],
                       [0, cx, -sx],
                       [0, sx, cx]], dtype=float)
        Ry = np.array([[cy, 0, sy],
                       [0, 1, 0],
                       [-sy, 0, cy]], dtype=float)
        Rz = np.array([[cz, -sz, 0],
                       [sz,  cz, 0],
                       [0,   0,  1]], dtype=float)

        # Apply Rx first, then Ry, then Rz (right-multiplication on column vectors)
        return Rz @ Ry @ Rx

    @staticmethod
    def centroids_to_rgb(
        centroids_xy: np.ndarray,
        domain_bbox: tuple[float, float, float, float],
        *,
        angles: tuple[float, float, float] = (0.9, 0.6, 0.2),
        eps: float = 1e-12,
    ) -> np.ndarray:
        """
        Convert centroid coordinates into RGB colors.

        Parameters
        ----------
        centroids_xy:
            Array of shape (K, 2).
        domain_bbox:
            (xmin, xmax, ymin, ymax) used to normalize x and y into [-1, 1].
        angles:
            Euler rotation angles (ax, ay, az) in radians.
        eps:
            Numerical stabilizer for degenerate bounding boxes.

        Returns
        -------
        np.ndarray:
            Array of shape (K, 3) with values clamped to [0, 1].
        """
        xmin, xmax, ymin, ymax = domain_bbox
        centroids_xy = np.asarray(centroids_xy, dtype=float)

        x = centroids_xy[:, 0]
        y = centroids_xy[:, 1]

        # Normalize to [-1, 1] within bbox
        xn = 2.0 * (x - xmin) / max(xmax - xmin, eps) - 1.0
        yn = 2.0 * (y - ymin) / max(ymax - ymin, eps) - 1.0

        # Lift to 3D (simple plane field)
        zn = 0.5 * (xn + yn)

        P = np.stack([xn, yn, zn], axis=1)  # (K, 3)

        # Rotate in 3D, then normalize to [0, 1] channel-wise
        R = CentroidColoring._rotation_matrix(*angles)
        Q = (R @ P.T).T  # (K, 3)

        qmin = Q.min(axis=0)
        qmax = Q.max(axis=0)
        rgb = (Q - qmin) / (qmax - qmin + eps)
        return np.clip(rgb, 0.0, 1.0)