import math
from typing import Iterable

import numpy as np

class PolygonOps:
    """
    Small collection of polygon utility routines.

    The power diagram and solver treat regions as convex polygons, but some of
    these helpers (centroid, point-in-poly) also work for general simple
    polygons.

    Notes
    -----
    - `ccw` ensures consistent winding (counter-clockwise) which is useful for
      area/centroid computations.
    - `sample_in` uses rejection sampling in the polygon bbox; it's simple and
      robust enough for moderate sizes.
    """

    EPS = 1e-9

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    @staticmethod
    def area_signed(poly: np.ndarray) -> float:
        """Signed polygon area via the shoelace formula (CCW => positive)."""
        poly = np.asarray(poly, dtype=float)
        x, y = poly[:, 0], poly[:, 1]
        return 0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))

    @classmethod
    def area_abs(cls, poly: np.ndarray) -> float:
        """Absolute polygon area."""
        return abs(cls.area_signed(poly))

    @classmethod
    def ccw(cls, poly: np.ndarray) -> np.ndarray:
        """
        Ensure polygon vertices are ordered counter-clockwise.

        Returns a float array copy if a reversal is needed.
        """
        poly = np.asarray(poly, dtype=float)
        return poly if cls.area_signed(poly) > 0 else poly[::-1].copy()

    def point_in_poly(self, pt: np.ndarray, poly: np.ndarray) -> bool:
        """
        Ray casting point-in-polygon test.

        Parameters
        ----------
        pt:
            Query point (2,).
        poly:
            Polygon vertices (N,2).

        Returns
        -------
        bool:
            True if point is inside polygon (boundary behavior is tolerant).
        """
        x, y = float(pt[0]), float(pt[1])
        poly = np.asarray(poly, dtype=float)
        inside = False
        n = len(poly)

        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]

            # Does the segment straddle the horizontal ray from pt?
            crosses = ((y1 > y) != (y2 > y)) and (
                x < (x2 - x1) * (y - y1) / (y2 - y1 + self.EPS) + x1
            )
            if crosses:
                inside = not inside

        return inside

    def sample_in(self, poly: np.ndarray, tries: int = 10_000) -> np.ndarray:
        """
        Sample a random point inside a polygon.

        Uses rejection sampling within the polygon bbox. If sampling fails after
        `tries` attempts (rare for skinny polygons), falls back to polygon mean.

        Parameters
        ----------
        poly:
            Polygon vertices (N,2).
        tries:
            Maximum number of rejection attempts.

        Returns
        -------
        np.ndarray:
            Point (2,) inside `poly` (best-effort).
        """
        poly = np.asarray(poly, dtype=float)
        mn, mx = poly.min(axis=0), poly.max(axis=0)

        for _ in range(tries):
            p = self.rng.uniform(mn, mx)
            if self.point_in_poly(p, poly):
                return p

        return poly.mean(axis=0)

    @staticmethod
    def centroid(poly: np.ndarray) -> np.ndarray:
        """
        Area-weighted polygon centroid (robust for convex polygons).

        For degenerate polygons (tiny area) or <3 vertices, falls back to mean.

        Parameters
        ----------
        poly:
            Polygon vertices (N,2).

        Returns
        -------
        np.ndarray:
            Centroid (2,).
        """
        poly = np.asarray(poly, dtype=float)
        if len(poly) < 3:
            return poly.mean(axis=0) if len(poly) else np.array([0.0, 0.0], dtype=float)

        x = poly[:, 0]
        y = poly[:, 1]
        x2 = np.roll(x, -1)
        y2 = np.roll(y, -1)

        cross = x * y2 - x2 * y
        A = 0.5 * np.sum(cross)
        if abs(A) < 1e-12:
            return poly.mean(axis=0)

        cx = np.sum((x + x2) * cross) / (6.0 * A)
        cy = np.sum((y + y2) * cross) / (6.0 * A)
        return np.array([cx, cy], dtype=float)