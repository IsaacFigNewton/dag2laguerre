import math
from typing import Iterable

import numpy as np

class HalfPlaneClipper:
    """
    Clip a convex polygon against a half-plane.

    Implements one step of Sutherland–Hodgman clipping specialized to a single
    inequality:

        a*x + b*y <= c

    This is the core primitive used to compute power diagram cells as successive
    half-plane intersections.

    Notes
    -----
    - Assumes the input polygon is convex.
    - Produces a possibly empty polygon (len < 3 => degenerate).
    """

    def __init__(self, eps: float = 1e-9):
        self.EPS = eps

    def clip(self, poly: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        """
        Clip `poly` by the half-plane a*x + b*y <= c.

        Parameters
        ----------
        poly:
            Convex polygon vertices (N,2).
        a, b, c:
            Half-plane coefficients.

        Returns
        -------
        np.ndarray:
            Clipped polygon vertices (M,2). May be empty or degenerate.
        """
        poly = np.asarray(poly, dtype=float)
        if len(poly) == 0:
            return poly

        def inside(p: np.ndarray) -> bool:
            # Small tolerance avoids numeric cracks at boundaries
            return a * p[0] + b * p[1] <= c + 1e-12

        def intersect(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
            """
            Line segment intersection with the boundary a*x + b*y = c.

            Returns a point on [p1, p2] (clamped) even if nearly parallel.
            """
            d = p2 - p1
            denom = a * d[0] + b * d[1]
            if abs(denom) < self.EPS:
                # Segment nearly parallel to boundary; choose p2 as a safe fallback
                return p2
            t = (c - (a * p1[0] + b * p1[1])) / denom
            t = float(np.clip(t, 0.0, 1.0))
            return p1 + t * d

        out: list[np.ndarray] = []
        n = len(poly)

        for i in range(n):
            cur = poly[i]
            nxt = poly[(i + 1) % n]
            ic, inx = inside(cur), inside(nxt)

            if ic and inx:
                out.append(nxt)
            elif ic and not inx:
                out.append(intersect(cur, nxt))
            elif (not ic) and inx:
                out.append(intersect(cur, nxt))
                out.append(nxt)

        out_arr = np.asarray(out, dtype=float)

        # Remove near-duplicate consecutive points (stabilizes downstream area/centroid)
        if len(out_arr) >= 2:
            keep = [0]
            for i in range(1, len(out_arr)):
                if np.linalg.norm(out_arr[i] - out_arr[keep[-1]]) > 1e-10:
                    keep.append(i)
            out_arr = out_arr[keep]

        return out_arr