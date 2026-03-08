from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt
from .coloring import *
from .geometry import *
from .geometry.power_diagram import *

class RecursivePowerDiagram:
    """
    Recursive power diagram renderer with hierarchical centroid field coloring.

    Parameters
    ----------
    hierarchy:
        Nested dict structure where each node maps:
            key -> child_node_dict (or {} / missing for leaf)
        Keys are expected to be iterables (e.g. sets/tuples) if you use the
        default CapacityTargets logic.
    seed:
        RNG seed for reproducibility.
    root:
        Root convex polygon region (default: unit square).
    color_angles:
        Euler angles for the centroid field coloring.

    Example hierarchy
    -----------------
    hierarchy = {
        frozenset({1,2,3}): {
            frozenset({1}): {},
            frozenset({2,3}): {...}
        },
        frozenset({4,5}): {...}
    }
    """

    def __init__(
        self,
        hierarchy: dict,
        seed: int = 42,
        root=((0, 0), (1, 0), (1, 1), (0, 1)),
        *,
        color_angles: tuple[float, float, float] = (0.9, 0.6, 0.2),
    ):
        self.h = hierarchy
        self.rng = np.random.default_rng(seed)
        self.root = np.asarray(root, dtype=float)

        # Geometry + solver pipeline
        self.poly = PolygonOps(self.rng)
        self.clipper = HalfPlaneClipper(eps=self.poly.EPS)
        self.diagram = PowerDiagram(self.clipper)
        self.targets = CapacityTargets(self.poly)
        self.fitter = WeightFitter(self.poly, self.diagram)
        self.lloyd = LloydRelaxer(self.poly, self.diagram, self.fitter)

        # Hierarchy-consistent coloring
        self.colorer = HierarchicalFieldColorer(self.poly, angles=color_angles)

    @staticmethod
    def _label(key) -> str:
        """Pretty label for a node key (assumes iterable of sortable items)."""
        return "{" + ",".join(map(str, sorted(key))) + "}"

    def _solve_node(
        self,
        node_label,
        keys: list,
        region: np.ndarray,
        *,
        lloyd_iters: int = 8,
        site_step: float = 0.7,
        fit_iters: int = 60,
        fit_tol_rel: float = 2e-2,
    ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
        """
        Solve a single hierarchy node.

        Steps
        -----
        1) Sample initial sites inside the region.
        2) Compute target areas from CapacityTargets.
        3) Run LloydRelaxer to obtain stable sites + fitted weights + final cells.
        """
        K = len(keys)
        sites0 = np.array([self.poly.sample_in(region) for _ in range(K)], dtype=float)
        target_areas = self.targets.target_areas(node_label, keys, region)

        sites, weights, cells = self.lloyd.relax(
            sites0,
            target_areas,
            region,
            lloyd_iters=lloyd_iters,
            site_step=site_step,
            fit_iters=fit_iters,
            fit_tol_rel=fit_tol_rel,
        )
        return sites, weights, cells

    def _draw(self, ax, node_label, node: dict, region: np.ndarray, labels: bool, depth: int = 0) -> None:
        """
        Recursively draw the hierarchy within `region`.

        For each node:
          - Solve power diagram cells for its children.
          - Compute sibling colors using the parent region bbox (hierarchical coloring).
          - Draw each cell and recurse into its child subtree.
        """
        region = self.poly.ccw(region)
        keys = list(node.keys())
        if not keys:
            return

        sites, _weights, cells = self._solve_node(node_label, keys, region)

        # Hierarchy-aware colors: siblings share a palette based on THIS parent region.
        rgbs = self.colorer.colors_for_cells(cells, region)

        for (key, site, cell, rgb) in zip(keys, sites, cells, rgbs):
            if len(cell) < 3:
                continue

            ax.fill(cell[:, 0], cell[:, 1], color=rgb, alpha=0.35, linewidth=1.2)
            ax.plot(
                np.r_[cell[:, 0], cell[0, 0]],
                np.r_[cell[:, 1], cell[0, 1]],
                linewidth=1.2,
                color=(0, 0, 0, 0.35),
            )
            if labels:
                ax.text(site[0], site[1], self._label(key), fontsize=9, ha="center", va="center")

            child = node.get(key)
            if isinstance(child, dict) and child:
                self._draw(ax, key, child, cell, labels, depth + 1)

    def show(self, figsize=(7, 7), labels:bool=True) -> None:
        """Render the full hierarchy starting from the root region."""
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.axis("off")

        # Root label: union of top-level keys (works if keys are sets/frozensets)
        try:
            root_label = set().union(*self.h)
        except TypeError:
            root_label = "root"

        self._draw(ax, root_label, self.h, self.root, labels, depth=0)

        # Outline root region
        r = self.poly.ccw(self.root)
        ax.plot(np.r_[r[:, 0], r[0, 0]], np.r_[r[:, 1], r[0, 1]], linewidth=2.0)

        plt.tight_layout()
        plt.show()