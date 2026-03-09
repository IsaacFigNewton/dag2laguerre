from __future__ import annotations

import math
from typing import Iterable, Dict, FrozenSet, Any, Optional

import numpy as np
import matplotlib.pyplot as plt
from .geometry import *
from .geometry.power_diagram import *
from .config import DEFAULT_CONFIG

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
    cmap:
        Matplotlib colormap name.
    config:
        Dict of renderer/solver parameters. See DEFAULT_CONFIG.

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
        cmap: str = "hsv",
        label_map: Dict = dict(),
        config: Dict[str, Any] = DEFAULT_CONFIG,
    ):
        self.h = hierarchy
        self.rng = np.random.default_rng(seed)
        self.root = np.asarray(root, dtype=float)

        # Merge defaults with user overrides (user keys win)
        self.config: Dict[str, Any] = config

        # Geometry + solver pipeline
        self.poly = PolygonOps(self.rng)
        self.clipper = HalfPlaneClipper(eps=self.poly.EPS)
        self.diagram = PowerDiagram(self.clipper)
        self.targets = CapacityTargets(self.poly)
        self.fitter = WeightFitter(self.poly, self.diagram)
        self.lloyd = LloydRelaxer(self.poly, self.diagram, self.fitter)

        # Hierarchy-consistent coloring
        self.cmap = plt.get_cmap(cmap)
        self.label_map = label_map
        self.cell_colors = self._partition_cmap(self.h, 0, 1)

    def _partition_cmap(
        self,
        subtree: Dict[FrozenSet, Any],
        start,
        end,
    ) -> Dict[FrozenSet, np.ndarray]:
        keys = list(subtree.keys())
        n = len(keys)

        if n == 0:
            return {}

        width = (end - start) / n
        results = {}
        for i, k in enumerate(keys):
            child_start = start + i * width
            child_end = child_start + width
            mid = (child_start + child_end) / 2

            color = self.cmap(mid)[:3]
            results[k] = color

            v = subtree[k]
            if isinstance(v, dict):
                results.update(self._partition_cmap(v, child_start, child_end))
        return results

    def _label(self, key, set_labels:bool) -> str:
        """Pretty label for a node key (assumes iterable of sortable items)."""
        fallback = "{" + ",".join(map(str, sorted(key))) + "}" if set_labels else ""
        return self.label_map.get(key, fallback)

    def _solve_node(
        self,
        node_label,
        keys: list,
        region: np.ndarray,
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
        # 1) Sample initial sites inside the region
        sites0 = np.array([self.poly.sample_in(region) for _ in range(K)], dtype=float)
        # 2) Compute target areas from CapacityTargets
        target_areas = self.targets.target_areas(node_label, keys, region)

        # 3) Run LloydRelaxer to obtain stable sites + fitted weights + final cells
        sites, weights, cells = self.lloyd.relax(
            sites0,
            target_areas,
            region,
            lloyd_iters=self.config["lloyd_iters"],
            site_step=self.config["site_step"],
            fit_iters=self.config["fit_iters"],
            fit_tol_rel=self.config["fit_tol_rel"],
        )
        return sites, weights, cells

    def _draw(self,
              ax,
              node_label,
              node: dict,
              region: np.ndarray,
              labels: bool,
              set_labels: bool,
              depth: int = 0
        ) -> None:
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

        for (key, site, cell) in zip(keys, sites, cells):
            if len(cell) < 3:
                continue

            # Fill cell with hierarchy-consistent color
            ax.fill(
                cell[:, 0],
                cell[:, 1],
                color=self.cell_colors[key],
                alpha=self.config["cell_alpha"],
                linewidth=self.config["cell_fill_linewidth"],
            )

            # Draw cell boundary
            ax.plot(
                np.r_[cell[:, 0], cell[0, 0]],
                np.r_[cell[:, 1], cell[0, 1]],
                linewidth=self.config["cell_edge_linewidth"],
                color=self.config["cell_edge_color"],
            )

            # Optional label at the site
            if labels:
                ax.text(
                    site[0],
                    site[1],
                    self._label(key, set_labels),
                    fontsize=self.config["label_fontsize"],
                    ha="center",
                    va="center",
                )

            # Recurse into child subtree if present
            child = node.get(key)
            if isinstance(child, dict) and child:
                self._draw(ax, key, child, cell, labels, set_labels, depth + 1)

    def show(self, figsize=None, labels: bool = True, set_labels: bool = True) -> None:
        """Render the full hierarchy starting from the root region."""
        figsize = self.config["figsize"] if figsize is None else figsize

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(*self.config["xlim"])
        ax.set_ylim(*self.config["ylim"])
        ax.axis("off")

        # Root label: union of top-level keys (works if keys are sets/frozensets)
        try:
            root_label = set().union(*self.h)
        except TypeError:
            root_label = "root"

        self._draw(ax, root_label, self.h, self.root, labels, set_labels, depth=0)

        # Outline root region
        r = self.poly.ccw(self.root)
        ax.plot(
            np.r_[r[:, 0], r[0, 0]],
            np.r_[r[:, 1], r[0, 1]],
            linewidth=self.config["root_outline_linewidth"],
        )

        plt.tight_layout()
        plt.show()