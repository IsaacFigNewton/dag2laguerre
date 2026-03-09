from typing import Dict, Any

DEFAULT_CONFIG: Dict[str, Any] = {
    # --- solver defaults (used by _solve_node) ---
    "lloyd_iters": 8,
    "site_step": 0.7,
    "fit_iters": 60,
    "fit_tol_rel": 2e-2,
    # --- draw defaults ---
    "cell_alpha": 0.35,
    "cell_edge_linewidth": 1.2,
    "cell_fill_linewidth": 1.2,
    "cell_edge_color": (0, 0, 0, 0.35),
    "label_fontsize": 9,
    # --- show defaults ---
    "figsize": (7, 7),
    "xlim": (-0.02, 1.02),
    "ylim": (-0.02, 1.02),
    "root_outline_linewidth": 2.0,
}