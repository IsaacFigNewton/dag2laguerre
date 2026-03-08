# Recursive (hierarchical) Laguerre / power diagram demo.

This module builds *area-constrained* power diagrams (a.k.a. Laguerre diagrams)
inside an arbitrary convex polygonal region, then recursively subdivides each
cell according to a user-provided hierarchy.

## Key ideas
---------
1) **Power diagram cell construction**:
   Each cell is computed as an intersection of half-planes derived from the
   Laguerre distance inequality between sites i and j.

2) **Area targeting via weight fitting**:
   Laguerre weights are iteratively adjusted so each cell's area matches a
   target area (up to tolerance).

3) **Lloyd-style relaxation**:
   With weights fitted, sites are moved toward their cell centroids to improve
   cell quality (damped Lloyd steps), while maintaining area constraints via
   refitting.

4) **Hierarchical field coloring**:
   Per-level siblings are colored consistently using a centroid->RGB mapping
   that is *normalized by the current parent region's bounding box*. This keeps
   colors stable within each subtree, rather than shifting as the global domain
   changes.

## Notes / assumptions
-------------------
- Regions are treated as convex polygons. The clipping method assumes convexity.
- Degenerate / empty cells can occur during fitting; we resample sites as needed.
- Coordinates are assumed 2D and stored as float numpy arrays of shape (N, 2).