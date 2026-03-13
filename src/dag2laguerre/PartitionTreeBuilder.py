"""
partition_tree_builder.py

Refactors the original functional code into a small, testable class with:
- Clear method boundaries
- Type hints
- Docstrings and inline comments
- A simple public API

Core idea:
1) Walk a directed graph from a root and collect "descendant sets" as frozensets.
2) Build a recursive partition tree from those sets by repeatedly splitting the
   most-overlapping pair into (A\C, B\C, C), where C = A∩B.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Iterable, Optional, Set, Tuple

import networkx as nx


PartitionTree = Dict[FrozenSet[Any], "PartitionTree"]

class PartitionTreeBuilder:
    """
    Builds a partition tree from descendant sets derived from a directed graph.

    Public usage patterns:
        builder = PartitionTreeBuilder(G)

        # 1) Build sets from a root, then compute the tree
        sets, descendants = builder.to_sets(root)
        tree = builder.get_partition_tree(sets)

        # 2) Convenience: do everything in one call
        tree = builder.build_from_root(root)

    Notes:
    - The graph is treated as a rooted *reachability* structure via successors().
    - The algorithms below assume the reachable subgraph from `root` is a DAG or,
      at minimum, does not contain cycles reachable from the root. If cycles exist,
      a recursion loop may occur.
    """

    def __init__(self, graph: nx.DiGraph):
        self.graph = graph
        
        self.descendant_hye_node_map = dict()
        for root in self.graph.nodes():
            if self.graph.in_degree(root) == 0:
                descendant_hyes, root_hye = self.get_descendant_hyes(root)
                self.descendant_hye_node_map.update(descendant_hyes)
        
        # get a nested dictionary representing the partition tree.
        self.partition_tree: PartitionTree =  self.get_partition_tree(set(self.descendant_hye_node_map.keys()))
        

    def get_descendant_hyes(self, root: Any) -> Tuple[Dict[FrozenSet[Any], Any], Set[Any]]:
        """
        Convert the reachable subgraph under `root` into a collection of frozensets.

        For every node `n` reachable from `root`, we compute the set of descendants
        of `n` including itself, and add that as a frozenset to the returned set.

        This mirrors the original behavior:
          - `child_sets` is the union of all descendant-sets from children
          - plus the descendant-set for the current `root`.

        Args:
            root: Node to start the recursion.

        Returns:
            (child_sets, all_descendants)
            - child_sets: dict of frozenset-parent node pairs, each representing a descendant set
            - all_descendants: the (mutable) set of descendants of `root` including itself
        """
        children = list(self.graph.successors(root))

        # Collect all descendant-sets contributed by children.
        child_sets: Dict[FrozenSet[Any], Any] = dict()

        # This node is always part of its own descendant set.
        all_descendants: Set[Any] = {root}

        for child in children:
            # Recursively compute:
            # - all sets found under the child
            # - the child's full descendant set
            new_child_sets, descendants = self.get_descendant_hyes(child)

            # Merge results into our running aggregates.
            child_sets.update(new_child_sets)
            all_descendants.update(descendants)

        # Add the descendant set for this root itself.
        child_sets[frozenset(all_descendants)] = root

        return child_sets, all_descendants

    def get_partition_tree(self, sets: Set[FrozenSet[Any]]) -> PartitionTree:
        """
        Recursively build a partition tree from a set of frozensets.

        Algorithm (same as original):
          - Base case: 0 or 1 sets -> return empty subtree {}
          - Find the pair (A, B) with maximum overlap |A∩B|
          - Define:
                C  = A ∩ B
                A1 = A - C
                B1 = B - C
            Then recurse on each non-empty partition using subsets restricted
            to that partition's elements.

        Args:
            sets: A set of frozensets to partition.

        Returns:
            Nested dict mapping each partition (as a frozenset) to its subtree.
        """
        if len(sets) <= 1:
            return {}

        pair = self.get_max_overlapping_pair(sets)
        if pair is None:
            # Defensive: should only happen if `sets` is empty or malformed.
            return {}

        A, B = pair
        C = A.intersection(B)
        partitions = {
            A: A - C,
            B: B - C,
            C: C
        }

        # Build children for each non-empty partition.
        tree: PartitionTree = {}
        for key, partition in partitions.items():
            if not partition:
                continue
            # key = frozenset(partition)
            tree[partition] = self.get_partition_tree(self.get_subsets(partition, sets))
        return tree

    def get_max_overlapping_pair(
        self, sets: Set[FrozenSet[Any]]
    ) -> Optional[Tuple[FrozenSet[Any], FrozenSet[Any]]]:
        """
        Return the pair of sets with the maximum intersection size.

        Args:
            sets: Set of frozensets to compare.

        Returns:
            (A, B) maximizing |A∩B|, or None if fewer than 2 sets are provided.
        """
        if len(sets) < 2:
            return None

        best_pair: Optional[Tuple[FrozenSet[Any], FrozenSet[Any]]] = None
        best_overlap = -1

        # sets_list = sorted(list(sets), key=lambda x: len(x), reverse=True)
        sets_list = list(sets)

        # Brute force O(n^2) over all pairs.
        for i in range(len(sets_list)):
            for j in range(i + 1, len(sets_list)):
                A, B = sets_list[i], sets_list[j]
                overlap = len(A & B)

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_pair = (A, B)


        return best_pair

    def get_subsets(
        self, seed: Iterable[Any], sets: Set[FrozenSet[Any]]
    ) -> Set[FrozenSet[Any]]:
        """
        Filter `sets` down to only those frozensets that are subsets of `seed`.

        Args:
            seed: Elements defining the universe to keep.
            sets: Candidate frozensets.

        Returns:
            All s in `sets` such that s ⊆ seed.
        """
        seed_set = set(seed)
        return {s for s in sets if set(s).issubset(seed_set)}