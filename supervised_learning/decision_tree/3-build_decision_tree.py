#!/usr/bin/env python3
"""Decision tree implementation."""
import numpy as np


class Node:
    """Tree node that splits data based on a feature and threshold."""

    def __init__(
        self,
        feature=None,
        threshold=None,
        left_child=None,
        right_child=None,
        is_root=False,
        depth=0,
    ):
        """Create a node with feature, threshold, children and depth."""
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """Return maximum depth of this node and its children."""
        if self.right_child is None and self.left_child is None:
            return self.depth

        max_depth = self.depth

        if self.right_child is not None:
            max_depth = max(max_depth, self.right_child.max_depth_below())

        if self.left_child is not None:
            max_depth = max(max_depth, self.left_child.max_depth_below())

        return max_depth

    def count_nodes(self, only_leaves=False):
        """Return the number of nodes in the tree."""
        count = 0
        if not only_leaves or self.is_leaf:
            count += 1
        if self.left_child is not None:
            count += self.right_child.count_nodes(only_leaves)
        if self.right_child is not None:
            count += self.left_child.count_nodes(only_leaves)
        return count

    def left_child_add_prefix(self, text):
        """Add visual prefix for left child."""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += "    |  " + x + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """Add visual prefix for right child."""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += "       " + x + "\n"
        return new_text

    def __str__(self):
        """Return visual string of the tree with root label."""
        if self.is_root:
            result = (
                f"root [feature={self.feature}, "
                f"threshold={self.threshold}]"
            )
        else:
            result = (
                f"-> node [feature={self.feature}, "
                f"threshold={self.threshold}]"
            )

        if self.left_child is not None:
            left_part = str(self.left_child)
            result += "\n" + self.left_child_add_prefix(left_part).rstrip()

        if self.right_child is not None:
            right_part = str(self.right_child)
            result += "\n" + self.right_child_add_prefix(right_part).rstrip()

        return result

    def get_leaves_below(self):
        """Return list of leaves."""
        leaves = []
        if self.right_child is not None:
            leaves += self.left_child.get_leaves_below()
        if self.left_child is not None:
            leaves += self.right_child.get_leaves_below()
        if self.is_leaf:
            leaves.append(self)

        return leaves


class Leaf(Node):
    """Leaf node that stores a prediction value."""

    def __init__(self, value, depth=None):
        """Create a leaf with value and depth."""
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """Return the depth of this leaf."""
        return self.depth

    def __str__(self):
        """Return visual string of the leaf."""
        return f"-> leaf [value={self.value}]"

    def get_leaves_below(self):
        """Return list of leaves."""
        return [self]


class Decision_Tree:
    """Decision tree that builds from nodes and leaves."""

    def __init__(
        self,
        max_depth=10,
        min_pop=1,
        seed=0,
        split_criterion="random",
        root=None,
    ):
        """Create a tree with max depth, min samples, seed and split rule."""
        self.rng = np.random.default_rng(seed)
        self.root = root if root else Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """Return maximum depth of the tree."""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Return the number of nodes in the tree."""
        return self.root.count_nodes(only_leaves=only_leaves)

    def __str__(self):
        """Return visual string of the tree."""
        return str(self.root)

    def get_leaves(self):
        """Return leaves of the tree."""
        return self.root.get_leaves_below()
