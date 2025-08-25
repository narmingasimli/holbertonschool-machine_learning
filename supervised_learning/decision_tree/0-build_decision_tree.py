#!/usr/bin/env python3
"""Self of init code"""
import numpy as np


class Node:
    """Represents an internal node in the decision tree."""

    def __init__(self, feature=None, threshold=None,
                 left_child=None, right_child=None,
                 is_root=False, depth=0):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False  # Not a leaf by default
        self.depth = depth

    def max_depth_below(self):
        """Calculates the maximum depth of the subtree rooted at this node."""
        if self.is_leaf:
            return self.depth
        # Recursively find max depth of left and right children
        left_depth = self.left_child.max_depth_below() if self.left_child else self.depth
        right_depth = self.right_child.max_depth_below() if self.right_child else self.depth
        return max(left_depth, right_depth)


class Leaf(Node):
    """Represents a terminal node (leaf) in the decision tree."""

    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True  # Always a leaf
        self.depth = depth

    def max_depth_below(self):
        """Returns the depth of this specific leaf node."""
        return self.depth


class Decision_Tree():
    """A basic implementation of a Decision Tree."""

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        self.rng = np.random.default_rng(seed)
        self.root = root if root else Node(is_root=True, depth=0)
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion

    def depth(self):
        """Returns the overall maximum depth of the entire decision tree."""
        return self.root.max_depth_below()
