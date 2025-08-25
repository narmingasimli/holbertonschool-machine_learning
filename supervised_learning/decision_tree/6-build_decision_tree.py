#!/usr/bin/env python3
"""decision tree code."""

import numpy as np


class Node:
    """A decision tree node with optional children and split feature."""

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """Initialize a Node with optional children and depth."""
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """Return the maximum depth below this node, including leaves."""
        if self.is_leaf:
            return self.depth
        left_depth = self.depth
        right_depth = self.depth
        if self.left_child:
            left_depth = self.left_child.max_depth_below()
        if self.right_child:
            right_depth = self.right_child.max_depth_below()
        return max(left_depth, right_depth)

    def count_nodes_below(self, only_leaves=False):
        """Calculate the number of nodes below."""
        if self.is_leaf:
            return 1

        if self.left_child:
            left = self.left_child.count_nodes_below(only_leaves)
        else:
            left = 0

        if self.right_child:
            right = self.right_child.count_nodes_below(only_leaves)
        else:
            right = 0

        if only_leaves:
            return left + right
        return 1 + left + right

    def get_leaves_below(self):
        """get leaves"""
        if self.is_leaf:
            return [self]

        leaves = []
        if self.left_child:
            leaves.extend(self.left_child.get_leaves_below())
        if self.right_child:
            leaves.extend(self.right_child.get_leaves_below())
        return leaves

    def update_bounds_below(self):
        """update bounds below"""
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -1*np.inf}

        for child in [self.left_child, self.right_child]:
            if not child:
                continue

            child.lower = self.lower.copy()
            child.upper = self.upper.copy()

            if child is self.left_child:
                child.lower[self.feature] = self.threshold

            if child is self.right_child:
                child.upper[self.feature] = self.threshold

        for child in [self.left_child, self.right_child]:
            if child:
                child.update_bounds_below()

    def update_indicator(self):
        """update indicators"""
        def is_large_enough(x):
            """Check if values are large enough."""
            return np.all(
                np.array([
                    x[:, key] >= self.lower[key]
                    for key in self.lower.keys()
                ]).T,
                axis=1,
            )

        def is_small_enough(x):
            """Check if values are small enough."""
            return np.all(
                np.array([
                    x[:, key] <= self.upper[key]
                    for key in self.upper.keys()
                ]).T,
                axis=1,
            )

        self.indicator = lambda x: np.all(
            np.array([is_large_enough(x), is_small_enough(x)]),
            axis=0,
        )

    def pred(self, x):
        """pred func"""
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


class Leaf(Node):
    """A leaf node in a decision tree containing a value."""

    def __init__(self, value, depth=None):
        """Initialize a Leaf with a value and optional depth."""
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """Return the depth of this leaf."""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """Always return 1 since this is a leaf."""
        return 1

    def __str__(self):
        """str for leaf"""
        return (f"-> leaf [value={self.value}]")

    def get_leaves_below(self):
        """get leaves"""
        return [self]

    def update_bounds_below(self):
        """update bounds below"""
        pass

    def pred(self, x):
        """pred func"""
        return self.value


class Decision_Tree:
    """Decision tree object containing the root node."""

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """Initialize a Decision_Tree with optional parameters."""
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """Return the maximum depth of the tree."""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Return the number of nodes in the tree."""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """str for decision tree"""
        return self.root.__str__()

    def get_leaves(self):
        """get leaves"""
        return self.root.get_leaves_below()

    def update_bounds(self):
        """update bounds"""
        self.root.update_bounds_below()

    def pred(self, x):
        """pred func"""
        return self.root.pred(x)

    def update_predict(self):
        """update predict function."""
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        self.predict = lambda A: np.sum(
            np.array([leaf.indicator(A) * leaf.value for leaf in leaves]),
            axis=0
        )#!/usr/bin/env python3
"""
Decision Tree and Random Forest implementation
with Node and Leaf classes.
"""

import numpy as np


class Node:
    """A decision tree node with optional children and split feature."""

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """Initialize a Node with optional children and depth."""
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """Return the maximum depth below this node, including leaves."""
        if self.is_leaf:
            return self.depth
        left_depth = self.depth
        right_depth = self.depth
        if self.left_child:
            left_depth = self.left_child.max_depth_below()
        if self.right_child:
            right_depth = self.right_child.max_depth_below()
        return max(left_depth, right_depth)

    def count_nodes_below(self, only_leaves=False):
        """Calculate the number of nodes below."""
        if self.is_leaf:
            return 1

        if self.left_child:
            left = self.left_child.count_nodes_below(only_leaves)
        else:
            left = 0

        if self.right_child:
            right = self.right_child.count_nodes_below(only_leaves)
        else:
            right = 0

        if only_leaves:
            return left + right
        return 1 + left + right

    def get_leaves_below(self):
        """get leaves"""
        if self.is_leaf:
            return [self]

        leaves = []
        if self.left_child:
            leaves.extend(self.left_child.get_leaves_below())
        if self.right_child:
            leaves.extend(self.right_child.get_leaves_below())
        return leaves

    def update_bounds_below(self):
        """update bounds below"""
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -1*np.inf}

        for child in [self.left_child, self.right_child]:
            if not child:
                continue

            child.lower = self.lower.copy()
            child.upper = self.upper.copy()

            if child is self.left_child:
                child.lower[self.feature] = self.threshold

            if child is self.right_child:
                child.upper[self.feature] = self.threshold

        for child in [self.left_child, self.right_child]:
            if child:
                child.update_bounds_below()

    def update_indicator(self):
        """update indicators"""
        def is_large_enough(x):
            """Check if values are large enough."""
            return np.all(
                np.array([
                    x[:, key] >= self.lower[key]
                    for key in self.lower.keys()
                ]).T,
                axis=1,
            )

        def is_small_enough(x):
            """Check if values are small enough."""
            return np.all(
                np.array([
                    x[:, key] <= self.upper[key]
                    for key in self.upper.keys()
                ]).T,
                axis=1,
            )

        self.indicator = lambda x: np.all(
            np.array([is_large_enough(x), is_small_enough(x)]),
            axis=0,
        )

    def pred(self, x):
        """pred func"""
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


class Leaf(Node):
    """A leaf node in a decision tree containing a value."""

    def __init__(self, value, depth=None):
        """Initialize a Leaf with a value and optional depth."""
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """Return the depth of this leaf."""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """Always return 1 since this is a leaf."""
        return 1

    def __str__(self):
        """str for leaf"""
        return (f"-> leaf [value={self.value}]")

    def get_leaves_below(self):
        """get leaves"""
        return [self]

    def update_bounds_below(self):
        """update bounds below"""
        pass

    def pred(self, x):
        """pred func"""
        return self.value


class Decision_Tree:
    """Decision tree object containing the root node."""

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """Initialize a Decision_Tree with optional parameters."""
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """Return the maximum depth of the tree."""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Return the number of nodes in the tree."""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """str for decision tree"""
        return self.root.__str__()

    def get_leaves(self):
        """get leaves"""
        return self.root.get_leaves_below()

    def update_bounds(self):
        """update bounds"""
        self.root.update_bounds_below()

    def pred(self, x):
        """pred func"""
        return self.root.pred(x)

    def update_predict(self):
        """update predict function."""
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        self.predict = lambda A: np.sum(
            np.array([leaf.indicator(A) * leaf.value for leaf in leaves]),
            axis=0
        )
