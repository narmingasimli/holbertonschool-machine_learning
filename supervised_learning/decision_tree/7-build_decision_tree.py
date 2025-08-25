#!/usr/bin/env python3
"""Decision tree: nodes, leaves, and tree structure."""
import numpy as np


class Node:
    """A node class containing leaves,roots."""

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        if self.is_leaf:
            return self.depth
        left = self.left_child.max_depth_below() \
            if self.left_child else self.depth
        right = self.right_child.max_depth_below() \
            if self.right_child else self.depth
        return max(left, right)

    def count_nodes_below(self, only_leaves=False):
        if self.is_leaf:
            return 1

        if only_leaves:
            if self.left_child:
                left = self.left_child.count_nodes_below(True)
            else:
                left = 0
            if self.right_child:
                right = self.right_child.count_nodes_below(True)
            else:
                right = 0
            return left + right
        else:
            if self.left_child:
                left = self.left_child.count_nodes_below(False)
            else:
                left = 0
            if self.right_child:
                right = self.right_child.count_nodes_below(False)
            else:
                right = 0
            return 1 + left + right

    def get_leaves_below(self):
        if self.is_leaf:
            return [self]

        leaves = []
        if self.left_child:
            leaves.extend(self.left_child.get_leaves_below())
        if self.right_child:
            leaves.extend(self.right_child.get_leaves_below())
        return leaves

    def update_bounds_below(self):
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -1 * np.inf}

        for child in [self.left_child, self.right_child]:
            if child is not None:
                child.upper = self.upper.copy()
                child.lower = self.lower.copy()

                if child == self.left_child:
                    child.lower[self.feature] = self.threshold
                else:
                    child.upper[self.feature] = self.threshold

        for child in [self.left_child, self.right_child]:
            if child is not None:
                child.update_bounds_below()

    def left_child_add_prefix(self, text):
        lines = text.split("\n")
        new_text = "    +---> " + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("    |  " + x) + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        lines = text.split("\n")
        new_text = "    +---> " + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("       " + x) + "\n"
        return new_text

    def __str__(self):
        if self.is_root:
            label = (f"root [feature={self.feature}"
                     f", threshold={self.threshold}]")
        else:
            label = (f"node [feature={self.feature}"
                     f", threshold={self.threshold}]")

        result = label
        if self.left_child:
            result += "\n" + self.left_child_add_prefix(str(self.left_child))\
                    .rstrip("\n")
        if self.right_child:
            result += "\n" +\
                    self.right_child_add_prefix(str(self.right_child))\
                    .rstrip("\n")
        return result

    def update_indicator(self):
        def is_large_enough(x):
            if not self.lower:
                return np.ones(x.shape[0], dtype=bool)

            conditions = []
            for key in self.lower.keys():
                if key < x.shape[1]:
                    conditions.append(np.greater(x[:, key], self.lower[key]))
                else:
                    conditions.append(np.ones(x.shape[0], dtype=bool))
            if conditions:
                return np.all(np.array(conditions), axis=0)
            else:
                return np.ones(x.shape[0], dtype=bool)

        def is_small_enough(x):
            if not self.upper:
                return np.ones(x.shape[0], dtype=bool)

            conditions = []
            for key in self.upper.keys():
                if key < x.shape[1]:
                    conditions.append(np.less_equal(x[:, key],
                                                    self.upper[key]))
                else:
                    conditions.append(np.ones(x.shape[0], dtype=bool))

            if conditions:
                return np.all(np.array(conditions), axis=0)
            else:
                return np.ones(x.shape[0], dtype=bool)

        self.indicator = lambda x: np.all(np.array([is_large_enough(x),
                                                   is_small_enough(x)]),
                                          axis=0)

    def pred(self, x):
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


class Leaf(Node):
    """Leaf class."""

    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        return 1

    def get_leaves_below(self):
        return [self]

    def update_bounds_below(self):
        pass

    def __str__(self):
        return f"-> leaf [value={self.value}]"

    def pred(self, x):
        return self.value


class Decision_Tree():
    """Decision Tree class."""

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
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
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def get_leaves(self):
        return self.root.get_leaves_below()

    def update_
