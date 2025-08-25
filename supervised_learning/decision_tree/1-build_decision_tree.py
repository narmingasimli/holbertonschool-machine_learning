#!/usr/bin/env python3
"""Self numpy code"""
import numpy as np


class Node:

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


class Leaf(Node):

    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        return 1


class Decision_Tree():

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
