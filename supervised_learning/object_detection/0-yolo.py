#!/usr/bin/env python3
"""Object Detection"""
from tensorflow import keras as K


class Yolo:
    """Class of Yolo"""
    def __init__(self, model_path, classes_path,
                 class_t, nms_t, anchors):
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip()
                                for line in f.readlines()]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
