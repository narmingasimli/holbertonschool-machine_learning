#!/usr/bin/env python3
"""Object Detection"""
from tensorflow import keras as K
import numpy as np
import os
import cv2


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

    def process_outputs(self, outputs, image_size):
        """Process Outputs"""
        boxes = []
        box_confidences = []
        box_class_probs = []
        image_h, image_w = image_size

        for i, output in enumerate(outputs):
            grid_h, grid_w, anchor_boxes = output.shape[:3]

            tx = output[..., 0]
            ty = output[..., 1]
            tw = output[..., 2]
            th = output[..., 3]

            box_confidence = 1 / (1 + np.exp(-output[..., 4]))
            box_class_prob = 1 / (1 + np.exp(-output[..., 5:]))

            cx = np.arange(grid_w).reshape(1, grid_w, 1)
            cy = np.arange(grid_h).reshape(grid_h, 1, 1)

            bx = (1 / (1 + np.exp(-tx)) + cx) / grid_w
            by = (1 / (1 + np.exp(-ty)) + cy) / grid_h

            bw = ((self.anchors[i, :, 0] * np.exp(tw)) /
                  self.model.input.shape[1])
            bh = ((self.anchors[i, :, 1] * np.exp(th)) /
                  self.model.input.shape[2])

            x1 = (bx - (bw / 2)) * image_w
            y1 = (by - (bh / 2)) * image_h
            x2 = (bx + (bw / 2)) * image_w
            y2 = (by + (bh / 2)) * image_h

            boxes.append(np.stack([x1, y1, x2, y2], axis=-1))
            box_confidences.append(box_confidence[..., np.newaxis])
            box_class_probs.append(box_class_prob)

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filtered Boxes"""
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for i in range(len(boxes)):
            box_scores_i = box_confidences[i] * box_class_probs[i]

            box_classes_i = np.argmax(box_scores_i, axis=-1)
            box_classes_scores_i = np.max(box_scores_i, axis=-1)

            filtering_mask = box_classes_scores_i >= self.class_t

            filtered_boxes.append(boxes[i][filtering_mask])
            box_classes.append(box_classes_i[filtering_mask])
            box_scores.append(box_classes_scores_i[filtering_mask])

        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """Non-Max Suppression (NMS)"""
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        def iou(box, boxes):
            """IOU"""
            x1, y1, x2, y2 = box
            x1s, y1s, x2s, y2s = (boxes[:, 0], boxes[:, 1],
                                  boxes[:, 2], boxes[:, 3])

            inter_x1 = np.maximum(x1, x1s)
            inter_y1 = np.maximum(y1, y1s)
            inter_x2 = np.minimum(x2, x2s)
            inter_y2 = np.minimum(y2, y2s)
            inter_area = (np.maximum(0, inter_x2 - inter_x1) *
                          np.maximum(0, inter_y2 - inter_y1))

            box_area = (x2 - x1) * (y2 - y1)
            boxes_area = (x2s - x1s) * (y2s - y1s)

            union_area = box_area + boxes_area - inter_area
            iou = inter_area / union_area

            return iou

        unique_classes = np.unique(box_classes)

        for cls in unique_classes:
            cls_mask = box_classes == cls
            cls_boxes = filtered_boxes[cls_mask]
            cls_scores = box_scores[cls_mask]

            sorted_idx = np.argsort(cls_scores)[::-1]
            cls_boxes = cls_boxes[sorted_idx]
            cls_scores = cls_scores[sorted_idx]

            while len(cls_boxes) > 0:
                box_predictions.append(cls_boxes[0])
                predicted_box_scores.append(cls_scores[0])
                predicted_box_classes.append(cls)

                if len(cls_boxes) == 1:
                    break

                ious = iou(cls_boxes[0], cls_boxes[1:])

                mask = ious < self.nms_t
                cls_boxes = cls_boxes[1:][mask]
                cls_scores = cls_scores[1:][mask]

        box_predictions = np.array(box_predictions)
        predicted_box_classes = np.array(predicted_box_classes)
        predicted_box_scores = np.array(predicted_box_scores)

        return box_predictions, predicted_box_classes, predicted_box_scores

    @staticmethod
    def load_images(folder_path):
        """Folder Path"""
        images = []
        image_paths = []

        for filename in os.listdir(folder_path):
            path = os.path.join(folder_path, filename)

            if path.lower().endswith(('.jpg', '.jpeg', '.png')):
                image = cv2.imread(path)
                if image is not None:
                    images.append(image)
                    image_paths.append(path)

        return images, image_paths

    def preprocess_images(self, images):
        """Preprocess Images"""
        image_shapes = []

        input_h = self.model.input.shape.as_list()[1]
        input_w = self.model.input.shape.as_list()[2]

        pimages = []

        for img in images:
            h, w = img.shape[:2]
            image_shapes.append((h, w))

            resized = cv2.resize(img, (input_w, input_h),
                                 interpolation=cv2.INTER_CUBIC)
            normalized = resized.astype('float32') / 255.0

            pimages.append(normalized)

        pimages = np.array(pimages)
        image_shapes = np.array(image_shapes)

        return pimages, image_shapes

    def show_boxes(self, image, boxes, box_classes,
                   box_scores, file_name):
        """Show Boxes"""
        image2 = image.copy()
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            cls = box_classes[i]
            score = box_scores[i]

            class_name = self.class_names[cls]

            label = "{} {:.2f}".format(class_name, score)

            cv2.rectangle(image2, (x1,y1), (x2,y2), (255,0,0), 2)
            cv2.putText(image2, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0,0,255),
                        cv2.LINE_AA)

        cv2.imshow(file_name, image2)
        key = cv2.waitKey(0)

        if key == ord('s'):
            if not os.path.exists('detections'):
                os.makedirs('detections')

            path = os.path.join('detections', os.path.basename(file_name))
            cv2.imwrite(path, image2)
            print(f"Saved path: {path}")

        cv2.destroyAllWindows()
