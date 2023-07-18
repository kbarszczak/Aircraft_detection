from abc import abstractmethod
import numpy as np
import torch
import torch.nn as nn

import detection.utils.pytorch_utils as utils


class Metric:
    def __init__(self, **kwargs):
        super(Metric, self).__init__(**kwargs)
        self.history = []

    def mean(self):
        return np.mean(self.history)

    def numpy(self):
        return np.array(self.history)

    @abstractmethod
    def append(self, value: float):
        pass

    @abstractmethod
    def __call__(self, y_true, y_pred):
        pass


class PytorchMetric(Metric):
    def __init__(self, **kwargs):
        super(PytorchMetric, self).__init__(**kwargs)

    def append(self, value: torch.Tensor):
        self.history.append(value.item())

    @abstractmethod
    def __call__(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        pass


class PytorchYoloV1Loss(Metric):
    def __init__(self, cell_count=5, cell_boxes=1, classes=43, **kwargs):
        super(PytorchYoloV1Loss, self).__init__(**kwargs)

        assert cell_boxes == 2, f"Cell boxes for value {cell_boxes} is not implemented"

        self.cell_count = cell_count
        self.cell_boxes = cell_boxes
        self.classes = classes
        self.mse = nn.MSELoss(reduction='sum')
        self.lambda_noobj = 0.5
        self.lambda_coord = 5.0

    @abstractmethod
    def __call__(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        ious = [utils.intersection_over_union(y_true[..., self.classes + 1 + b * 5:self.classes + (b + 1) * 5],  y_pred[..., self.classes + 1:self.classes + 5]) for b in range(self.cell_boxes)]
        ious = torch.cat([iou.unsqueeze(0) for iou in ious], dim=0)

        iou_max_val, best_bbox = torch.max(ious, dim=0)
        actual_box = y_true[..., self.classes].unsqueeze(3)
        box_pred = actual_box * (
            (
                    best_bbox * y_pred[..., self.classes + 6:self.classes + 10]
                    + (1 - best_bbox) * y_pred[..., self.classes + 1:self.classes + 5]
            )
        )

        box_pred[..., 2:4] = torch.sign(box_pred[..., 2:4]) * (torch.sqrt(torch.abs(box_pred[..., 2:4] + 1e-6)))
        box_target = actual_box * y_true[..., self.classes+1:self.classes+5]
        box_target[..., 2:4] = torch.sqrt(box_target[..., 2:4])

        box_coord_loss = self.mse(
            torch.flatten(box_pred, end_dim=-2),
            torch.flatten(box_target, end_dim=-2)
        )

        # object loss
        pred_box = (best_bbox * y_pred[..., self.classes+5:self.classes+6] + (1 - best_bbox) * y_pred[..., self.classes:self.classes+1])

        obj_loss = self.mse(
            torch.flatten(actual_box * pred_box),
            torch.flatten(actual_box * y_true[..., self.classes:self.classes+1])
        )

        # no object loss
        no_obj_loss = self.mse(
            torch.flatten((1 - actual_box) * y_pred[..., self.classes:self.classes+1], start_dim=1),
            torch.flatten((1 - actual_box) * y_true[..., self.classes:self.classes+1], start_dim=1)
        )

        no_obj_loss += self.mse(
            torch.flatten((1 - actual_box) * y_pred[..., self.classes+5:self.classes+6], start_dim=1),
            torch.flatten((1 - actual_box) * y_true[..., self.classes:self.classes+1], start_dim=1)
        )

        # class loss
        class_loss = self.mse(
            torch.flatten(actual_box * y_pred[..., :self.classes], end_dim=-2),
            torch.flatten(actual_box * y_true[..., :self.classes], end_dim=-2)
        )

        return self.lambda_coord * box_coord_loss + obj_loss + self.lambda_noobj * no_obj_loss + class_loss
