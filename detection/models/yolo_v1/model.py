import torch
import torch.nn as nn


class YoloModel(nn.Module):
    def __init__(self, input_shape, cell_count=5, cell_boxes=2, classes=43, **kwargs):
        super(YoloModel, self).__init__(**kwargs)

        # the output shape is cell_count x cell_count x (cell_boxes * 5 + classes)



    def forward(self, x):
        # todo: write forward for yolo model
        pass
