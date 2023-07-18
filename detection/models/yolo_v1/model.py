import torch
import torch.nn as nn

import detection.components.modules.cnn_block as cb
import detection.components.modules.linear_block as lb


class YoloModel(nn.Module):
    def __init__(self, filters: tuple[int, int, int, int, int] = (32, 64, 128, 256, 512), cell_count=5, cell_boxes=2,
                 classes=43, **kwargs):
        super(YoloModel, self).__init__(**kwargs)

        assert len(filters) == 5, 'There should be 5 cnn blocks. Please provide filters list of length 5'
        assert cell_boxes == 2, f'Cell boxes is not implemented for value {cell_boxes}'

        self.cell_count = cell_count
        self.cell_boxes = cell_boxes
        self.classes = classes

        self.cnn_1 = cb.Conv2dBlock(3, filters[0], 3, 1, 1)
        self.cnn_2 = cb.Conv2dBlock(filters[0], filters[1], 3, 1, 1)
        self.cnn_3 = cb.Conv2dBlock(filters[1], filters[2], 3, 1, 1)
        self.cnn_4 = cb.Conv2dBlock(filters[2], filters[3], 3, 1, 1)
        self.cnn_5 = cb.Conv2dBlock(filters[3], filters[4], 3, 1, 1)

        self.pool_1 = nn.MaxPool2d((2, 2))
        self.pool_2 = nn.MaxPool2d((2, 2))
        self.pool_3 = nn.MaxPool2d((2, 2))
        self.pool_4 = nn.MaxPool2d((2, 2))
        self.pool_5 = nn.MaxPool2d((2, 2))

        self.flatten = nn.Flatten()
        self.lin_1 = lb.LinearBlock(cell_count * cell_count * filters[4], filters[4])
        self.lin_2 = nn.Linear(filters[4], cell_count * cell_count * (cell_boxes * 5 + classes))

    def forward(self, x):
        x = self.pool_1(self.cnn_1(x))
        x = self.pool_2(self.cnn_2(x))
        x = self.pool_3(self.cnn_3(x))
        x = self.pool_4(self.cnn_4(x))
        x = self.pool_5(self.cnn_5(x))
        x = self.flatten(x)
        x = self.lin_1(x)
        x = self.lin_2(x)
        return torch.reshape(x, (-1, self.cell_count, self.cell_count, self.classes + self.cell_boxes*5))
