import torch
import numpy as np
import pandas as pd
import os


class YoloDataset(torch.utils.data.Dataset):
    def __init__(self, path: str, subdir: str, filename: str, labels: str, shape: tuple[int, int, int] = (160, 160, 3),
                 cell_count=5, cell_boxes=1,
                 classes=43, **kwargs):
        super(YoloDataset, self).__init__(**kwargs)

        assert cell_boxes == 1, f"Cell boxes for value {cell_boxes} is not implemented"

        self.path = os.path.join(path, subdir)
        self.shape = shape
        self.cell_count = cell_count
        self.cell_boxes = cell_boxes
        self.classes = classes
        self.ids = pd.read_csv(os.path.join(path, filename), names=["ids"])
        self.labels = pd.read_csv(os.path.join(path, labels))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        filename = str(self.ids.iloc[idx, 0])
        image_filename = f'{filename}.bin'
        image_b_boxes = f'{filename}.csv'

        image = self._load_process_image(os.path.join(self.path, image_filename))
        b_boxes = self._load_process_b_boxes(os.path.join(self.path, image_b_boxes))

        return image, b_boxes

    def _load_process_image(self, path):
        with open(path, 'rb') as bf:
            return torch.from_numpy(np.reshape(np.frombuffer(bf.read(), dtype='float32'), (self.shape[2], self.shape[0], self.shape[1])).copy())

    def _load_process_b_boxes(self, path):
        boxes = torch.zeros((self.cell_count, self.cell_count, (self.cell_boxes * 5 + self.classes)),
                            dtype=torch.float32)
        for _, row in pd.read_csv(path).iterrows():
            class_index = row['class_index']
            width, height, xcenter, ycenter = row['width'], row['height'], row['xcenter'], row['ycenter']

            x, y = int(xcenter * self.cell_count), int(ycenter * self.cell_count)
            xcell, ycell = xcenter * self.cell_count - x, ycenter * self.cell_count - y
            widthcell, heightcell = width * self.cell_count, height * self.cell_count

            if boxes[y, x, self.classes] == 0:
                boxes[y, x, self.classes] = 1.0
                boxes[y, x, class_index] = 1.0
                boxes[y, x, self.classes + 1:self.classes + 5] = torch.tensor([xcell, ycell, widthcell, heightcell],
                                                                              dtype=torch.float32)

        return boxes
