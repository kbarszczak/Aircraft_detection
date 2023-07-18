import torch
import numpy as np
import pandas as pd
import os
import cv2


class YoloDataset(torch.utils.data.Dataset):
    def __init__(self, path: str, subdir: str, filename: str, labels: str, shape: tuple[int, int, int] = (160, 160, 3),
                 cell_count=5, cell_boxes=2,
                 classes=43, **kwargs):
        super(YoloDataset, self).__init__(**kwargs)

        assert cell_boxes == 2, f"Cell boxes for value {cell_boxes} is not implemented"

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
        image_filename = f'{filename}.jpg'
        image_b_boxes = f'{filename}.csv'

        image = self._load_process_image(os.path.join(self.path, image_filename))
        b_boxes = self._load_process_b_boxes(os.path.join(self.path, image_b_boxes))

        return image, b_boxes

    def _load_process_image(self, path):
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.shape[1], self.shape[0]), interpolation=cv2.INTER_LINEAR)
        return torch.from_numpy(np.transpose((img / 255.0).astype('float32'), (2, 0, 1)))

    def _load_process_b_boxes(self, path):
        boxes = torch.zeros((self.cell_count, self.cell_count, (self.cell_boxes * 5 + self.classes)),
                            dtype=torch.float32)
        for _, row in pd.read_csv(path).iterrows():
            class_index = self.labels.loc[self.labels['classes'] == row['class'], 'indices'].values[0]

            width, height, xmin, xmax, ymin, ymax = row['width'], row['height'], row['xmin'], row['xmax'], row['ymin'], \
                                                    row['ymax']

            width_coef = width / self.shape[1]
            height_coef = height / self.shape[0]

            xmin /= width_coef
            xmax /= width_coef

            ymin /= height_coef
            ymax /= height_coef

            x_center = ((xmin + xmax) / 2.0) / self.shape[1]
            y_center = ((ymin + ymax) / 2.0) / self.shape[0]
            width = (xmax - xmin) / self.shape[1]
            height = (ymax - ymin) / self.shape[0]

            x, y = int(x_center * self.cell_count), int(y_center * self.cell_count)
            x_cell, y_cell = x_center * self.cell_count - x, y_center * self.cell_count - y
            width_cell, height_cell = width * self.cell_count, height * self.cell_count

        return boxes


d = torch.utils.data.DataLoader(YoloDataset('D:/Data/Aircraft_Detection', 'dataset', 'train.txt', "labels.csv",
                                            (5 * 2 * 2 * 2 * 2 * 2, 5 * 2 * 2 * 2 * 2 * 2, 3)), batch_size=10,
                                shuffle=True)
b = next(iter(d))
print(b[1][0])
print(b[1].shape)
