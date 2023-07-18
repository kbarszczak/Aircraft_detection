import torch

import detection.models.provider as provider
import detection.models.model as m
import detection.models.yolo_v1.model as yolo
import detection.models.yolo_v1.dataset as dataset
import detection.components.metrics.metric as metric


class YoloProvider(provider.Provider):
    def __init__(self, **kwargs):
        super(YoloProvider, self).__init__(**kwargs)

    def get_model(self, device: str, target_path: str) -> type(m.Model):
        if device == 'gpu' or device == 'cuda:0' or device == 'cpu':
            device = torch.device(device)
        else:
            raise ValueError(f"Unsupported device type '{device}' was passed. Supported: [cpu, gpu, cuda:0]")

        model = yolo.YoloModel()
        pytorch_model = m.PytorchModel(
            model=model,
            loss=metric.PytorchYoloV1Loss(),
            metrics=[],
            callbacks=[],
            optimizer=torch.optim.NAdam(model.parameters(), lr=1e-4),
            device=device
        )

        return pytorch_model

    def get_data(self, data_path: str, train_file: str, test_file: str, valid_file: str, batch_size: int, width: int,
                 height: int) -> (object, object, object):
        train = torch.utils.data.DataLoader(
            dataset=dataset.YoloDataset(
                path=data_path,
                subdir='dataset',
                filename=train_file,
                labels='labels.csv',
                shape=(height, width, 3)
            ),
            shuffle=True,
            batch_size=batch_size,
            drop_last=True,
            prefetch_factor=10,
            num_workers=2
        )

        test = torch.utils.data.DataLoader(
            dataset=dataset.YoloDataset(
                path=data_path,
                subdir='dataset',
                filename=test_file,
                labels='labels.csv',
                shape=(height, width, 3)
            ),
            shuffle=False,
            batch_size=batch_size,
            drop_last=True,
            prefetch_factor=10,
            num_workers=2
        )

        valid = torch.utils.data.DataLoader(
            dataset=dataset.YoloDataset(
                path=data_path,
                subdir='dataset',
                filename=valid_file,
                labels='labels.csv',
                shape=(height, width, 3)
            ),
            shuffle=False,
            batch_size=batch_size,
            drop_last=True,
            prefetch_factor=10,
            num_workers=2
        )

        return train, test, valid
