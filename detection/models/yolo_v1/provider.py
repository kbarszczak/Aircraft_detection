import torch

import detection.models.provider as provider
import detection.models.model as m
import detection.models.yolo_v1.model as yolo
import detection.models.yolo_v1.dataset as dataset
import detection.components.metrics.metric as metric
import detection.components.callbacks.callback as callback


class YoloProvider(provider.Provider):
    def __init__(self, **kwargs):
        super(YoloProvider, self).__init__(**kwargs)

    def get_model(self, device: str, target_path: str, input_shape: tuple) -> type(m.Model):
        if device == 'cuda' or device == 'cpu':
            device = torch.device(device)
        else:
            raise ValueError(f"Unsupported device type '{device}' was passed. Supported: [cpu, cuda]")

        model = yolo.YoloModel().to(device)
        pytorch_model = m.PytorchModel(
            model=model,
            loss=metric.PytorchYoloV1Loss(),
            metrics=[
            ],
            callbacks=[
                callback.SaveModelCallback(
                    target_path=target_path,
                    model=model,
                    filename='yolo_v1.pt'
                ),
                callback.SaveHistoryCallback(
                    target_path=target_path,
                    filename='history.pickle'
                ),
                callback.SaveNetStateCallback(
                    target_path=target_path,
                    prefix='yolo',
                    per_epoch_frequency=1
                ),
                callback.SaveHistoryPlotsCallback(
                    target_path=target_path,
                    configs=(
                        ("history.png", None, False, None),
                    )
                ),
                callback.VisualizePytorchCNNLayersCallback(
                    target_path=target_path,
                    model=model,
                    device=device,
                    height=input_shape[0],
                    width=input_shape[1],
                    channels=input_shape[2],
                    margin=3,
                    steps=10,
                    lr=0.1,
                    include_nested=True
                )
            ],
            optimizer=torch.optim.NAdam(model.parameters(), lr=1e-4),
            device=device
        )

        return pytorch_model

    def get_data(self, data_path: str, train_file: str, test_file: str, valid_file: str, batch_size: int,
                 input_shape: tuple) -> (object, object, object):
        train = torch.utils.data.DataLoader(
            dataset=dataset.YoloDataset(
                path=data_path,
                subdir='dataset_processed',
                filename=train_file,
                labels='labels.csv',
                shape=input_shape
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
                subdir='dataset_processed',
                filename=test_file,
                labels='labels.csv',
                shape=input_shape
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
                subdir='dataset_processed',
                filename=valid_file,
                labels='labels.csv',
                shape=input_shape
            ),
            shuffle=False,
            batch_size=batch_size,
            drop_last=True,
            prefetch_factor=10,
            num_workers=2
        )

        return train, test, valid
