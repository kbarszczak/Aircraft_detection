import detection.providers.provider as provider
import detection.models.model as model


class YoloProvider(provider.Provider):
    def __init__(self, **kwargs):
        super(YoloProvider, self).__init__(**kwargs)

    def get_model(self, device: str, target_path: str) -> model.Model:
        # todo: write get_model for yolo provider v1
        pass

    def get_data(self, data_path: str, train_file: str, test_file: str, valid_file: str, batch_size: int, width: int,
                 height: int) -> (object, object, object):
        # todo: write get_data for yolo provider v1
        pass
