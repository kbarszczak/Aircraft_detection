import detection.models.model as model
from abc import abstractmethod


class Provider:
    def __init__(self, **kwargs):
        super(Provider, self).__init__(**kwargs)

    @abstractmethod
    def get_model(self, device: str, target_path: str, input_shape: tuple) -> type(model.Model):
        pass

    @abstractmethod
    def get_data(self, data_path: str, train_file: str, test_file: str, valid_file: str, batch_size: int,
                 input_shape: tuple) -> (object, object, object):
        pass
