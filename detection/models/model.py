import detection.components.callbacks.callback as cs
from abc import abstractmethod


class Model:
    def __init__(self, model, loss, metrics: list, callbacks: list[cs.Callback], **kwargs):
        super(Model, self).__init__(**kwargs)

        assert model is not None, "Model has to be present"
        assert loss is not None, "Loss has to be present"

        if metrics is None:
            metrics = []

        if callbacks is None:
            callbacks = []

        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.callbacks = callbacks

    @abstractmethod
    def predict(self, data):
        pass

    @abstractmethod
    def fit(self, train_data, valid_data, epochs):
        pass

    @abstractmethod
    def evaluate(self, data):
        pass
