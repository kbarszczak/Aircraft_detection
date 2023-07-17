import detection.components.callbacks.callback as cs
from detection.models import model as md


class PytorchModel(md.Model):
    def __init__(self, model, loss, metrics: list, callbacks: list[cs.Callback], **kwargs):
        super(PytorchModel, self).__init__(model, loss, metrics, callbacks, **kwargs)

    def predict(self, data):
        # todo: write predict for pytorch
        pass

    def fit(self, train_data, valid_data, epochs):
        # todo: write fit for pytorch
        pass

    def evaluate(self, data):
        # todo: write evaluate for pytorch
        pass
