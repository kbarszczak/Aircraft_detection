import detection.components.callbacks.callback as cs
from detection.models import model as md


class TensorFlowModel(md.Model):
    def __init__(self, model, loss, metrics: list, callbacks: list[cs.Callback], **kwargs):
        super(TensorFlowModel, self).__init__(model, loss, metrics, callbacks, **kwargs)

    def predict(self, data):
        # todo: write predict for tensorflow
        pass

    def fit(self, train_data, valid_data, epochs):
        # todo: write fit for tensorflow
        pass

    def evaluate(self, data):
        # todo: write evaluate for tensorflow
        pass
