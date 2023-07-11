import callbacks as cs
import model as md


class PytorchModel(md.Model):
    def __init__(self, model, loss, metrics: list, callbacks: list[cs.Callback], **kwargs):
        super(PytorchModel, self).__init__(model, loss, metrics, callbacks, **kwargs)

    def predict(self, data):
        pass

    def fit(self, data):
        pass

    def evaluate(self, data):
        pass
