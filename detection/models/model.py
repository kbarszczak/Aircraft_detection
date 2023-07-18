import detection.components.callbacks.callback as cs
from abc import abstractmethod

import detection.components.metrics.metric as m

import numpy as np
import torch
import time


class Model:
    def __init__(self, model, loss: m.Metric, metrics: list[m.Metric], callbacks: list[cs.Callback], **kwargs):
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


class PytorchModel(Model):
    def __init__(self, model: torch.nn.Module, loss: m.PytorchMetric, metrics: list[m.PytorchMetric], callbacks: list[cs.Callback], optimizer: torch.optim.Optimizer,
                 device: torch.device, **kwargs):
        super(PytorchModel, self).__init__(model, loss, metrics, callbacks, **kwargs)

        assert optimizer is not None, "Optimizer has to be set"
        assert device is not None, "Device has to be set"

        self.optimizer = optimizer
        self.device = device

    def predict(self, data) -> torch.tensor:
        result = []
        self.model.train(False)

        # loop over batches
        for x, _ in data:
            # move the x to the proper device
            x = x.to(self.device)

            # do the forward pass
            with torch.no_grad():
                y = self.model(x)

            # remember the result
            result.append(y)

        # return the tensor
        return torch.tensor(result)

    def fit(self, train_data, valid_data, epochs) -> dict:
        # create dict for the history
        history = {self.loss.__name__: []} | {metric.__name__: [] for metric in self.metrics} | {
            'val_' + self.loss.__name__: []} | {
                      "val_" + metric.__name__: [] for metric in self.metrics}

        # loop over the epochs
        for epoch in range(epochs):
            print(f"Epoch: {epoch + 1}/{epochs}")

            # create empty dict for loss and metrics
            loss_metrics = {self.loss.__name__: []} | {metric.__name__: [] for metric in self.metrics}

            # loop over the training data
            self.model.train(True)
            for step, (x, y) in enumerate(train_data):
                start = time.time()

                # move the data to the proper device, clear gradient, and forward pass
                x, y = x.to(self.device), y.to(self.device)
                self.model.zero_grad()
                y_pred = self.model(x)

                # calculate loss, metrics and apply the gradient
                loss_value = self.loss(y, y_pred)
                loss_value.backward()
                self.optimizer.step()
                y_pred_detached = y_pred.detach()
                metrics_values = [metric(y, y_pred_detached) for metric in self.metrics]

                # save the loss and metrics
                loss_metrics[self.loss.__name__].append(loss_value.item())
                for metric, value in zip(self.metrics, metrics_values):
                    loss_metrics[metric.__name__].append(value.item())

                end = time.time()

                # log the state & serve step callbacks
                PytorchModel._log_state(start, end, len(train_data), step, loss_metrics)
                self._serve_callbacks(cs.PerStepCallback, [self.model, self.loss, self.metrics, loss_metrics, epoch, step, len(train_data), len(valid_data)])

            # save the training history
            for metric, values in loss_metrics.items():
                history[metric].extend(values)

            # setup dict for validation loss and metrics
            loss_metrics = {self.loss.__name__: []} | {metric.__name__: [] for metric in self.metrics}

            # loop over validating data
            self.model.train(False)
            for step, (x, y) in enumerate(valid_data):
                start = time.time()

                # move the data to the proper device & forward pass
                x, y = x.to(self.device), y.to(self.device)
                with torch.no_grad():
                    y_pred = self.model(x)

                # save the loss and metrics
                loss_metrics[self.loss.__name__].append(self.loss(y, y_pred).item())
                for metric, value in zip(self.metrics, [metric(y, y_pred) for metric in self.metrics]):
                    loss_metrics[metric.__name__].append(value.item())

                end = time.time()

                # log the state
                PytorchModel._log_state(start, end, len(valid_data), step, loss_metrics)

            # save the validation history
            for metric, values in loss_metrics.items():
                history[f"val_{metric}"].extend(values)

            # restart state printer
            print()

            # serve per epoch callbacks
            self._serve_callbacks(cs.PerEpochCallback, [self.model, self.loss, self.metrics, history, epoch, len(train_data), len(valid_data)])

        # serve per training callbacks
        self._serve_callbacks(cs.PerTrainingCallback, [self.model, self.loss, self.metrics, history, len(train_data), len(valid_data)])

        return history

    def evaluate(self, data) -> dict:
        # create empty dict for loss and metrics
        loss_metrics = {self.loss.__name__: 0} | {metric.__name__: 0 for metric in self.metrics}

        # loop over batches
        self.model.train(False)
        for step, (x, y) in enumerate(data):
            start = time.time()

            # move the data to the proper device
            x, y = x.to(self.device), y.to(self.device)

            # forward pass
            with torch.no_grad():
                y_pred = self.model(x)

            # calculate loss & metrics
            loss_metrics[self.loss.__name__] += self.loss(y, y_pred).item()
            for metric in self.metrics:
                loss_metrics[metric.__name__] += metric(y, y_pred).item()

            end = time.time()

            # log the state
            PytorchModel._log_state(start, end, len(data), step, loss_metrics)

        # restart state printer
        print()

        return {metric: (value / len(data)) for metric, value in loss_metrics.items()}

    def _serve_callbacks(self, callback_type, args):
        for callback in self.callbacks:
            if issubclass(type(callback), callback_type):
                callback(args)

    @staticmethod
    def _log_state(start, end, size, step, loss_metrics):
        time_left = (end - start) * (size - (step + 1))
        print('\r[%5d/%5d] (eta: %s)' % (
            (step + 1), size, time.strftime('%H:%M:%S', time.gmtime(time_left))), end='')
        for metric, values in loss_metrics.items():
            print(f' {metric}=%.4f' % (np.mean(values)), end='')

