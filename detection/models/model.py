import detection.components.callbacks.callback as cs
from abc import abstractmethod

import detection.components.metrics.metric as m

import torch
import time


class Model:
    def __init__(self, model, loss: type(m.Metric), metrics: list[type(m.Metric)], callbacks: list[type(cs.Callback)], **kwargs):
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
    def __init__(self, model: torch.nn.Module, loss: type(m.PytorchMetric), metrics: list[type(m.PytorchMetric)], callbacks: list[type(cs.Callback)], optimizer: torch.optim.Optimizer,
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
        # set up the metrics
        self._restart_metrics()

        # create dict for the history
        history = {self.loss.name(): []} | {metric.name(): [] for metric in self.metrics} | {
            'val_' + self.loss.name(): []} | {
                      "val_" + metric.name(): [] for metric in self.metrics}

        # loop over the epochs
        for epoch in range(epochs):
            # log the current epoch
            print(f"Epoch: {epoch + 1}/{epochs}")

            # loop over the training data
            self.model.train(True)
            for step, (x, y) in enumerate(train_data):
                start = time.time()

                # move the data to the proper device
                x, y = x.to(self.device), y.to(self.device)

                # clear the gradient
                self.model.zero_grad()

                # forward pass
                y_pred = self.model(x)

                # calculate loss and gradients
                loss_value = self.loss(y, y_pred)
                loss_value.backward()

                # apply the gradient change
                self.optimizer.step()

                # calculate metrics and save their values and the loss value
                y_pred_detached = y_pred.detach()
                self.loss.append(loss_value, 'train')
                for metric in self.metrics:
                    metric.append(metric(y, y_pred_detached), 'train')

                end = time.time()

                # log the state
                PytorchModel._log_state(start, end, len(train_data), step, self.loss, self.metrics, 'train')

                # serve step callbacks
                self._serve_callbacks(cs.PerStepCallback, (self.model, self.loss, self.metrics, epoch, step, len(train_data), len(valid_data)))

            # loop over the validating data
            self.model.train(False)
            for step, (x, y) in enumerate(valid_data):
                start = time.time()

                # move the data to the proper device
                x, y = x.to(self.device), y.to(self.device)

                # do the forward pass with no grad calculation
                with torch.no_grad():
                    y_pred = self.model(x)

                # calculate and save the loss
                self.loss.append(self.loss(y, y_pred), 'valid')

                # calculate and save the metrics
                for metric in self.metrics:
                    metric.append(metric(y, y_pred), 'valid')

                end = time.time()

                # log the state
                PytorchModel._log_state(start, end, len(valid_data), step, self.loss, self.metrics, 'valid')

            # save the history
            history[self.loss.name()].extend(self.loss.list("train"))
            history[f'val_{self.loss.name()}'].extend(self.loss.list("valid"))
            for metric in self.metrics:
                history[metric.name()].extend(metric.list("train"))
                history[f'val_{metric.name()}'].extend(metric.list("valid"))

            # restart the metrics
            self._restart_metrics()

            # restart state printer
            print()

            # serve per epoch callbacks
            self._serve_callbacks(cs.PerEpochCallback, (self.model, self.loss, self.metrics, history, epoch, len(train_data), len(valid_data)))

        # serve per training callbacks
        self._serve_callbacks(cs.PerTrainingCallback, (self.model, self.loss, self.metrics, history, len(train_data), len(valid_data)))

        return history

    def evaluate(self, data) -> dict:
        # restart the metrics
        self._restart_metrics()

        # loop over batches
        self.model.train(False)
        for step, (x, y) in enumerate(data):
            start = time.time()

            # move the data to the proper device
            x, y = x.to(self.device), y.to(self.device)

            # forward pass without gradient calculation
            with torch.no_grad():
                y_pred = self.model(x)

            # calculate and save loss
            self.loss.append(self.loss(y, y_pred), 'train')

            # calculate and save metrics
            for metric in self.metrics:
                metric.append(metric(y, y_pred), 'train')

            end = time.time()

            # log the state
            PytorchModel._log_state(start, end, len(data), step, self.loss, self.metrics, 'train')

        # restart state printer
        print()

        # transform the result to the dict and return it
        return {self.loss.name(): self.loss.mean('train')} | {metric.name(): metric.mean('train') for metric in self.metrics}

    def _serve_callbacks(self, callback_type, args):
        for callback in self.callbacks:
            if issubclass(type(callback), callback_type):
                callback(args)

    def _restart_metrics(self):
        self.loss.clear()
        for metric in self.metrics:
            metric.clear()

    @staticmethod
    def _log_state(start, end, size, step, loss, metrics, mode='train'):
        time_left = (end - start) * (size - (step + 1))
        print('\r[%5d/%5d] (eta: %s)' % (
            (step + 1), size, time.strftime('%H:%M:%S', time.gmtime(time_left))), end='')
        print(f' {loss.name()}=%.4f' % loss.mean(mode), end='')
        for metric in metrics:
            print(f' {metric.name()}=%.4f' % metric.mean(mode), end='')


