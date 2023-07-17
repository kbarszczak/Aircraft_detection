from abc import abstractmethod
import numpy as np
import os
import torch


class Callback:
    def __init__(self, **kwargs):
        super(Callback, self).__init__(**kwargs)

    @abstractmethod
    def __call__(self, args):
        pass


class PerEpochCallback(Callback):
    def __init__(self, **kwargs):
        super(PerEpochCallback, self).__init__(**kwargs)

    @abstractmethod
    def __call__(self, args):
        pass


class PerStepCallback(Callback):
    def __init__(self, **kwargs):
        super(PerStepCallback, self).__init__(**kwargs)

    @abstractmethod
    def __call__(self, args):
        pass


class PerTrainingCallback(Callback):
    def __init__(self, **kwargs):
        super(PerTrainingCallback, self).__init__(**kwargs)

    @abstractmethod
    def __call__(self, args):
        pass


class SaveStateCallback(PerStepCallback):
    def __init__(self, target_path, frequency=500, **kwargs):
        super(SaveStateCallback, self).__init__(**kwargs)
        self.target_path = target_path
        self.frequency = frequency
        self.best_loss = None

    def __call__(self, args):
        if args[1] % self.frequency != 0:
            return

        model = args[0]
        loss_metrics = args[1]
        loss_fun = args[2]
        epoch = args[4]
        step = args[5]
        loss = np.mean(loss_metrics[loss_fun.__name__])

        if self.best_loss is None or self.best_loss > loss:
            filename = os.path.join(self.target_path, f'yolo_l={loss}_e={epoch + 1}_s={step + 1}.pt')
            torch.save(model.state_dict(), filename)
            self.best_loss = loss
