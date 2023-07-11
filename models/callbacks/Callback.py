from abc import abstractmethod


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
