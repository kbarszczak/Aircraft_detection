from abc import abstractmethod
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from tqdm import tqdm
import cv2


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


class SaveNetStateCallback(PerStepCallback):
    def __init__(self, target_path, prefix: str = 'yolo', per_epoch_frequency=2, **kwargs):
        super(SaveNetStateCallback, self).__init__(**kwargs)

        assert per_epoch_frequency >= 1, "Per epoch frequency has to be at least 1"

        self.target_path = target_path
        self.per_epoch_frequency = per_epoch_frequency
        self.prefix = prefix
        self.best_loss = None
        self.frequency = None

    def __call__(self, args):
        if self.frequency is None:
            self.frequency = int(args[5] / self.per_epoch_frequency) - 1

        if args[4] <= 1 or args[4] % self.frequency != 0:
            return

        model = args[0]
        loss_fn = args[1]
        epoch = args[3]
        step = args[4]
        loss = loss_fn.mean('train')

        if self.best_loss is None or self.best_loss > loss:
            filename = os.path.join(self.target_path,
                                    (f'{self.prefix}_l=%.4f_e=%3d_s=%6d.pt' % (loss, epoch + 1, step + 1)).replace(' ',
                                                                                                                   '0'))
            torch.save(model.state_dict(), filename)
            self.best_loss = loss


class SaveHistoryPlotsCallback(PerTrainingCallback):
    def __init__(self, target_path, configs: tuple = (
            ("history.png", None, False, None),
            ("history_normalized.png", None, True, None)
    ), **kwargs):
        super(SaveHistoryPlotsCallback, self).__init__(**kwargs)
        self.target_path = target_path
        self.configs = configs

    def __call__(self, args):
        print("[Callback] Saving history plots ...")
        for config in self.configs:
            SaveHistoryPlotsCallback._save_history_plot(os.path.join(self.target_path, config[0]), args[3], config[1],
                                                        config[2], config[3])

    @staticmethod
    def _save_history_plot(path: str, history: dict, include: list[str] | None, use_norm: bool, steps: int | None,
                           fig_size: tuple[int] = (10, 5)):
        def compress_loss(loss, steps):
            return [np.average(loss[steps * i:steps * (i + 1)]) for i in range(len(loss) // steps)]

        def norm(data):
            return (data - np.min(data)) / (np.max(data) - np.min(data))

        def gcd(a, b):
            while b:
                a, b = b, a % b
            return a

        plt.clf()
        fig = plt.figure(figsize=fig_size)

        history = history.copy()
        if include is not None:
            history = {metric: values for metric, values in history.items() if metric in include}

        for metric, values in history.items():
            # skip val
            if "val" in metric:
                continue

            # make values equal size
            val_values = history[f'val_{metric}']
            if len(val_values) != len(values):
                gdc = gcd(len(val_values), len(values))
                values = compress_loss(values, len(values) // gdc)
                val_values = compress_loss(val_values, len(val_values) // gdc)

            if use_norm:
                pivot = len(values)
                data = norm(values + val_values)
                values = data[pivot:]
                val_values = data[:pivot]

            if steps is not None:
                values = compress_loss(values, steps)
                val_values = compress_loss(val_values, steps)

            history[metric] = values
            history[f'val_{metric}'] = val_values

        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        for index, (metric, values) in enumerate(history.items()):
            if "val" in metric:
                continue

            steps = range(1, len(values) + 1)
            plt.plot(steps, values, color=colors[index], linestyle='-', marker='o', label=metric)
            plt.plot(steps, history[f'val_{metric}'], color=colors[index], linestyle='--', marker='o', label=f"val_{metric}")

        plt.title("Comparision of training and validating scores")
        plt.xlabel('Steps')
        plt.ylabel("Values" if not use_norm else "Values normalized")
        plt.legend(loc='upper left')
        plt.grid(True, axis='y')
        plt.savefig(path)
        plt.close(fig)


class SaveHistoryCallback(PerTrainingCallback):
    def __init__(self, target_path, filename: str = 'history.pickle', **kwargs):
        super(SaveHistoryCallback, self).__init__(**kwargs)
        self.target_path = target_path
        self.filename = filename

    def __call__(self, args):
        print("[Callback] Saving history ...")
        with open(os.path.join(self.target_path, self.filename), 'wb') as file:
            pickle.dump(args[3], file, pickle.HIGHEST_PROTOCOL)


class SaveModelCallback(PerTrainingCallback):
    def __init__(self, target_path, model, filename: str = 'yolo.pt', **kwargs):
        super(SaveModelCallback, self).__init__(**kwargs)
        self.target_path = target_path
        self.model = model
        self.filename = filename

    def __call__(self, args):
        print("[Callback] Saving model ...")
        torch.save(self.model.state_dict(), os.path.join(self.target_path, self.filename))


class VisualizePytorchCNNLayersCallback(PerTrainingCallback):
    def __init__(self, target_path, model, device, height, width, channels, margin=3, steps=10, lr=0.1,
                 include_nested=True, **kwargs):
        super(VisualizePytorchCNNLayersCallback, self).__init__(**kwargs)

        assert margin >= 0, "Margin cannot be negative"
        assert steps > 0, "Steps has to be positive"

        self.target_path = target_path
        self.model = model
        self.device = device
        self.height = height
        self.channels = channels
        self.width = width
        self.margin = margin
        self.steps = steps
        self.lr = lr
        self.include_nested = include_nested

    def __call__(self, args):
        print("[Callback] Visualizing PyTorch CNNs ...")

        activations = {}

        def hook_fun(model, input, output):
            activations['activation'] = output

        self.model.train(False)
        queue = list(self.model.named_children())
        while queue:
            name, layer = queue.pop(0)
            if self.include_nested:
                queue.extend([(f'{name}_{n}', l) for n, l in layer.named_children()])

            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
                f_count = layer.out_channels
                rows, cols = VisualizePytorchCNNLayersCallback._rows_cols(f_count)
                result = np.zeros(
                    (rows * self.height + (rows - 1) * self.margin, cols * self.width + (cols - 1) * self.margin, 3),
                    dtype='uint8')

                for index in tqdm(range(rows * cols)):
                    i, j = index // cols, index % cols
                    filter_index = j + (i * cols)

                    hook = layer.register_forward_hook(hook_fun)
                    noise = (np.random.rand(1, self.channels, self.height, self.width) * 0.2 + 0.4).astype('float32')
                    tensor = torch.from_numpy(noise).to(self.device).requires_grad_(True)
                    optimizer = optim.NAdam([tensor], lr=self.lr)

                    for step in range(self.steps):
                        optimizer.zero_grad()
                        _ = self.model(tensor)
                        activation = activations['activation'][:, filter_index, :, :].unsqueeze(dim=1)
                        loss = torch.mean(activation)
                        loss.backward()
                        optimizer.step()

                    tensor = tensor.detach()[0].permute((1, 2, 0))
                    filter_img = VisualizePytorchCNNLayersCallback._deprocess_image(tensor.cpu().numpy())
                    result = VisualizePytorchCNNLayersCallback._append_image(result, filter_img, i, j, self.margin,
                                                                             self.height, self.width)
                    hook.remove()

                cv2.imwrite(os.path.join(self.target_path, f'{name}.png'), result)

    @staticmethod
    def _deprocess_image(img):
        img -= img.mean()
        img /= img.std() + 1e-5
        img *= 0.15

        img += 0.5
        img = np.clip(img, 0, 1)

        img *= 255
        img = np.clip(img, 0, 255).astype("uint8")

        return img

    @staticmethod
    def _append_image(image, append_image, row, col, margin, height, width):
        horizontal_start = row * height + row * margin
        horizontal_end = horizontal_start + height
        vertical_start = col * width + col * margin
        vertical_end = vertical_start + width
        image[horizontal_start: horizontal_end, vertical_start: vertical_end, :] = append_image
        return image

    @staticmethod
    def _rows_cols(value):
        assert value >= 1
        rows, cols = 1, value
        for i in range(2, value // 2 + 1):
            if value % i == 0:
                if np.abs(i - int(value / i)) < np.abs(rows - cols):
                    rows = i
                    cols = int(value / i)
        return rows, cols
