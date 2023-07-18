import torch.nn as nn


class LinearBlock(nn.Module):
    def __init__(self, in_features, out_features, alpha=0.1, **kwargs):
        super(LinearBlock, self).__init__(**kwargs)

        self.lin = nn.Linear(in_features, out_features)
        self.act = nn.LeakyReLU(alpha)

    def forward(self, x):
        return self.act(self.lin(x))
