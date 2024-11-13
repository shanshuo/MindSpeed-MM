import torch


class Sigmoid:
    def __call__(self, x):
        # swish
        return x * torch.sigmoid(x)
