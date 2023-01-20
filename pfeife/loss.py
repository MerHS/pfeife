import torch.nn as nn


class LossWrapper(nn.Module):
    def forward(self, input, target):
        return input.sum()


class SumLoss(LossWrapper):
    def forward(self, input, target):
        return input.sum()
