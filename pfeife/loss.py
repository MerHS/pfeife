import torch.nn as nn


class LossWrapper(nn.Module):
    def forward(self, input, target, model=None):
        return input.sum()


class SumLoss(LossWrapper):
    def forward(self, input, target, model=None):
        if isinstance(input, tuple) or isinstance(input, list):
            input = input[0]
        return input.sum()


class MaskedLMOutputLoss(LossWrapper):
    def forward(self, input, target, model=None):
        if input.loss is not None:
            return input.loss
        else:
            return input.logits.sum()


class RescaleLoss(LossWrapper):
    def __init__(self, loss_fn, scale=1.0):
        super().__init__()
        self.loss_fn = loss_fn
        self.scale = scale

    def forward(self, input, target, model=None):
        return self.loss_fn(input, target, model) * self.scale


class NormalizeLoss(LossWrapper):
    def __init__(self, loss_fn):
        super().__init__()
        self.loss_fn = loss_fn

    def forward(self, input, target, model=None):
        loss = self.loss_fn(input, target, model)
        if loss.item() == 0:
            return loss

        while abs(loss.item()) < 1:
            loss = loss * 2

        while abs(loss.item()) > 1:
            loss = loss / 2

        return loss


class CrossEntropyLoss(LossWrapper):
    def __init__(self, num_labels=10):
        super().__init__()
        self.num_labels = num_labels
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input, target, model=None):
        return self.loss_fn(input.view(-1, self.num_labels), target)


class HFCrossEntropyLoss(LossWrapper):
    def __init__(self, num_labels=10):
        super().__init__()
        self.num_labels = num_labels
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input, target, model=None):
        return self.loss_fn(input.logits.view(-1, self.num_labels), target)
