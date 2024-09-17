import torch


class MetricWrapper:
    def __init__(self, metric, label):
        self.metric = metric
        self.label = label

    def to(self, device):
        self.metric = self.metric.to(device)
        return self

    def __call__(self, pred, targ):
        if targ.ndim == 2:
            targ = torch.argmax(targ, dim=1)
        return self.metric(pred, targ)
