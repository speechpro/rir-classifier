import torch.nn as nn


class Joiner(nn.Module):
    def __init__(self, body, head):
        super().__init__()
        self.body = body
        self.head = head

    def forward(self, inputs, inputs_len=None):
        outputs, outputs_len = self.body(inputs, inputs_len)
        outputs, outputs_len = self.head(outputs, outputs_len)
        return outputs, outputs_len


class TripleJoiner(nn.Module):
    def __init__(self, feet, body, head):
        super().__init__()
        self.feet = feet
        self.body = body
        self.head = head

    def forward(self, inputs, inputs_len=None):
        outputs, outputs_len = self.feet(inputs, inputs_len)
        outputs, outputs_len = self.body(outputs, outputs_len)
        outputs, outputs_len = self.head(outputs, outputs_len)
        return outputs, outputs_len
