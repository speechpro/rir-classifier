import torch.nn as nn


class Linear(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, inputs, inputs_len=None):
        outputs = self.norm(inputs)
        outputs = self.relu(outputs)
        outputs = self.linear(outputs)
        return outputs, inputs_len
