import torch
import torch.nn as nn
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN


class EcapaTdnn(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            **kwargs,
    ):
        super().__init__()
        self.model = ECAPA_TDNN(
            input_size=input_dim,
            lin_neurons=output_dim,
            **kwargs,
        )

    def forward(self, inputs, inputs_len=None):
        outputs = self.model(inputs)
        outputs = torch.squeeze(outputs, dim=1)
        return outputs, inputs_len
