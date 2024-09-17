from rirtk.models.linear import Linear
from rirtk.models.ecapa_tdnn import EcapaTdnn
from rirtk.models.joiners import Joiner, TripleJoiner


def from_name(model_name, **kwargs):
    if model_name == 'EcapaTdnn':
        return EcapaTdnn(**kwargs)
    if model_name == 'Linear':
        return Linear(**kwargs)
    if model_name == 'Joiner':
        return Joiner(**kwargs)
    if model_name == 'TripleJoiner':
        return TripleJoiner(**kwargs)
    assert False, f'Wrong model name {model_name}'
