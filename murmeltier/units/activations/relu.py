import numpy as np
from ..unit import Unit


class ReLU(Unit):
    '''
    Lets positive values pass, replaces negative ones with zeros
    '''
    def __init__(self, in_specs, out_specs = None, params = None):
        if out_specs is None:
            out_specs = in_specs
        if in_specs != out_specs:
            raise ValueError('in_specs and out_specs must be the same')
        Unit.construct(self, in_specs = in_specs, out_specs = out_specs)


    def get_output(self, input):
        return np.maximum(0, input)
