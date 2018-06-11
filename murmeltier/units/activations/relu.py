import numpy as np
from ..unit import Unit
from murmeltier.utils import assert_equal_or_none


class ReLU(Unit):
    '''
    Lets positive values pass, replaces negative ones with zeros
    '''
    def __init__(self, in_specs = None, out_specs = None, **kwargs):
        in_specs = out_specs = assert_equal_or_none(in_specs, out_specs)
        config(in_specs = in_specs, out_specs = out_specs)


    def get_output(self, input):
        return np.maximum(0, input)
