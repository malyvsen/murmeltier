import numpy as np
from ..unit import Unit
from murmeltier.initializers import normal
from murmeltier.utils import assert_equal_or_none


class LeakyReLU(Unit):
    '''
    max(input, input * alpha) with trainable alpha
    '''
    def __init__(self, in_specs = None, out_specs = None, initializer = normal, **kwargs):
        in_specs = out_specs = assert_equal_or_none(in_specs = in_specs, out_specs = out_specs)
        initializers = {}
        initializers['alpha'] = initializer(shape = ())
        self.config(in_specs = in_specs, out_specs = out_specs, initializers = initializers)
        self.initialize(**kwargs)


    def get_output(self, input):
        return np.maximum(input, input * self.params['alpha'])
