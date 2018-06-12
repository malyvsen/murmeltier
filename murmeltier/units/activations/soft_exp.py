import numpy as np
from ..unit import Unit
from murmeltier.initializers import normal
from murmeltier.utils import assert_equal_or_none


class SoftExp(Unit):
    '''
    Soft exponential activation from https://arxiv.org/abs/1602.01321
    for alpha < 0: -ln(1 - alpha * (input + alpha)) / alpha
    for alpha = 0: input
    for alpha > 0: (exp(input * alpha) - 1) / alpha + alpha
    '''
    def __init__(self, in_specs, out_specs = None, initializer = normal, **kwargs):
        in_specs = out_specs = assert_equal_or_none(in_specs = in_specs, out_specs = out_specs)
        initializers = {}
        initializers['alpha'] = initializer(shape = ())
        self.config(in_specs = in_specs, out_specs = out_specs, initializers = initializers)
        self.initialize(**kwargs)


    def get_output(self, input):
        alpha = self.params['alpha']
        if alpha < -1:
            alpha = -1
        elif alpha > 1:
            alpha = 1

        if alpha < -1e-15:
            return -np.log(1 - alpha * (input + alpha)) / alpha
        if alpha > 1e-15:
            return (np.exp(input * alpha) - 1) / alpha + alpha
        return input
