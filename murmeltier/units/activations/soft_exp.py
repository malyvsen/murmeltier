import numpy as np
from ..unit import Unit


class SoftExp(Unit):
    '''
    Soft exponential activation from https://arxiv.org/abs/1602.01321
    for alpha < 0: -ln(1 - alpha * (input - alpha)) / alpha
    for alpha = 0: input
    for alpha > 0: (exp(input * alpha) - 1) / alpha + alpha
    '''
    def __init__(self, in_specs, out_specs = None, params = None, stddev = None):
        if out_specs is None:
            out_specs = in_specs
        if in_specs != out_specs:
            raise ValueError('in_specs and out_specs must be the same')
        param_randomizers = {}
        param_randomizers['alpha'] = lambda stddev: np.random.normal(scale = stddev)
        Unit.construct(self, in_specs = in_specs, out_specs = out_specs, param_names = param_randomizers.keys(), params = params, param_randomizers = param_randomizers, stddev = stddev)


    def get_output(self, input):
        alpha = self.params['alpha']
        if alpha < -1e-15:
            return -np.log(1 - alpha * (input - alpha)) / alpha
        if alpha > 1e-15:
            return (np.exp(input * alpha) - 1) / alpha + alpha
        return input
