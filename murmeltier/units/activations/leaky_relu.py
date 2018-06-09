import numpy as np
from ..unit import Unit


class LeakyReLU(Unit):
    '''
    max(input, input * alpha) with trainable alpha
    '''
    def __init__(self, in_specs, out_specs = None, params = None, initializer = None, initializers = None):
        if out_specs is None:
            out_specs = in_specs
        if in_specs != out_specs:
            raise ValueError('in_specs and out_specs must be the same')
        param_randomizers = {}
        param_randomizers['alpha'] = lambda param: initializers['alpha'](shape = ()) # TODO
        Unit.reset(self, in_specs = in_specs, out_specs = out_specs, param_names = param_randomizers.keys(), params = params, param_randomizers = param_randomizers, stddev = stddev)


    def get_output(self, input):
        return np.maximum(input, input * self.params['alpha'])
