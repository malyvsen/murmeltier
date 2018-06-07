import numpy as np
from ..unit import Unit


class Weights(Unit):
    '''
    Matrix multiplication
    '''
    def __init__(self, in_specs, out_specs, params = None, stddev = None):
        param_randomizers = {}
        param_randomizers['weights'] = lambda stddev: np.random.normal(scale = stddev, size = (in_specs, out_specs))
        Unit.construct(self, in_specs = in_specs, out_specs = out_specs, param_names = param_randomizers.keys(), params = params, param_randomizers = param_randomizers, stddev = stddev)


    def get_output(self, input):
        return np.matmul(input, self.params['weights'])
