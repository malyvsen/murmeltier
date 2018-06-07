import numpy as np
from ..unit import Unit


class Biases(Unit):
    '''
    Vector addition
    '''
    def __init__(self, in_specs, out_specs = None, params = None, stddev = None):
        if out_specs is None:
            out_specs = in_specs
        if in_specs != out_specs:
            raise ValueError('in_specs and out_specs must be the same')
        param_randomizers = {}
        param_randomizers['biases'] = lambda stddev: np.random.normal(scale = stddev, size = in_specs)
        Unit.construct(self, in_specs = in_specs, out_specs = out_specs, param_names = param_randomizers.keys(), params = params, param_randomizers = param_randomizers, stddev = stddev)


    def get_output(self, input):
        return input + params['biases']
