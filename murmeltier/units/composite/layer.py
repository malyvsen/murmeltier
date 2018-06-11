from ..unit import Unit
from ..lin import Weights, Biases
from ..activations import Identity
from murmeltier.utils import assert_equal_or_none


class Layer(Unit):
    '''
    weights -> biases -> activation
    '''
    def __init__(in_specs = None, out_specs = None, activation_type = Identity, **kwargs):
        in_specs = out_specs = assert_equal_or_none(in_specs = in_specs, out_specs = out_specs)
        config(in_specs = in_specs, out_specs = out_specs)
        self.params['weights'] = Weights(in_specs = in_specs, out_specs = out_specs, **kwargs)
        self.params['biases'] = Biases(in_specs = out_specs, out_specs = out_specs, **kwargs)
        self.params['activation'] = activation_type(in_specs = out_specs, out_specs = out_specs, **kwargs)


    def get_output(self, input):
        return self.params['activation'].get_output(input * self.params['weights'] + self.params['biases'])
