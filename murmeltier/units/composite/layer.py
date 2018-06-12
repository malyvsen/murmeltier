from ..unit import Unit
from ..lin import Weights, Biases
from ..activations import Identity
from murmeltier.utils import assert_equal_or_none


class Layer(Unit):
    '''
    weights -> biases -> activation
    '''
    def __init__(self, in_specs, out_specs, activation_type = Identity, **kwargs):
        self.config(in_specs = in_specs, out_specs = out_specs)
        self.params['weights'] = Weights(in_specs = in_specs, out_specs = out_specs, **kwargs)
        self.params['biases'] = Biases(in_specs = out_specs, out_specs = out_specs, **kwargs)
        self.params['activation'] = activation_type(in_specs = out_specs, out_specs = out_specs, **kwargs)


    def get_output(self, input):
        pre_biases = self.params['weights'].get_output(input)
        post_biases = self.params['biases'].get_output(pre_biases)
        return self.params['activation'].get_output(post_biases)
