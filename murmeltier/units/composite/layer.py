from ..lin import Weights, Biases
from ..structural import Stack


def layer(in_specs = None, out_specs = None, stddev = None, activation_type = None):
    '''
    weights -> biases -> activation
    Returns a layer constructor (which can be treated as a unit type later on)
    The constructor will only require the parameters not provided to this function
    activation_type is the only exception - if you do not provide it, no activation will be used
    '''
    def constructor(in_specs = in_specs, out_specs = out_specs, stddev = stddev):
        if in_specs is None:
            raise ValueError('Provide in_specs either before or after currying')
        if out_specs is None:
            raise ValueError('Provide out_specs either beofre or after currying')
        if stddev is None:
            raise ValueError('Provide stddev either before or after currying')

        weights = Weights(in_specs = in_specs, out_specs = out_specs, stddev = stddev)
        biases = Biases(in_specs = out_specs, stddev = stddev)
        if activation_type is None:
            return Stack(units = [weights, biases])

        activation = activation_type(in_specs = out_specs, out_specs = out_specs, stddev = stddev)
        return Stack(units = [weights, biases, activation])

    return constructor
