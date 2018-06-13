import numpy as np
from ..unit import Unit
from murmeltier.initializers import constant
from murmeltier.utils import assert_one


class Memory(Unit):
    '''
    Stores an array that is fed to internal sub-unit together with the input (appended to it)
    A chunk of the same size is then cut from the end of the sub-unit's output and overwrites the old array
    Effectively, implements an RNN
    '''
    def __init__(self, in_specs = None, out_specs = None, memory_size = None, hidden_unit = None, hidden_unit_type = None, initializer = constant, **kwargs):
        assert_one(hidden_unit = hidden_unit, hidden_unit_type = hidden_unit_type)
        if hidden_unit_type is not None:
            if in_specs is None:
                raise ValueError('Provide in_specs when providing hidden_unit_type')
            if out_specs is None:
                raise ValueError('Provide out_specs when providing hidden_unit_type')
            if memory_size is None:
                raise ValueError('Provide memory_size when providing hidden_unit_type')
            hidden_unit = hidden_unit_type(in_specs = in_specs + memory_size, out_specs = out_specs + memory_size, **kwargs)

        if memory_size is None:
            if in_specs is None and out_specs is None:
                raise ValueError('Provide in_specs or out_specs when not providing memory_size')
            if in_specs is not None:
                memory_size = hidden_unit.in_specs - in_specs
            if out_specs is not None:
                memory_size = hidden_unit.out_specs - out_specs

        if in_specs is None:
            in_specs = hidden_unit.in_specs - memory_size
        if in_specs != hidden_unit.in_specs - memory_size:
            raise ValueError('in_specs must be equal to hidden_unit.in_specs - memory_size')

        if out_specs is None:
            out_specs = hidden_unit.out_specs - memory_size
        if out_specs != hidden_unit.out_specs - memory_size:
            raise ValueError('out_specs must be equal to hidden_unit.out_specs - memory_size')

        self.config(in_specs = in_specs, out_specs = out_specs, initializers = {'memory': initializer(shape = memory_size)})
        self.initialize(params = {'hidden': hidden_unit}, initialize_subunits = False, **kwargs)


    def get_output(self, input):
        full_output = self.params['hidden'].get_output(np.concatenate((input, self.params['memory'])))
        self.params['memory'] = full_output[-len(self.params['memory']) : ]
        return full_output[ : -len(self.params['memory'])]
