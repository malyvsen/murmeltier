from copy import deepcopy
from murmeltier.utils import assert_disjoint, assert_keys


class Unit:
    '''
    Base class for all units
    Only intended to be used as base class
    To inherit, override __init__ and get_output, see any other unit for example
    Derived classes must have attributes:
    * in_specs - the input specification of the unit, usually just the dimensionality
    * out_specs - same for output
    * params - a dictionary of {parameter_name: parameter_value}
    '''
    def __init__(self, params = None, in_specs = None, out_specs = None):
        '''
        Obligatory arguments in derived classes as shown here
        See Unit documentation for their meaning
        '''
        raise NotImplementedError('Attempted to initialize base unit type')


    def get_output(self, input = None):
        '''
        Obligatory argument in all derived classes: input
        '''
        raise NotImplementedError('Attempted to get the output of base unit type')


    def reset(self, in_specs = None, out_specs = None, param_names = None, params = None, initializers = None, init_param = None, init_params = None):
        '''
        Perform standard operations needed to initialize or reset
        * param_names - the param names of the unit, they will checked for if provided
        * params - exact parameters to set, these will not be initialized or copied
        * initializers - a dictionary of functions that initialize the parameter whose name is the key
        '''
        if params is None:
            params = {}
        if not hasattr(self, 'params'):
            self.params = {}
        if initializers is None:
            initializers = {}
        if not hasattr(self, 'initializers'):
            self.initializers = {}
        if init_params is None:
            init_params = {}
        if param_names is None:
            param_names = set(self.params.keys()) | set(params.keys()) | set(self.initializers.keys()) | set(initializers.keys())

        if in_specs is not None:
            self.in_specs = in_specs
        if not hasattr(self, 'in_specs'):
            raise ValueError('Provide in_specs when resetting for the first time')
        if out_specs is not None:
            self.out_specs = out_specs
        if not hasattr(self, 'out_specs'):
            raise ValueError('Provide out_specs when resetting for the first time')

        self.initializers.update(initializers)
        assert_keys(keys = param_names, initializers = self.initializers)

        if init_param is not None:
            if len(params) > 0:
                raise ValueError('Do not provide params and init_param simultaneously, as one would override the other')
            for param_name in param_names:
                if param_name not in init_params:
                    init_params[param_name] = init_param

        assert_disjoint(init_params = init_params, params = params)

        for param_name in self.init_params:
            self.params[param_name] = self.initializers[param_name](param = init_params[param_name])

        self.params.update(params)
        assert_keys(keys = param_names, params = self.params)


    def __add__(self, other):
        result = deepcopy(self)
        result.add_equals(other)
        return result


    def add_equals(self, other):
        '''
        Used internally to perform the += operation
        This behavior may be subject to change
        '''
        if not isinstance(other, Unit):
            raise TypeError('Cannot add non-unit to unit')
        if type(self) != type(other):
            raise TypeError('Cannot add units of different types')
        if self.params.keys() != other.params.keys():
            raise ValueError('Cannot add units of different architectures')
        for key in self.params:
            if isinstance(self.params[key], Unit):
                self.params[key].add_equals(other.params[key])
            else:
                self.params[key] += other.params[key]


    def __sub__(self, other):
        if type(self) != type(other):
            raise TypeError('Cannot subtract units of different types')
        return self + (other * -1)


    def __mul__(self, scalar):
        result = deepcopy(self)
        result.mul_equals(scalar)
        return result


    def mul_equals(self, scalar):
        '''
        Used internally to perform the *= operation
        This behavior may be subject to change
        '''
        for key in self.params:
            if isinstance(self.params[key], Unit):
                self.params[key].mul_equals(scalar)
            else:
                self.params[key] *= scalar


    def __truediv__(self, scalar):
        return self * (1.0 / scalar)
