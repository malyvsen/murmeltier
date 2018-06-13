from copy import deepcopy
from murmeltier.utils import trimmed_dict
import inspect


class Unit:
    '''
    Base class for all units
    Only intended to be used as base class
    To inherit, override __init__ and get_output, see any other unit for example
    Derived classes must have attributes:
    * in_specs - the input specification of the unit, usually just the dimensionality
    * out_specs - same for output
    * params - a dictionary of {parameter_name: parameter_value}
    * initializers - a dictionary of {param_name: initializer_function}, does not need to contain all params
    It is a good idea to set up these attributes by calling config(...), possibly followed by initialize(...)
    '''
    def __init__(self, in_specs = None, out_specs = None, **kwargs):
        '''
        Obligatory arguments in derived classes as shown here
        See Unit documentation for their meaning
        kwargs are used to pass values to initializers
        '''
        raise NotImplementedError('Attempted to initialize base unit type')


    def get_output(self, input = None):
        '''
        Obligatory argument in all derived classes: input
        '''
        raise NotImplementedError('Attempted to get the output of base unit type')


    def config(self, in_specs = None, out_specs = None, param_names = None, initializers = None):
        '''
        Perform standard operations needed to configure/re-configure
        * in_specs - overwrite old in_specs if provided
        * out_specs - overwrite old out_specs if provided
        * param_names - if provided, params and initializers will be trimmed to this list
        * initializers - dictionary of initializers, old ones are updated with those if provided
        '''
        if not hasattr(self, 'params'):
            self.params = {}
        if initializers is None:
            initializers = {}
        if not hasattr(self, 'initializers'):
            self.initializers = {}

        if in_specs is not None:
            self.in_specs = in_specs
        if not hasattr(self, 'in_specs'):
            raise ValueError('Provide in_specs when configuring for the first time')
        if out_specs is not None:
            self.out_specs = out_specs
        if not hasattr(self, 'out_specs'):
            raise ValueError('Provide out_specs when configuring for the first time')

        self.initializers.update(initializers)
        if param_names is not None:
            self.params = trimmed_dict(dict = self.params, keys = param_names)
            self.params = trimmed_dict(dict = self.params, keys = param_names)


    def initialize(self, params = None, init_params = None, initialize_subunits = True, **kwargs):
        '''
        Perform standard operations needed to initialize/reset
        Valid input:
        * initialize(stddev = 1.0, loc = 0.0) # where possible, initializers are called like: params[param_name] = initializers[param_name](stddev = 1.0, loc = 0.0)
        * initialize(params = {'alpha': 0.5}, stddev = 1.0) # sets params['alpha'] to 0.5, everything else is initialized with stddev = 1.0
        * initialize(init_params = {'beta': {'stddev': 0.5}}, stddev = 1.0) # everything is initialized, beta initializer given distinct arguments
        * initialize(params = {'gamma': 0.5}, init_params = {'delta': {'stddev': 0.5}}, stddev = 1.0) # combination of the above
        '''
        if params is None:
            params = {}
        if init_params is None:
            init_params = {}
        self.params.update(params)
        for param_name in init_params:
            if param_name in params:
                continue
            if param_name in self.initializers:
                self.params[param_name] = self.initializers[param_name](**init_params[param_name])
            else:
                if initialize_subunits and isinstance(self.params[param_name], Unit):
                    self.params[param_name].initialize(**init_params[param_name])
                else:
                    raise ValueError('init_params contains key for param which cannot be initialized - the key is ' + str(param_name))
        for param_name in self.initializers:
            if param_name in params or param_name in init_params:
                continue
            self.params[param_name] = self.initializers[param_name](**kwargs)
        for param_name in self.params:
            if param_name in params or param_name in init_params or param_name in self.initializers:
                continue
            if not initialize_subunits or not isinstance(self.params[param_name], Unit):
                continue
            self.params[param_name].initialize(**kwargs)


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
