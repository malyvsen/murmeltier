from copy import deepcopy


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


    def construct(self, in_specs, out_specs, param_names = None, params = None, param_randomizers = None, stddev = None, stddevs = None):
        '''
        Perform standard operations needed to initialize
        param_names - the param names of the unit, they will be required and checked for
        params - exact parameters to set, these will not be randomized or copied
        param_randomizers - a dictionary of functions that randomize the parameter whose name is the key
        These functions should accept one argument: stddev
        '''
        self.in_specs = in_specs
        self.out_specs = out_specs

        if param_names is None or len(param_names) == 0:
            if params is not None and len(params) != 0:
                raise ValueError('param_names is empty, but params is not')
            self.params = {}
            return

        if sum((params is not None, stddev is not None, stddevs is not None)) != 1:
            raise ValueError('Provide exactly one of: params, stddev, stddevs')
        if params is not None:
            if params.keys() != param_names:
                raise ValueError('params should exactly have keys: ' + str(param_names))
            self.params = params
            return

        if param_randomizers is None:
            raise ValueError('Provide param_randomizers when not providing params')
        if stddev is not None:
            stddevs = {key: stddev for key in param_names}
        if stddevs.keys() != param_names:
            raise ValueError('stddevs should exactly have keys: ' + str(param_names))

        self.params = {}
        for key in param_names:
            self.params[key] = param_randomizers[key](stddev = stddevs[key])


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
