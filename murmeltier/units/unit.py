class Unit:
    '''
    Base class for all units
    Only intended to be used as base class
    To inherit, override __init__ and get_output, see any other unit for example
    '''
    def __init__(self, params = None, in_space = None, out_space = None):
        '''
        Obligatory arguments in all derived classes:
        params (in dict form {name: param})
        in_specs, out_specs - specifications for input and output, usually just their dimensionality
        '''
        raise NotImplementedError('Attempted to initialize base unit type')


    def get_output(self, input = None):
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
        if type(self) != type(other):
            raise TypeError('Cannot add layers of different types')
        return type(self)(params = {key: self.params[key] + other.params[key] for key in self.params}, in_specs = self.in_specs, out_specs = self.out_specs)


    def __sub__(self, other):
        if type(self) != type(other):
            raise TypeError('Cannot subtract layers of different types')
        return self + (other * -1)


    def __mul__(self, scalar):
        return type(self)(params = {key: self.params[key] * scalar for key in self.params}, in_specs = self.in_specs, out_specs = self.out_specs)


    def __truediv__(self, scalar):
        return self * (1.0 / scalar)
