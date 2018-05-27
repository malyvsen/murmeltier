import numpy as np


class Layer:
    '''
    Base class for layers
    Only intended to be used as base class
    To inherit, override __init__ and get_output, see BasicLayer below for example
    '''
    def __init__(self, params):
        '''Obligatory argument: params'''
        raise NotImplementedError('Do not initialize base layer type')


    def get_output(self, input):
        raise NotImplementedError('Do not get the output of base layer type')


    def construct(self, target_keys, param_randomizers, params = None, stddev = None, stddevs = None):
        '''
        Perform standard operations needed to initialize
        target_keys - the param names of the layer
        param_randomizers - a dictionary of functions that randomize the parameter given by key
        Should accept one argument: stddev
        '''
        if sum((params is not None, stddev is not None, stddevs is not None)) != 1:
            raise ValueError('Provide exactly one of: params, stddev, stddevs')
        if params is not None:
            if params.keys() != target_keys:
                raise ValueError('params should exactly have keys: ' + str(target_keys))
            self.params = params
            return

        _stddevs = stddevs
        if stddev is not None:
            _stddevs = {key: stddev for key in target_keys}
        if _stddevs.keys() != target_keys:
            raise ValueError('stddevs should exactly have keys: ' + str(target_keys))

        self.params = {}
        for key in target_keys:
            self.params[key] = param_randomizers[key](stddev = _stddevs[key])


    def mutate(self, stddev = None, stddevs = None):
        if sum((stddev is not None, stddevs is not None)) != 1:
            raise ValueError('Provide exactly one of: stddev, stddevs')
        _stddevs = stddevs
        if stddev is not None:
            _stddevs = {key: stddev for key in self.params}
        for key in self.params:
            self.params[key] += np.random.normal(scale = _stddevs[key], size = np.shape(self.params[key]))


    def __add__(self, other):
        if type(self) != type(other):
            raise ValueError('Cannot add layers of different types')
        return type(self)(params = {key: self.params[key] + other.params[key] for key in self.params})


    def __sub__(self, other):
        if type(self) != type(other):
            raise ValueError('Cannot subtract layers of different types')
        return self + (other * -1)


    def __mul__(self, scalar):
        return type(self)(params = {key: self.params[key] * scalar for key in self.params})


    def __truediv__(self, scalar):
        return self * (1.0 / scalar)



class Basic(Layer):
    '''
    A basic layer consisting of only weights and biases
    '''
    def __init__(self, params = None, stddev = None, stddevs = None, in_dim = None, out_dim = None):
        if params is None:
            if in_dim is None or out_dim is None:
                raise ValueError('Please provide in_dim and out_dim')
        target_keys = set(['weights', 'biases'])
        param_randomizers = {}
        param_randomizers['weights'] = lambda stddev: np.random.normal(scale = stddev, size = (out_dim, in_dim))
        param_randomizers['biases'] = lambda stddev: np.random.normal(scale = stddev, size = out_dim)
        Layer.construct(self, target_keys = target_keys, param_randomizers = param_randomizers, params = params, stddev = stddev, stddevs = stddevs)


    def get_output(self, input):
        return np.matmul(self.params['weights'], input) + self.params['biases']



class ReLU(Layer):
    '''
    A layer consisting of weights and biases followed by a ReLU activation
    '''
    def __init__(self, params = None, stddev = None, stddevs = None, in_dim = None, out_dim = None):
        if params is None:
            if in_dim is None or out_dim is None:
                raise ValueError('Please provide in_dim and out_dim')
        target_keys = set(['weights', 'biases'])
        param_randomizers = {}
        param_randomizers['weights'] = lambda stddev: np.random.normal(scale = stddev, size = (out_dim, in_dim))
        param_randomizers['biases'] = lambda stddev: np.random.normal(scale = stddev, size = out_dim)
        Layer.construct(self, target_keys = target_keys, param_randomizers = param_randomizers, params = params, stddev = stddev, stddevs = stddevs)


    def get_output(self, input):
        return np.maximum(np.matmul(self.params['weights'], input) + self.params['biases'], 0)



class LeakyReLU(Layer):
    '''
    A layer consisting of weights and biases followed by a leaky ReLU activation
    Negative leakiness is acceptable and causes the layer to produce output like abs(pre_activation)
    '''
    def __init__(self, params = None, stddev = None, stddevs = None, in_dim = None, out_dim = None):
        if params is None:
            if in_dim is None or out_dim is None:
                raise ValueError('Please provide in_dim and out_dim')
        target_keys = set(['weights', 'biases', 'leakiness'])
        param_randomizers = {}
        param_randomizers['weights'] = lambda stddev: np.random.normal(scale = stddev, size = (out_dim, in_dim))
        param_randomizers['biases'] = lambda stddev: np.random.normal(scale = stddev, size = out_dim)
        param_randomizers['leakiness'] = lambda stddev: np.random.normal(scale = stddev)
        Layer.construct(self, target_keys = target_keys, param_randomizers = param_randomizers, params = params, stddev = stddev, stddevs = stddevs)


    def get_output(self, input):
        pre_activation = np.matmul(self.params['weights'], input) + self.params['biases']
        return np.maximum(pre_activation, pre_activation * self.params['leakiness'])
