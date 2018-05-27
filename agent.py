import util


class Agent:
    '''
    An agent for interacting with any gym environment
    Interpretable as a vector in agent space, hence has vector operations implemented
    '''
    def __init__(self, env = None, observation_space = None, action_space = None, layers = None, layer_type = None, layer_types = None, hidden_layer_sizes = [], stddev = None, stddevs = None):
        if sum((env is not None, observation_space is not None or action_space is not None)) != 1:
            raise ValueError('Provide either env or both observation_space and action_space')
        if env is not None:
            self.observation_space = env.observation_space
            self.action_space = env.action_space
        else:
            if observation_space is None or action_space is None:
                raise ValueError('Provide both observation_space and action_space')
            self.observation_space = observation_space
            self.action_space = action_space

        if sum((layers is not None, layer_type is not None, layer_types is not None)) != 1:
            raise ValueError('Provide exactly one of: layers, layer_type, layer_types')
        if layers is not None:
            self.layers = layers
            return

        _layer_types = layer_types
        if layer_type is not None:
            _layer_types = [layer_type for i in range(len(hidden_layer_sizes) + 1)]
        if len(_layer_types) != len(hidden_layer_sizes) + 1:
            raise ValueError('layer_types must have length same as hidden_layer_sizes + 1')

        if sum((stddev is not None, stddevs is not None)) != 1:
            raise ValueError('Provide exactly one of: stddev, stddevs')
        _stddevs = stddevs
        if stddev is not None:
            _stddevs = [stddev for i in range(len(_layer_types))]
        if len(_stddevs) != len(_layer_types):
            raise ValueError('stddevs must have same length as layer_types')

        layer_sizes = [util.space_size(env.observation_space)] + hidden_layer_sizes + [util.space_size(env.action_space)]
        self.layers = []
        for i in range(len(_layer_types)):
            self.layers.append(_layer_types[i](in_dim = layer_sizes[i], out_dim = layer_sizes[i + 1], stddev = _stddevs[i]))


    def get_action(self, observation):
        val = util.observation_to_array(observation, self.observation_space)
        for layer in self.layers:
            val = layer.get_output(val)
        return util.array_to_action(val, self.action_space)


    def mutate(self, stddev):
        for layer in self.layers:
            layer.mutate(stddev = stddev)


    def __add__(self, other):
        if type(self) != type(other):
            raise ValueError('Cannot add agents of different types')
        result_layers = []
        for i in range(len(self.layers)):
            result_layers.append(self.layers[i] + other.layers[i])
        return type(self)(observation_space = self.observation_space, action_space = self.action_space, layers = result_layers)


    def __sub__(self, other):
        if type(self) != type(other):
            raise ValueError('Cannot subtract agents of different types')
        return self + (other * -1)


    def __mul__(self, scalar):
        result_layers = []
        for i in range(len(self.layers)):
            result_layers.append(self.layers[i] * scalar)
        return type(self)(observation_space = self.observation_space, action_space = self.action_space, layers = result_layers)


    def __truediv__(self, scalar):
        result_layers = []
        for i in range(len(self.layers)):
            result_layers.append(self.layers[i] * scalar)
        return type(self)(observation_space = self.observation_space, action_space = self.action_space, layers = result_layers)
