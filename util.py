import numpy as np
from gym.spaces import *


def space_size(space):
    '''
    The number of real values needed to represent a space
    One-hot representation is chosen for discrete spaces
    Sign representation is chosen for binary spaces
    '''
    if isinstance(space, Box):
        return space.shape[0]
    if isinstance(space, Discrete):
        return space.n
    if isinstance(space, MultiDiscrete):
        return sum(space.nvec)
    if isinstance(space, MultiBinary):
        return space.n
    if isinstance(space, Tuple):
        return sum([get_space_size(subspace) for subspace in space.spaces])
    if isinstance(space, Dict):
        return sum([get_space_size(space.spaces[key]) for key in space.spaces])
    raise NotImplementedError('Unknown space')


def observation_to_array(observation, space):
    '''
    Observation represented as an 1-D np.array of np.float32
    One-hot representation is chosen for discrete spaces
    Sign representation is chosen for binary spaces
    '''
    if isinstance(space, Box):
        return np.array(observation)
    if isinstance(space, Discrete):
        return np.array([1.0 if observation == i else 0.0 for i in range(space.n)])
    if isinstance(space, MultiDiscrete):
        return np.array([[1.0 if observation[i] == i else 0.0 for i in range(n)] for n in space.nvec]).flatten()
    if isinstance(space, MultiBinary):
        return np.array(1.0 if value else -1.0 for value in observation)
    if isinstance(space, Tuple):
        return np.array([observation_to_array(observation[i], space.spaces[i]) for i in range(len(space.spaces))]).flatten()
    if isinstance(space, Dict):
        return np.array([observation_to_array(observation[key], space.spaces[key]) for key in space.spaces]).flatten()
    raise NotImplementedError('Unknown space')


def array_to_action(array, space):
    '''
    np.array of output neuron activations converted to action in given space
    One-hot representation is chosen for discrete spaces
    Sign representation is chosen for binary spaces
    '''
    if isinstance(space, Box):
        return tuple(action)
    if isinstance(space, Discrete):
        return np.argmax(array)
    if isinstance(space, MultiDiscrete):
        result_list = []
        beginning = 0
        for n in space.nvec:
            result_list.append(np.argmax(array[beginning : beginning + n]))
            beginning += n
        return np.array(result_list)
    if isinstance(space, MultiBinary):
        return np.array([1 if element > 0.0 else 0 for element in array], dtype = space.dtype)
    if isinstance(space, Tuple):
        result_list = []
        beginning = 0
        for subspace in space.spaces:
            n = space_size(subspace)
            result_list.append(array_to_action(array[beginning : beginning + n], subspace))
            beginning += n
        return tuple(result_list)
    if isinstance(space, Dict):
        result = {}
        beginning = 0
        for key in space.spaces:
            subspace = space.spaces[key]
            n = space_size(subspace)
            result[key] = array_to_action(array[beginning : beginning + n], subspace)
            beginning += n
        return result
    raise NotImplementedError('Unknown space')
