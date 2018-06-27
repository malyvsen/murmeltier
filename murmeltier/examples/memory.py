import gym
from murmeltier.units.agents import Feedfoward
from murmeltier.units.structural import Stack, Memory
from murmeltier.units.composite import Layer
from murmeltier.units.activations import LeakyReLU
from murmeltier.learning import Evo
from murmeltier.utils import curry, demo


env_name = 'Pendulum-v0'
proto_env = gym.make(env_name)
hidden_unit_type = curry(Layer, activation_type = LeakyReLU)
stack_type = curry(Stack, unit_type = hidden_unit_type, hidden_specs = [6])
memory_type = curry(Memory, hidden_unit_type = stack_type, memory_size = 1)


def agent_randomizer():
    return Feedfoward(env = proto_env, hidden_unit_type = memory_type, hidden_specs = [], stddev = 0.125, init_params = {'memory': {}})


evo = Evo(env_name = env_name, agent_randomizer = agent_randomizer)
evo.train(learning_rate = 1e-4)
demo(env_name = env_name, agent = evo.optimal_agent)
