import gym
from murmeltier.units.agents import Feedfoward
from murmeltier.units.composite import Layer
from murmeltier.units.activations import LeakyReLU
from murmeltier.learning import Evo
from murmeltier.utils import curry, demo


env_name = 'CartPole-v0' # try replacing this with Acrobot-v1
proto_env = gym.make(env_name)
hidden_unit_types = [curry(Layer, activation_type = LeakyReLU), curry(Layer, activation_type = LeakyReLU), curry(Layer, activation_type = LeakyReLU)]
hidden_specs = [8, 6]


def agent_randomizer(stddev):
    return Feedfoward(env = proto_env, hidden_unit_types = hidden_unit_types, hidden_specs = hidden_specs, stddev = stddev)


evo = Evo(env_name = env_name, agent_randomizer = agent_randomizer)
evo.train()
demo(env_name = env_name, agent = evo.optimal_agent)
