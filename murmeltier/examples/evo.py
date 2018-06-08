import gym
from murmeltier.units.agents import Feedfoward
from murmeltier.units.composite import layer
from murmeltier.units.activations import LeakyReLU, SoftExp
from murmeltier.learning import Evo
from murmeltier.utils import demo


env_name = 'CartPole-v0'
proto_env = gym.make(env_name)
hidden_unit_types = [layer(activation_type = SoftExp), layer(activation_type = LeakyReLU), layer(activation_type = LeakyReLU)]
hidden_specs = [8, 6]


def agent_randomizer(stddev):
    return Feedfoward(env = proto_env, hidden_unit_types = hidden_unit_types, hidden_specs = hidden_specs, stddev = stddev)


evo = Evo(env_name = env_name, agent_randomizer = agent_randomizer)
evo.train()
demo(env_name = env_name, agent = evo.optimal_agent)
