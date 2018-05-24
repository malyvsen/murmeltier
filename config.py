import gym

env_name = 'Acrobot-v1'
env = gym.make(env_name)
observation_dim = env.observation_space.shape[0]
num_possible_actions = env.action_space.n
population_size = 64
population_stddev = 1
learning_rate = 0.5
num_epochs = 256
