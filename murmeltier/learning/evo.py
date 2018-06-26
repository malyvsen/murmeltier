import numpy as np
from .bubble import Bubble
from ..utils import if_print


class Evo:
    '''
    A basic evolutionary algorithm for training agents interacting with OpenAI Gym environments
    Essentially discretely estimated gradient ascent
    '''
    def __init__(self, env_name, agent_randomizer):
        self.env_name = env_name
        self.agent_randomizer = agent_randomizer
        self.optimal_agent = agent_randomizer()


    def train(self, num_epochs = 128, population_size = 64, learning_rate = 1.0, verbosity = 2):
        bubbles = [Bubble(env_name = self.env_name) for i in range(population_size)]

        for epoch in range(num_epochs):
            if_print(verbosity >= 1, 'Epoch ' + str(epoch + 1) + '/' + str(num_epochs))
            for bubble in bubbles:
                bubble.agent = self.optimal_agent + self.agent_randomizer()
            for bubble in bubbles:
                bubble.episode()
                bubble.agent.initialize(explicit_only = True, init_params = {'memory': {}})

            rewards = np.array([bubble.reward for bubble in bubbles])
            mean_reward = np.mean(rewards)
            weights = rewards - mean_reward
            total_weights = np.sum(np.abs(weights))
            update_vector = self.optimal_agent * 0
            for i in range(len(bubbles)):
                update_vector += (bubbles[i].agent - self.optimal_agent) * weights[i] / total_weights
            self.optimal_agent += update_vector * learning_rate

            if_print(verbosity >= 2, 'Mean reward: ' + str(mean_reward))
            if_print(verbosity >= 1, '')

        return self.optimal_agent
