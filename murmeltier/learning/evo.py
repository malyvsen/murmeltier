import numpy as np
from .bubble import Bubble


class Evo:
    '''
    A basic evolutionary algorithm for training agents interacting with OpenAI Gym environments
    '''
    def __init__(self, env_name, agent_randomizer):
        self.env_name = env_name
        self.agent_randomizer = agent_randomizer
        self.optimal_agent = agent_randomizer(stddev = 0)


    def train(self, num_epochs = 128, population_size = 64, stddev = 1.0, learning_rate = 0.5, weighter = None):
        if weighter is None:
            def weighter(rewards):
                rms = np.sqrt(np.mean(np.square(rewards)))
                if rms == 0:
                    return rewards + 1
                return np.exp(rewards / rms)

        bubbles = [Bubble(env_name = self.env_name) for i in range(population_size)]

        for epoch in range(num_epochs):
            print('Epoch ' + str(epoch + 1) + '/' + str(num_epochs))
            for bubble in bubbles:
                bubble.agent = self.random_agent(stddev = stddev)
            for bubble in bubbles:
                bubble.episode()

            rewards = np.array([bubble.reward for bubble in bubbles])
            weights = weighter(rewards)
            total_weight = np.sum(weights)
            weighted_agent = self.optimal_agent * 0
            for i in range(len(bubbles)):
                weighted_agent += bubbles[i].agent * weights[i] / total_weight
            self.optimal_agent += (weighted_agent - self.optimal_agent) * learning_rate

            print('Average reward: ' + str(np.mean(rewards)))
            print('')

        return self.optimal_agent


    def random_agent(self, stddev):
        return self.optimal_agent + self.agent_randomizer(stddev = stddev)
