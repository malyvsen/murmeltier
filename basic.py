import gym
import numpy as np
from layers import LeakyReLU
from agent import Agent
import time
import datetime


env_name = 'CartPole-v0'
env = gym.make(env_name)
hidden_layer_sizes = [8, 6, 4]
population_size = 64
population_stddev = 1
learning_rate = 0.5


optimal_agent = Agent(env = env, layer_type = LeakyReLU, hidden_layer_sizes = hidden_layer_sizes, stddev = population_stddev)


class Bubble:
    '''An agent, its environment, and miscellaneous information such as the total reward'''
    def __init__(self, agent = None):
        global env_name
        self.env = gym.make(env_name)
        self.reset_env()
        if agent is None:
            self.randomize_agent()
        else:
            self.agent = agent

    def get_weight(self):
        return np.exp(self.reward)

    def weighted_agent(self):
        return self.agent * self.get_weight()

    def work(self):
        if self.done:
            return
        action = self.agent.get_action(self.observation)
        self.observation, current_reward, self.done, info = self.env.step(action)
        self.reward += current_reward

    def reset_env(self):
        self.reward = 0
        self.observation = self.env.reset()
        self.done = False

    def randomize_agent(self):
        global optimal_agent
        self.agent = optimal_agent + Agent(env = self.env, layer_type = LeakyReLU, hidden_layer_sizes = hidden_layer_sizes, stddev = population_stddev)


bubbles = [Bubble() for i in range(population_size)]


def reset():
    global bubbles
    for bubble in bubbles:
        bubble.reset_env()
        bubble.randomize_agent()


def update_optimal_agent():
    global optimal_agent, bubbles, learning_rate
    weighted_agent = optimal_agent * 0
    for bubble in bubbles:
        weighted_agent += bubble.weighted_agent()
    weighted_agent /= sum([bubble.get_weight() for bubble in bubbles])
    optimal_agent += (weighted_agent - optimal_agent) * learning_rate


def do_agents_work():
    global bubbles
    all_done = True
    for bubble in bubbles:
        bubble.work()
        if not bubble.done:
            all_done = False
    return not all_done


def get_average_reward():
    return sum([bubble.reward for bubble in bubbles]) / len(bubbles)


def demo(target_fps = 30):
    bubble = Bubble(optimal_agent)
    while not bubble.done:
        frame_start_time = time.time()
        bubble.env.render()
        bubble.work()
        sleep_time = 1 / target_fps + frame_start_time - time.time()
        if sleep_time > 0:
            time.sleep(sleep_time)
    bubble.env.close()


def train(num_epochs = 128):
    start_time = time.time()
    for epoch in range(num_epochs):
        print('epoch ' + str(epoch + 1) + '/' + str(num_epochs))
        while do_agents_work():
            pass
        print('epoch finished, averge reward = ' + str(get_average_reward()))
        eta = (time.time() - start_time) * (num_epochs - epoch - 1) / (epoch + 1)
        print('ETA ' + str(datetime.timedelta(seconds = eta)))
        update_optimal_agent()
        reset()


train()
demo()
