import gym
import numpy as np
from layer import *
from agent import *
import time
import datetime


optimal_agent = Agent(config.population_stddev)


class Bubble:
    '''An agent and its environment'''
    def __init__(self, agent = None):
        self.env = gym.make(config.env_name)
        self.reset(agent = agent)

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

    def reset(self, agent = None):
        global optimal_agent
        if agent is None:
            self.agent = optimal_agent + Agent(stddev = config.population_stddev)
        else:
            self.agent = agent
        self.reward = 0
        self.observation = self.env.reset()
        self.done = False


bubbles = [Bubble() for i in range(config.population_size)]


def reset():
    global bubbles
    for bubble in bubbles:
        bubble.reset()


def update_optimal_agent():
    global optimal_agent, bubbles
    weighted_agent = Agent(stddev = 0)
    for bubble in bubbles:
        weighted_agent += bubble.weighted_agent()
    weighted_agent /= sum([bubble.get_weight() for bubble in bubbles])
    optimal_agent += (weighted_agent - optimal_agent) * config.learning_rate


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


def train(num_epochs = config.num_epochs):
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
