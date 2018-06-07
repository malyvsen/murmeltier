import gym


class Bubble:
    '''
    An agent, its environment, and miscellaneous information such as the total reward
    The environment is an OpenAI Gym environment
    '''
    def __init__(self, env_name, agent = None):
        self.env = gym.make(env_name)
        self.agent = agent
        self.reset()


    def episode(self):
        if self.done:
            self.reset()
        while not self.done:
            self.step()
        return self.reward


    def step(self):
        if self.done:
            return
        action = self.agent.get_action(self.observation)
        self.observation, current_reward, self.done, info = self.env.step(action)
        self.reward += current_reward


    def reset(self):
        self.reward = 0
        self.observation = self.env.reset()
        self.done = False
