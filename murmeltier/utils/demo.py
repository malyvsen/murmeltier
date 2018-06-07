import time
from ..learning.bubble import Bubble


def demo(env_name, agent, target_fps = 30):
    bubble = Bubble(env_name = env_name, agent = agent)
    while not bubble.done:
        frame_start_time = time.time()
        bubble.env.render()
        bubble.step()
        sleep_time = 1 / target_fps + frame_start_time - time.time()
        if sleep_time > 0:
            time.sleep(sleep_time)
