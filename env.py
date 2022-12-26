import gym
from swarm_manager import SwarmManager


GLOBAL_CLOCK = 0


class AIGCEnv(gym.Env):

    def __init__(self):
        self._swarm_manager = SwarmManager()

    def reset(self):
        self._swarm_manager.reset()

    def step(self, action):
        pass

    def render(self, mode="human"):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        return


def make_env():
    return AIGCEnv()


if __name__ == "__main__":
    env = make_env()
    print(env)
