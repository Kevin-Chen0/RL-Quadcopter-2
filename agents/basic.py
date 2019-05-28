# -*- coding: utf-8 -*-
import random
from task import Task

class Basic_Agent():
    def __init__(self, task):
        self.task = task

    def reset_episode(self):
        self.total_reward = 0.0
        self.count = 0
        state = self.task.reset()
        return state

    def step(self, reward, done):
        # Save experience / reward
        self.total_reward += reward
        self.count += 1

    def act(self, state):
        new_thrust = random.gauss(450., 25.)
        return [new_thrust + random.gauss(0., 1.) for x in range(4)]