# -*- coding: utf-8 -*-
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from agents.agent import DDPG as DDPG_Agent
from task import Task


def perform(agent, num_episodes, key):
    results = defaultdict(list)
    phy_results = defaultdict(list)

    for i_episode in range(1, num_episodes+1):
        state = agent.reset_episode() # start a new episode
        duration = 0
        while True:
            action = agent.act(state)
            next_state, reward, done = task.step(action)
            agent.step(action, reward, next_state, done)
            state = next_state
            duration += task.sim.time

            if done:
                print("\rEpisode = {:4d}, score = {:7.3f} (best score = {:7.3f}), " \
                      "distance = {:7.3f} (best distance = {:7.3f})".format(
                    i_episode, agent.score, agent.best_score, task.target_distance,
                    task.best_distance), end="")

                results['episode'].append(i_episode)
                results['score'].append(agent.score)
                results['distance'].append(task.target_distance)
                results['duration'].append(duration)
                break

        sys.stdout.flush()

    return results, phy_results


if __name__ == '__main__':

    num_episodes = 2000  # 1000
    target_pos = np.array([10., 10., 10.])
    task = Task(target_pos=target_pos)

#    labels = ['episode', 'reward']
#    phy_labels = ['time', 'x', 'y', 'z', 'phi', 'theta', 'psi', 'x_velocity',
#          'y_velocity', 'z_velocity', 'phi_velocity', 'theta_velocity',
#          'psi_velocity', 'rotor_speed1', 'rotor_speed2', 'rotor_speed3', 'rotor_speed4']


#    print("Running Policy Agent...")
#    policy_agent = PolicySearch_Agent(task)
#    results, phy_results = perform(policy_agent, num_episodes, 'p')
#    plt.figure(0)
#    plt.plot(results['episode'], results['reward'], label='Reward / Episode')
#    plt.legend()
#    _ = plt.ylim()
#    plt.figure(1)
#    plt.plot(results['episode'], results['duration'], label='Duration / Episode')
#    plt.legend()
#    _ = plt.ylim()


    print("Running DDPG Agent...")
    ddpg_agent = DDPG_Agent(task)
    results, phy_results = perform(ddpg_agent, num_episodes, 'd')
    plt.figure(2)
    plt.plot(results['episode'], results['score'], label='Score / Episode')
    plt.legend()
    _ = plt.ylim()
    plt.figure(3)
    plt.plot(results['episode'], results['distance'], label='Distance / Episode')
    plt.legend()
    _ = plt.ylim()
    plt.figure(4)
    plt.plot(results['episode'], results['duration'], label='Duration / Episode')
    plt.legend()
    _ = plt.ylim()