# -*- coding: utf-8 -*-
import numpy as np
from agents.actor import Actor
from agents.critic import Critic
from agents.replay_buffer import ReplayBuffer
from agents.ou_noise import OUNoise

class DDPG():
    """Reinforcement Learning agent that learns using DDPG."""
    def __init__(self, task, mu=0.02, theta=0.16, sigma=0.21, buffer=500000,
                 batch=64, gamma=0.98, tau=0.02, learning=0.001, dropout=0.2):

        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high

        # Actor (Policy) Model
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high,
                                 learning, dropout)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high,
                                  learning, dropout)

        # Critic (Value) Model
        self.critic_local = Critic(self.state_size, self.action_size, learning, dropout)
        self.critic_target = Critic(self.state_size, self.action_size, learning, dropout)

        # Initialize target model parameters with local model parameters
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

        # Noise process
        self.exploration_mu = mu
        self.exploration_theta = theta
        self.exploration_sigma = sigma
        self.noise = OUNoise(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)

        # Replay memory
        self.buffer_size = buffer
        self.batch_size = batch
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        # Algorithm parameters
        self.gamma = gamma  # discount factor
        self.tau = tau  # for soft update of target parameters
        # Score tracker and learning parameters
#        self.best_w = None
        self.score = 0
        self.best_score = -np.inf
        self.noise_scale = 0.1

        # Episode variables
        self.reset_episode()


    def reset_episode(self):
        self.total_reward = 0.0
        self.count = 0
        self.noise.reset()
        state = self.task.reset()
        self.last_state = state
        return state

    def step(self, action, reward, next_state, done):
        # Save experience / reward
        self.total_reward += reward
        self.count += 1

         # Save experience / reward
        self.memory.add(self.last_state, action, reward, next_state, done)

        # Learn, if at terminal state and enough samples are available in memory
        if done and len(self.memory) > self.batch_size:
#        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

        # Roll over last state and action
        self.last_state = next_state

        # Track the score after every step (count)
        self.score = self.total_reward / float(self.count) if self.count else 0.0
        if self.score > self.best_score:
            self.best_score = self.score
#            self.best_w = self.w
#            self.noise_scale = max(0.5 * self.noise_scale, 0.01)
#        else:
#            self.w = self.best_w
#            self.noise_scale = min(2.0 * self.noise_scale, 3.2)

    def act(self, state):
        """Returns actions for given state(s) as per current policy."""
        state = np.reshape(state, [-1, self.state_size])
        action = self.actor_local.model.predict(state)[0]
        return list(action + self.noise.sample())  # add some noise for exploration

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples."""
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        # Get predicted next-state actions and Q values from target models
        #     Q_targets_next = critic_target(next_state, actor_target(next_state))
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

        # Train actor model (local)
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1])  # custom training function

        # Soft-update target models
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)

    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights), "Local and target model parameters must have the same size"

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)
