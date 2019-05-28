import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None,
        init_angle_velocities=None, runtime=5., target_pos=None, min_acc=2):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal: target this location
        self.target_pos = target_pos if target_pos is not None else np.array([10., 10., 10.])
        self.min_accuracy = min_acc

        # Distance between the agent position and target position
        self.target_distance = np.inf
        self.average_distance = np.inf
        self.best_distance = np.inf
        self.total_distance = []

    def get_reward(self, done):
        """Uses current pose of sim to return reward."""
        self.target_distance = (abs(self.sim.pose[:3] - self.target_pos)).sum()
        # Default reward
        reward = 1 - 0.3*self.target_distance

        if done:
            self.total_distance.append(self.target_distance)
            self.average_distance = np.mean(self.total_distance[-25:])
            if self.target_distance < self.best_distance:
                self.best_distance = self.target_distance

        if self.target_distance < self.min_accuracy * 4:
            reward += 1000 - 100*self.target_distance

            if self.target_distance < self.min_accuracy * 2:
                reward += 10000 - 1000*self.target_distance

                if self.target_distance < self.min_accuracy:
                    reward += 100000*(self.sim.time/(self.sim.runtime))

        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward(done)
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        """Once target position has been reached, automatic done"""
#        if np.array_equal(self.sim.pose[:3], self.target_pos):
#            done = True
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat)
        return state