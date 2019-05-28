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
        self.target_pos = target_pos if target_pos is not None else np.array([10., 0., 10.])
        self.min_accuracy = min_acc

        ## Numerical representation of the distance to the target
        self.target_proximity = 0
        self.total_proximity = []
        self.average_proximity = 0.
        self.best_proximity = np.inf


    def get_reward(self, done):
        """Uses current pose of sim to return reward."""
#        distance_to_target = np.linalg.norm(self.target_pos - self.sim.pose[:3])
#        sum_acceleration = np.linalg.norm(self.sim.linear_accel)
#        reward = (5. - distance_to_target) * 0.3 - sum_acceleration * 0.05

        self.target_proximity = np.linalg.norm(self.sim.pose[:3] - self.target_pos)
        # Default reward
        reward = 1 - 0.3*self.target_proximity

        if done:
            self.total_proximity.append(self.target_proximity)
            self.average_proximity = np.mean(self.total_proximity[-10:])
            if self.target_proximity < self.best_proximity:
                self.best_proximity = self.target_proximity

        if self.target_proximity < self.min_accuracy * 4:
            reward += 1000 - 100*self.target_proximity
            ## Set major reward if agent gets to the target location (within an acceptable minimum accuracy).
            ## And most importantly, it stops the simulation at that time, except hover, which can't stop early.
            if self.target_proximity < self.min_accuracy * 2:
                reward += 10000 - 1000*self.target_proximity

                if self.target_proximity < self.min_accuracy:
                    ## Add an additional reward factor for reaching the target around the penultimate timestep,
                    ## but not necessarily to rush there so quickly as to zoom past uncontrollably.
                    reward += 100000 * (self.sim.time / (self.sim.runtime - 1))

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