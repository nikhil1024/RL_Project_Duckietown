# import gym_duckietown
from gym_duckietown.simulator import Simulator
from gym import spaces
import gym
import numpy as np


class Environment:
	def __init__(self, map_name, max_steps=1000):
		self.env = None
		self.map_name = map_name
		self.max_steps = max_steps

	def create_env(self):
		self.env = Simulator(
			seed=48304,
			map_name=self.map_name,
			max_steps=self.max_steps,
			domain_rand=0,
			camera_width=640,
			camera_height=480,
			accept_start_angle_deg=4,
			full_transparency=True,
			distortion=True
		)

		self.env.action_space.np_random.seed(48304)

		return self.env


class DiscreteWrapper(gym.ActionWrapper):
	"""
	Duckietown environment with discrete actions (left, right, forward)
	instead of continuous control
	"""

	def __init__(self, env):
		gym.ActionWrapper.__init__(self, env)
		self.action_space = spaces.Discrete(3)

	def action(self, action):
		# Turn left
		if action == 0:
			vels = [0.6, +1.0]
		# Turn right
		elif action == 1:
			vels = [0.6, -1.0]
		# Go forward
		elif action == 2:
			vels = [0.7, 0.0]
		else:
			assert False, "unknown action"

		return np.array(vels)


class DtRewardWrapper(gym.RewardWrapper):
	def __init__(self, env):
		super(DtRewardWrapper, self).__init__(env)

	def reward(self, reward):
		# if reward == -1000:
		# 	reward = -10
		# elif reward > 0:
		# 	reward += 10
		# else:
		# 	reward += 4

		return reward / 100
