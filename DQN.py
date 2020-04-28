import logging
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.disable(sys.maxsize)

import random
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from environment import Environment
from environment import DiscreteWrapper
from environment import DtRewardWrapper
from matplotlib import pyplot as plt
from collections import namedtuple
from PIL import Image


class DQN(nn.Module):
	def __init__(self, h, w, output_dim):
		super(DQN, self).__init__()
		self.output_dim = output_dim

		self.conv1 = nn.Conv2d(in_channels=12, out_channels=16, kernel_size=3, stride=2)
		self.bn1 = nn.BatchNorm2d(16)
		self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2)
		self.bn2 = nn.BatchNorm2d(32)
		self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2)
		self.bn3 = nn.BatchNorm2d(64)

		def conv2d_size_out(size, kernel_size=3, stride=2):
			return (size - (kernel_size - 1) - 1) // stride + 1

		self.conv_w = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
		self.conv_h = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))

		self.linear = nn.Linear(self.conv_w*self.conv_h*64, output_dim)

	def forward(self, x):
		x = F.relu(self.bn1(self.conv1(x)))
		x = F.relu(self.bn2(self.conv2(x)))
		x = F.relu(self.bn3(self.conv3(x)))

		x = x.view(-1, self.conv_w*self.conv_h*64)
		x = self.linear(x)

		return x.view(-1, self.output_dim)


def weights_init(m):
	if type(m) == nn.Conv2d:
		nn.init.xavier_uniform_(m.weight.data)
		nn.init.constant_(m.bias.data, 0)
	if type(m) == nn.Linear:
		nn.init.xavier_uniform_(m.weight.data)
		nn.init.constant_(m.bias.data, 0)
	if type(m) == nn.BatchNorm2d:
		nn.init.normal_(m.weight.data, 1.0, 0.02)


class ReplayMemory(object):
	def __init__(self, capacity):
		self.capacity = capacity
		self.memory = []
		self.position = 0

	def push(self, state, next_state, action, reward, done):
		if len(self.memory) < self.capacity:
			self.memory.append(None)
		self.memory[self.position] = Transition(state, next_state, action, reward, done)
		self.position = (self.position + 1) % self.capacity

	def sample(self, batch_size):
		return random.sample(self.memory, batch_size)

	def __len__(self):
		return len(self.memory)


def get_screen():
	screen = env.render(mode='rgb_array').transpose((2, 0, 1))  # getting it into CHW format

	screen = screen / 255.0
	screen = torch.tensor(screen, dtype=torch.float32)

	return resize(screen)


def evaluate(env):
	num_episodes = 5
	episodic_rewards = []

	for episode in range(num_episodes):
		env.reset()
		total_reward = 0
		screens = [None] * num_screens_stacked
		screen_counter = 0
		for j in range(num_screens_stacked):
			screens[j] = get_screen()

		screen_counter = (screen_counter + 1) % num_screens_stacked

		state = torch.cat([x for x in screens], dim=0)
		action = torch.argmax(policy_net(state.unsqueeze(0).to(device)))
		for iters in range(env.max_steps):
			_, reward, done, info = env.step(action)
			total_reward += reward

			if done:
				break

			next_state = get_screen()
			screens[screen_counter] = next_state
			for j in range(num_screens_stacked - 1):
				next_state = torch.cat([next_state, screens[(screen_counter - j - 1) % num_screens_stacked]], dim=0)
			screen_counter = (screen_counter + 1) % num_screens_stacked

			action = torch.argmax(policy_net(next_state.unsqueeze(0).to(device)))

		episodic_rewards.append(total_reward)

	return sum(episodic_rewards) / num_episodes


def epsilon_greedy_action(state, epsilon):
	if np.random.uniform(0, 1) < epsilon:
		action = env.action_space.sample()
	else:
		Q_values = policy_net(state.unsqueeze(0).to(device))
		action = torch.argmax(Q_values)

	return action


def update():
	if len(memory) < BATCH_SIZE:
		return
	policy_net.zero_grad()

	transitions = memory.sample(BATCH_SIZE)
	batch = Transition(*zip(*transitions))

	states = torch.stack(batch.state).to(device)
	next_states = torch.stack(batch.next_state).to(device)
	actions = torch.stack(batch.action).to(device)
	rewards = torch.stack(batch.reward).to(device)
	not_done = 1 - torch.stack(batch.done).to(device)

	Q_values = policy_net(states).gather(1, actions)

	not_done.squeeze_(1)
	non_final_next_states = next_states[not_done > 0]
	Q_values_new = torch.zeros(BATCH_SIZE, 1).to(device)
	Q_values_new[not_done > 0] = target_net(non_final_next_states).max(1)[0].reshape(-1, 1).detach()

	target = Q_values_new * discount + rewards
	error = F.smooth_l1_loss(Q_values, target)
	error.backward()

	for param in policy_net.parameters():
		param.grad.data.clamp_(-1, 1)

	optimizer.step()


def train():
	num_episodes = 10000
	epsilon = start_epsilon
	decay_over_episodes = 100000
	decay_rate = np.exp(np.log(end_epsilon / start_epsilon) / decay_over_episodes)
	policy_quality = []

	for i in range(num_episodes):
		env.reset()
		screens = [None] * num_screens_stacked
		screen_counter = 0
		for j in range(num_screens_stacked):
			screens[j] = get_screen()

		screen_counter = (screen_counter + 1) % num_screens_stacked

		state = torch.cat([x for x in screens], dim=0)
		action = epsilon_greedy_action(state, epsilon)

		iters = 0
		episode_rewards = 0

		while True:
			_, reward, done, info = env.step(action)
			episode_rewards += reward

			if done:
				done = 1
			else:
				done = 0
			next_state = get_screen()
			screens[screen_counter] = next_state

			for j in range(num_screens_stacked-1):
				next_state = torch.cat([next_state, screens[(screen_counter-j-1) % num_screens_stacked]], dim=0)

			screen_counter = (screen_counter + 1) % num_screens_stacked

			reward = torch.tensor([reward], dtype=torch.float32)
			action = torch.tensor([action])
			done = torch.tensor([done])
			next_action = epsilon_greedy_action(next_state, epsilon)

			memory.push(state, next_state, action, reward, done)

			update()

			action = next_action

			if iters == env.max_steps - 1 or done:
				break

		if i % TARGET_UPDATE:
			target_net.load_state_dict(policy_net.state_dict())

		epsilon = start_epsilon * (decay_rate ** i)
		average_reward = evaluate(env)
		policy_quality.append(average_reward)

		print("--- Episode {} of {}: Average reward {} ---".format(i + 1, num_episodes, average_reward))

		if (i+1) % 50 == 0:
			state = {
				'episode': i+1,
				'lr': learning_rate,
				'discount': discount,
				'state_dict': policy_net.state_dict(),
				'optimizer': optimizer.state_dict(),
				'epsilon': epsilon,
				'policy_quality': policy_quality
			}

			model_save_path = r"trained_models\DQN\DQN_{}_{}_{}".format(BATCH_SIZE, learning_rate, discount)
			torch.save(state, model_save_path)
			print("Model saved at:", model_save_path)

	# model_save_path = r"trained_models\DQN\DQN_{}_{}".format(learning_rate, discount)
	# torch.save(policy_net.state_dict(), model_save_path)

	plt.plot([x for x in range(num_episodes)], policy_quality)
	plt.xlabel('Number of episodes')
	plt.ylabel('Total Rewards')
	plt.show()


if __name__ == '__main__':
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print(device)

	num_screens_stacked = 4
	start_epsilon = 1
	end_epsilon = 0.05
	discount = 0.99
	learning_rate = 0.0001
	BATCH_SIZE = 256
	TARGET_UPDATE = 100

	Transition = namedtuple('Transition', ('state', 'next_state', 'action', 'reward', 'done'))

	resize = T.Compose([
		T.ToPILImage(),
		T.Resize((120, 160), interpolation=Image.CUBIC),
		T.ToTensor()
	])

	env = Environment(map_name='loop_empty').create_env()
	env = DiscreteWrapper(env)
	env = DtRewardWrapper(env)
	env.reset()

	init_screen = get_screen()
	_, screen_height, screen_width = init_screen.shape
	n_actions = env.action_space.n

	policy_net = DQN(screen_height, screen_width, n_actions).to(device)
	policy_net.apply(weights_init)
	target_net = DQN(screen_height, screen_width, n_actions).to(device)
	target_net.apply(weights_init)
	target_net.load_state_dict(policy_net.state_dict())
	target_net.eval()

	optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)

	memory = ReplayMemory(10000)

	train()

