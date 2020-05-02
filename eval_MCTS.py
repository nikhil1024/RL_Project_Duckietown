# -*- coding: utf-8 -*-
"""
Created on Fri May  1 20:40:12 2020

@author: Pranav
"""
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import gym
from gym import spaces
from gym_duckietown.simulator import Simulator
import logging,sys
import warnings
warnings.filterwarnings("ignore")

logging.disable(sys.maxsize)

def launch_env(seed,map_name="loop_empty",id=None):
    env = None
    if id is None:
        # Launch the environment
        env = Simulator(
            seed=seed, # random seed
            map_name=map_name,
            max_steps=500001, # we don't want the gym to reset itself
            domain_rand=0,
            camera_width=640,
            camera_height=480,
            accept_start_angle_deg=4, # start close to straight
            full_transparency=True,
            distortion=True,
        )
    else:
        env = gym.make(id)

    return env

class ResizeWrapper(gym.ObservationWrapper):
    def __init__(self, env=None, shape=(120, 160, 3)):
        super(ResizeWrapper, self).__init__(env)
        self.observation_space.shape = shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            shape,
            dtype=self.observation_space.dtype)
        self.shape = shape

    def observation(self, observation):
        from PIL import Image
        return np.array(Image.fromarray(observation).resize(self.shape[0:2]))


class NormalizeWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(NormalizeWrapper, self).__init__(env)
        self.obs_lo = self.observation_space.low[0, 0, 0]
        self.obs_hi = self.observation_space.high[0, 0, 0]
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(0.0, 1.0, obs_shape, dtype=np.float32)

    def observation(self, obs):
        if self.obs_lo == 0.0 and self.obs_hi == 1.0:
            return obs
        else:
            return (obs - self.obs_lo) / (self.obs_hi - self.obs_lo)


class ImgWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ImgWrapper, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[0], obs_shape[1]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return observation.transpose(2, 0, 1)


class DtRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super(DtRewardWrapper, self).__init__(env)

    def reward(self, reward):
        if reward == -1000:
            reward = -10
        elif reward > 0:
            reward += 10
        else:
            reward += 4

        return reward

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


# Deprecated
class ActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(ActionWrapper, self).__init__(env)

    def action(self, action):
        action_ = [action[0], action[1]]
        return action_
    
def seed_lib(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

seed_lib(123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

actions = [2, 0, 1, 2, 0, 1, 0, 1, 2, 2, 1, 0, 0, 1, 2, 2, 0, 1, 2, 0, 1, 0, 2, 1, 2, 1, 0, 1,
           0, 2, 1, 2, 0, 0, 2, 1, 0, 1, 2, 0, 2, 1, 0, 2, 1, 2, 0, 1, 1, 2, 0, 0, 2, 1, 1, 2, 
           0, 2, 0, 1, 2, 1, 0, 0, 2, 1, 2, 0, 1, 0, 1, 2, 0, 2, 1, 0, 1, 2, 0, 2, 1, 2, 1, 0, 
           0, 1, 2, 1, 2, 0, 0, 1, 2, 2, 0, 1, 0, 1, 2, 1, 2, 0, 1, 0, 2, 2, 1, 0, 0, 1, 2, 0, 
           2, 1, 2, 0, 1, 2, 1, 0, 2, 0, 1, 2, 0, 1, 0, 2, 1, 2, 0, 1, 2, 0, 1, 0, 1, 2, 1, 2, 
           0, 2, 1, 0, 2, 1, 0, 1, 0, 2, 1, 2, 0, 0, 2, 1, 2, 0, 1, 2, 1, 0, 1, 2, 0, 0, 1, 2, 
           1, 2, 0, 0, 2, 1, 1, 2, 0, 2, 1, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 2, 1, 0, 1, 2, 0, 0, 
           2, 1, 1, 2, 0, 2, 0, 1, 0, 2, 1, 0, 2, 1, 0, 1, 2, 2, 0, 1, 2, 1, 0, 2, 1, 0, 1, 2, 
           0, 2, 0, 1, 0, 2, 1, 0, 1, 2, 1, 2, 0, 1, 2, 0, 2, 1, 0, 0, 2, 1, 1, 2, 0, 0, 1, 2, 
           2, 0, 1, 0, 2, 1, 0, 1, 2, 2, 1, 0, 1, 2, 0, 1, 0, 2, 1, 0, 2, 0, 2, 1, 1, 0, 2, 1, 
           0, 2, 0, 1, 2, 1, 0, 2, 0, 1, 2, 0, 1, 2, 2, 0, 1, 2, 1, 0, 1, 2, 0, 1, 2, 0, 1, 2, 
           0, 2, 0, 1, 0, 1, 2, 2, 0, 1, 2, 1, 0, 0, 1, 2, 1, 2, 0, 0, 1, 2, 1, 2, 0, 2, 0, 1, 
           2, 0, 1, 1, 0, 2, 0, 1, 2, 0, 1, 2, 2, 1, 0, 0, 2, 1, 0, 2, 1, 1, 0, 2, 1, 2, 0, 2, 
           1, 0, 0, 1, 2, 0, 2, 1, 2, 0, 1, 1, 2, 0, 1, 2, 0, 0, 1, 2, 1, 0, 2, 2, 1, 0, 0, 1, 
           2, 0, 1, 2, 1, 0, 2, 1, 0, 2, 0, 2, 1, 2, 0, 1, 0, 1, 2, 2, 0, 1, 2, 1, 0, 1, 2, 0, 
           2, 0, 1, 0, 2, 1, 1, 0, 2, 2, 1, 0, 2, 1, 0, 0, 2, 1, 0, 2, 1, 1, 0, 2, 0, 2, 1, 2,
           0, 1, 2, 1, 0, 0, 1, 2, 0, 2, 1, 0, 2, 1, 1, 2, 0, 2, 1, 0, 1, 0, 2, 1, 2, 0, 1, 0,
           2, 2, 0, 1, 0, 2, 1, 0, 2, 1, 2, 1, 0]

def lib_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

class ActorCNN(nn.Module):
    def __init__(self):
        super(ActorCNN, self).__init__()

        # ONLY TRU IN CASE OF DUCKIETOWN:
        flat_size = 32 * 9 * 14

        self.lr = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.sigm = nn.Sigmoid()

        self.conv1 = nn.Conv2d(3, 32, 8, stride=2)
        self.conv2 = nn.Conv2d(32, 32, 4, stride=2)
        self.conv3 = nn.Conv2d(32, 32, 4, stride=2)
        self.conv4 = nn.Conv2d(32, 32, 4, stride=1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(32)

        self.dropout = nn.Dropout(.5)

        self.lin1 = nn.Linear(flat_size, 512)
        self.lin2 = nn.Linear(512,3)



    def forward(self, x):
        x = self.bn1(self.lr(self.conv1(x)))
        x = self.bn2(self.lr(self.conv2(x)))
        x = self.bn3(self.lr(self.conv3(x)))
        x = self.bn4(self.lr(self.conv4(x)))
        x = x.reshape(x.size(0), -1)  # flatten
        x = self.dropout(x)
        x = self.lr(self.lin1(x))
        x = self.lin2(x)
        x = F.softmax(x)

        return x
    
    
env = launch_env(123,"loop_empty")
env = ResizeWrapper(env)
env = NormalizeWrapper(env)
env = ImgWrapper(env) # to make the images from 160x120x3 into 3x160x120
env = ActionWrapper(env)
env = DtRewardWrapper(env)
env = DiscreteWrapper(env)

states = []
img = env.reset()
states.append(img)
total_reward = 0
for action in actions:
    #env.render()
    img, reward, done, _ = env.step(action)
    total_reward += reward
    states.append(img)
 
    
print("Interim Reward:{}".format(total_reward))    
states = np.array(states[0:-1])
print(states.shape)


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
states_train,states_val,actions_train,actions_val = train_test_split(states,actions,test_size=0.2,
                                                                    stratify=actions,shuffle=True,random_state=123)
 

states_train = torch.tensor(states_train,dtype=torch.float).to(device)
states_val = torch.tensor(states_val,dtype=torch.float).to(device)
actions_train = torch.tensor(actions_train).to(device)
actions_val = torch.tensor(actions_val).to(device)
policy = ActorCNN().to(device)
optimizer  = torch.optim.Adam(policy.parameters())
criterion = nn.CrossEntropyLoss().to(device)


epoch = 0
stop = False
train_loss_array = []
val_loss_array = []
best_performance = (-1)*np.inf
rounds = 0
early_stopping_rounds = 10
while ((epoch <= 200)&(stop==False)):
  # Forward pass
  outputs = policy.forward(states_train)
  loss = criterion(outputs,actions_train)
  train_loss_array.append(loss.item())

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
    
        
    
  with torch.no_grad():
      val_outputs = policy.forward(states_val)
      val_loss  = criterion(val_outputs,actions_val)
      val_loss_array.append(val_loss.item())
      performance = accuracy_score(actions_val.detach().cpu(),torch.argmax(val_outputs,axis=1).cpu())
    
    
  #Check if we have an increase in performance
  if(performance > best_performance):
      rounds = 0
      best_performance = performance
      best_state_dict = policy.state_dict()
  else:
      rounds += 1
      if(rounds >= early_stopping_rounds):
        stop = True
    
    
    #Print statement, every 5 epochs or if it is the last epoch
  if((epoch%5==0)|(stop==True)):
      print("EPOCH:"+str(epoch))
      if(stop==True):
        print("Training to be concluded after this epoch") 
      print('Performance of the network in current epoch = '+str(round(performance,2)))
      print('Best performance of the network yet  = '+str(round(best_performance,2)))
    
    
  epoch += 1
    #While loop ends
      
print("BEST SCORE IS:"+str(round(best_performance,2)))
  # summarize history for loss
plt.plot(train_loss_array)
plt.plot(val_loss_array)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


policy.load_state_dict(best_state_dict)

def get_action(policy,img):
    img = torch.tensor(img,dtype=torch.float).reshape(1,3,120,160).to(device)
    action = torch.argmax(policy.forward(img),axis=1).cpu().numpy()[0]
    return(action)
    
  
env = launch_env(123,"loop_empty")
env = ResizeWrapper(env)
env = NormalizeWrapper(env)
env = ImgWrapper(env) # to make the images from 160x120x3 into 3x160x120
env = ActionWrapper(env)
env = DtRewardWrapper(env)
env = DiscreteWrapper(env)

states = []
img = env.reset()
states.append(img)
total_reward = 0
for action in actions:
#    env.render()
#    time.sleep(0.5)
    img, reward, done, _ = env.step(action)
    total_reward += reward
    states.append(img)
    
    
while(done==False):
    #env.render()
    #time.sleep(0.5)
    img, reward, done, _ = env.step(get_action(policy,img))
    total_reward+=reward
    
print("Total Reward:{}".format(total_reward))      
    
    
  