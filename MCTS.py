import numpy as np
import pandas as pd
import random
import gym
import argparse
from gym import spaces
from gym_duckietown.simulator import Simulator
import logging,sys
import warnings
warnings.filterwarnings("ignore")

logging.disable(sys.maxsize)
#x = os.system("export PYGLET_DEBUG_GL=True")

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


class env_wrapper():
    def __init__(self,seed,map_name="loop_empty",parent_action_stack=None,action=None):
        self.seed = seed
        self.map_name = map_name
        if(parent_action_stack==None):
            self.action_stack = []
        else:
            self.action_stack = parent_action_stack
        if(action!=None):
            self.action_stack.append(action)
        
    def get_env(self):
        env = launch_env(seed=self.seed,map_name=self.map_name)
        env = ResizeWrapper(env)
        env = NormalizeWrapper(env)
        env = ImgWrapper(env) # to make the images from 160x120x3 into 3x160x120
        env = ActionWrapper(env)
        env = DtRewardWrapper(env)
        env = DiscreteWrapper(env)
        if(len(self.action_stack)>0):
            for action in self.action_stack:
                obs, reward, done, _ = env.step(action)
        return(env)
        

class Node():
    def __init__(self,seed,map_name,state,parent_action_stack,parent_idx,action,idx):
        self.seed = seed
        self.map_name = map_name
        self.idx = idx
        #self.state = state
        self.env_object = env_wrapper(seed,map_name,parent_action_stack,action)
        self.is_leaf=True
        self.is_terminal = False
        
        self.child_idx = []
        self.parent_idx = parent_idx
        self.UCT = np.inf
        self.visited = 0
        #self.rollout_reward = self.rollout()
        
    def add_node(self,action,node_dict):
        self.is_leaf = False
        new_idx = len(node_dict.keys())
        env = self.env_object.get_env()
        curr_state, reward, done, _ = env.step(action)
        
        node_dict[new_idx] = Node(self.seed,self.map_name,curr_state,self.env_object.action_stack,self.idx,action,new_idx)
        self.child_idx.append(new_idx)
        
    def rollout(self):
        #print("Executing rollout for node idx:{}".format(self.idx))
        done = False
        env = self.env_object.get_env()
        action = np.random.choice(range(0,3),replace=True)
        obs, reward, done, _ = env.step(action)
        if(done==True):
            self.is_terminal = True
            print("Direct terminality reached for node:{}".format(self.idx))
        rollout_reward = reward
        while(done!=True):
            action = np.random.choice(range(0,3),replace=True)
            obs, reward, done, _ = env.step(action)
            rollout_reward += reward
        del(env)
        return(rollout_reward)
            
        
        
        
        
        
        
class MCTS():
    
    def __init__(self,seed,map_name):
        env = env_wrapper(seed).get_env()
        self.root_node = Node(seed,map_name,env.reset(),None,None,None,0)
        self.node_dict = dict()
        self.node_dict[0] =self.root_node
        
    def get_best_child(self,node_idx):
        current_child = self.node_dict[node_idx]
        current_child.visited+=1
        UCTs = np.array([self.node_dict[child_idx].UCT for child_idx in current_child.child_idx ])
        return(current_child.child_idx[np.argmax(UCTs)])
        
            
    def selection(self):
        current_idx = 0
        #current_path = []
        current_node = self.node_dict[0]
        current_node.visited += 1
        while(current_node.is_leaf==False):
            current_idx = self.get_best_child(current_idx)
            current_node = self.node_dict[current_idx] 
            current_node.visited += 1
        print("leaf reached in expansion:{}".format(current_idx))
        
        return(current_idx)
    
    def expansion(self,node_idx):
        current_node = self.node_dict[node_idx]
        current_node.visited+=1
        unique_actions = np.arange(0,3)
        np.random.shuffle(unique_actions)
        for action in unique_actions:
            current_node.add_node(action,self.node_dict)
        return
    
    def rollout_node(self,node_idx):
        return(self.node_dict[node_idx].rollout())
    
    def backpropagate(self,node_idx,rollout_reward):
        #print("Executing backpropagation from node:{}".format(node_idx))
        node = self.node_dict[node_idx]
        node.UCT = rollout_reward
        node.visited+=1
        if(node.parent_idx!=None):
            self.update_node(node.parent_idx)
        return
        
        
    def update_node(self,node_idx):
        #print("..updating node:{}".format(node_idx))
        node = self.node_dict[node_idx]

        children_idx = [child_idx for child_idx in node.child_idx if (self.node_dict[child_idx].visited>0)]
        #print("...active children:{}".format(children_idx))
        average_reward = np.average([self.node_dict[child_idx].UCT for child_idx in children_idx])
        
        if(node.parent_idx!=None):
            node.UCT = average_reward + 2*np.sqrt(np.log(self.node_dict[node.parent_idx].visited)/node.visited)
            self.update_node(node.parent_idx)
        else:
            node.UCT = average_reward
        terminal_children = [True for child_idx in node.child_idx if self.node_dict[child_idx].is_terminal==True]
        if(len(terminal_children)==len(node.child_idx)):
            node.is_terminal==True
            print("Indirect terminality reached for node:{}".format(node.idx))
        return
    
 
def print_nodes_summary(node_dict):
    print("NODES_SUMMARY:")
    for node_idx in list(node_dict.keys()):
        node = node_dict[node_idx]
        print("Node idx:{},visited count:{},UCT:{},parent:{},children:{},is_terminal:{}".format(node_idx,node.visited,node.UCT,node.parent_idx,node.child_idx,node.is_terminal))

def get_path_reward(path,seed,map_name):
    env = env_wrapper(seed,map_name).get_env()
    path_reward = 0
    for action in path:
        _,reward,_,_ = env.step(action)
        path_reward+=reward
    del(env)
    return(path_reward)
    
def build_mcts(args):
    seed_lib(args.seed)
    mcts = MCTS(seed=args.seed,map_name=args.map_name)
    current_idx = 0
    current_node = mcts.node_dict[current_idx]
    itr = 0
    log_df = pd.DataFrame(columns=["iteration","current_reward"])
    while ((mcts.node_dict[0].is_terminal==False)&(itr<args.max_iterations)):
        if(current_node.is_terminal==True):
            print("Absolute terminality reached")
            itr = args.max_iterations
        if(current_node.is_leaf==True):
            current_best_path = current_node.env_object.action_stack
            if(current_node.visited==0):
                mcts.backpropagate(current_idx,current_node.rollout())
                #next_idx = current_idx
            else:
                mcts.expansion(current_idx)
                #next_idx = mcts.node_dict[current_idx].child_idx[0]     
            next_idx = 0
            itr+=1
            if((itr%args.view_freq==0)&(itr>0)):
                print("******ITERATION:{}******".format(itr))
                print_nodes_summary(mcts.node_dict)
                print("current_best_path:{}".format(current_best_path))
                path_reward = get_path_reward(current_best_path,args.seed,args.map_name)
                print("Current path reward:{}".format(path_reward))
                current_best_path = np.array(current_best_path)
#                np.save("{}_best_path.npy".format(args.map_name),current_best_path)
#                log_df = log_df.append({"iterations":itr,"current_reward":path_reward},ignore_index=True)
#                log_df.to_csv("{}_logs.csv".format(args.map_name),index=False)
        else:
            next_idx = mcts.get_best_child(current_idx)
        current_idx = next_idx    
        current_node = mcts.node_dict[current_idx]
    print("Ending loop")
    return
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # DDPG Args
    parser.add_argument("--seed", default=123, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--map_name", default="loop_empty", type=str)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--max_iterations", default=5000, type=int)  # How many maximum iterations to run the tree for
    parser.add_argument("--view_freq", default=5, type=float)  # How often (time steps) we evaluate
    
    build_mcts(parser.parse_args())    
