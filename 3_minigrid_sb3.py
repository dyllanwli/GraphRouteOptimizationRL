#!/usr/bin/env python
# coding: utf-8

# In[4]:


import gym
import gym_minigrid
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO, A2C
from stable_baselines3 import DQN
from gym_minigrid import wrappers 


# In[ ]:


env = gym.make('MiniGrid-Empty-16x16-v0')
check_env(env, warn=True)
env = make_vec_env(lambda: env, n_envs=1)
model = DQN("MultiInputPolicy", env)


# In[ ]:


wrappers.FlatObsWrapper()

