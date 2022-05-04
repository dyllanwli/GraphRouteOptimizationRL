#!/usr/bin/env python
# coding: utf-8

# In[3]:


import osmnx as ox
import networkx as nx
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
from src.utils import plot_path
import wandb
from tqdm import tqdm
from collections import defaultdict
import pandas as pd

# for rendering
import imageio
import IPython
from PIL import Image
from pyvirtualdisplay import Display
display = Display(visible=0, size=(1400, 900))
display.start()
get_ipython().run_line_magic('matplotlib', 'inline')

G = ox.load_graphml("./houston_tx_usa_drive_500.graphml")
tx_record_path = "~/dev/GraphRouteOptimizationRL/datasets/tx_flood.csv"
df = pd.read_csv(tx_record_path)


# In[6]:


from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env


# In[4]:


from src.routegym.env import ShortestRouteEnv
env = ShortestRouteEnv(G, origin=0, goal=100, random_weights=(0, 100))


# In[ ]:


check_env(env)


# In[8]:


env.observation_space


# In[9]:


env.reset()


# In[11]:





# In[ ]:




