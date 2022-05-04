#!/usr/bin/env python
# coding: utf-8

# In[24]:


from karateclub import NetMF, Node2Vec, FeatherNode

import osmnx as ox
import networkx as nx
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from copy import deepcopy

repo_path = str(Path.home()) + "/dev/GraphRouteOptimizationRL/"
graph_path = repo_path +     "datasets/osmnx/houston_tx_usa_drive_2000_slope.graphml"

output = repo_path + "datasets/embeddings/"

G = ox.load_graphml(graph_path)
G = nx.relabel.convert_node_labels_to_integers(G, first_label=0, ordering='default')


# In[28]:


model = Node2Vec(dimensions=64, workers=16)
# model = NetMF()

print("Fitting")
model.fit(G)

print("Getting embedding")
X = model.get_embedding()


# In[26]:


model = FeatherNode(reduction_dimensions=32)

print("Fitting")
model.fit(G, X)

print("Getting embedding")
X = model.get_embedding()


# In[27]:


np.save(output + "houston_tx_usa_drive_2000_slope_node2vec_feather_32d.npy", X)


# In[ ]:





# %%
