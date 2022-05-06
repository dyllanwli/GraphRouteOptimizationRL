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

output = repo_path + "datasets/embeddings/houston_tx_usa_drive_2000_slope"


G = ox.load_graphml(graph_path)
G = nx.relabel.convert_node_labels_to_integers(G, first_label=0, ordering='default')


# In[28]:


model = Node2Vec(dimensions=8, workers=16)
print("Fitting")
model.fit(G)

print("Getting embedding")
X = model.get_embedding()

np.save(output + "_node2vec.npy", X)

model = NetMF(dimensions=8)
print("Fitting")
model.fit(G)

print("Getting embedding")
X = model.get_embedding()

np.save(output + "_netmf.npy", X)

# In[26]:


# model = FeatherNode(reduction_dimensions=32)

# print("Fitting")
# model.fit(G, X)

# print("Getting embedding")
# X = model.get_embedding()

# np.save(output + "e_netmf_feather_32d.npy", X)


# In[ ]:





# %%
