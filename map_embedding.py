# This script is used to generate map embedding for GraphMapEnvV3. 
from karateclub import FeatherGraph

import osmnx as ox
import networkx as nx
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from copy import deepcopy


repo_path = str(Path.home()) + "/dev/GraphRouteOptimizationRL/"
graph_path = repo_path + \
    "datasets/osmnx/houston_tx_usa_drive_2000_slope.graphml"
output = repo_path + "datasets/osmnx/

model = FeatherGraph()
model.fit(graphs)
X = model.get_embedding()