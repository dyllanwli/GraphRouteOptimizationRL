import osmnx as ox
import networkx as nx
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from copy import deepcopy

from torch import scalar_tensor
from get_elevation import add_node_elevations_opentopo

cities = ["nyc", "houston_tx"]
scales = ["2000", "20000"]

repo_path = str(Path.home()) + "/dev/GraphRouteOptimizationRL/"

for city in cities:
    for scale in scales:

        graph_path = repo_path + "datasets/osmnx/{}_usa_drive_{}.graphml".format(city, scale)
        print("Loading graph", graph_path)
        G = ox.load_graphml(graph_path)
        print(city + " " + scale + "has", len(G.nodes()), "nodes")
        print(city + " " + scale + "has", len(G.edges()), "edges")

