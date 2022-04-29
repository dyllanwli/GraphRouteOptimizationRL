import osmnx as ox
import networkx as nx
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from copy import deepcopy
from get_elevation import add_node_elevations_opentopo

repo_path = str(Path.home()) + "/dev/GraphRouteOptimizationRL/"

graph_path = repo_path + "datasets/osmnx/houston_tx_usa_drive_2000.graphml"

G = ox.load_graphml(graph_path)
nG = deepcopy(G)
for x in G.nodes():
    if len(list(G.neighbors(x))) == 0:
        print("Node {} has no neighbors".format(x))
        nG.remove_node(x)

del G
nG = ox.add_edge_speeds(nG)
nG = ox.add_edge_travel_times(nG)

nG = add_node_elevations_opentopo(nG)


ox.save_graphml(nG, repo_path + "datasets/osmnx/houston_tx_usa_drive_2000_no_isolated_nodes.graphml")