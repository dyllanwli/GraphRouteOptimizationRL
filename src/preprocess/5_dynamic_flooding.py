import osmnx as ox
import networkx as nx
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from copy import deepcopy


repo_path = str(Path.home()) + "/dev/GraphRouteOptimizationRL/"
graph_path = repo_path + "datasets/osmnx/houston_tx_usa_drive_20000_slope.graphml"

print("Loading graph", graph_path)
G = ox.load_graphml(graph_path)

def remove_isolated_nodes(G):
    nG = deepcopy(G)
    for x in G.nodes():
        if len(list(G.neighbors(x))) == 0:
            print("Node {} has no neighbors".format(x))
            nG.remove_node(x)
        if len(list(G.neighbors(x))) > 4:
            print("Node {} has more than 4 neighbors".format(x))
            nG.remove_node(x)
    # checking
    for x in nG.nodes():
        if len(list(nG.neighbors(x))) == 0:
            print("Node {} has no neighbors recursively".format(x))
            return remove_isolated_nodes(nG)
    return nG

G = remove_isolated_nodes(G)

# checking marshes
marshes = []
for x in G.nodes():
    neighbors = list(G.neighbors(x))
    # edges = list(G.edges(x, data=True))
    marsh = [n for n in neighbors if G.nodes[n]
             ["elevation"] > G.nodes[x]["elevation"]]
    # marsh = [x for u, v, data in edges if data["grade"] >= 0]
    # print(marsh, len(edges))
    # if len(marsh) == len(edges) and len(marsh) > 0:
    # marshes.append(x)
    if len(marsh) == len(neighbors):
        # marshes.append(x)
        nx.set_node_attributes(G, {x: {"marsh": True}})
    else:
        nx.set_node_attributes(G, {x: {"marsh": False}})


ox.save_graphml(G, graph_path)
print("Done")
# print(marshes)
# print(len(marshes))
