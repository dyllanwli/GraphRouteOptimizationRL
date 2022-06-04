import osmnx as ox
import networkx as nx
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from copy import deepcopy
from tqdm import tqdm

repo_path = str(Path.home()) + "/dev/GraphRouteOptimizationRL/"
graph_path = repo_path + "datasets/osmnx/houston_tx_usa_drive_20000_slope.graphml"

print("Loading graph", graph_path)
G = ox.load_graphml(graph_path)

G = ox.add_edge_bearings(G)


# deprecated features below

directions = ("North", "Northeast", "East", "Southeast", "South", "Southwest", "West", "Northwest")
reverse_directions_dict = {i: d for i, d in enumerate(directions)}
directions_dict = {d: i for i, d in enumerate(directions)}

def get_direction(bearing):
    if bearing >= 337.5 or bearing < 22.5:
        return directions_dict["North"]
    elif bearing >= 22.5 and bearing < 67.5:
        return directions_dict["Northeast"]
    elif bearing >= 67.5 and bearing < 112.5:
        return directions_dict["East"]
    elif bearing >= 112.5 and bearing < 157.5:
        return directions_dict["Southeast"]
    elif bearing >= 157.5 and bearing < 202.5:
        return directions_dict["South"]
    elif bearing >= 202.5 and bearing < 247.5:
        return directions_dict["Southwest"]
    elif bearing >= 247.5 and bearing < 292.5:
        return directions_dict["West"]
    elif bearing >= 292.5 and bearing < 337.5:
        return directions_dict["Northwest"]
    else:
        raise ValueError("Bearing {} is not in the range [0, 360]".format(bearing))

for u, v, k, data in tqdm(G.edges(data=True, keys=True)):
    direction = get_direction(data["bearing"])
    nx.set_edge_attributes(G, {(u, v, k): {"direction": direction}})

ox.save_graphml(G, graph_path)
print("Done")