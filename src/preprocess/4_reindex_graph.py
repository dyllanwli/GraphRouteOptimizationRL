import osmnx as ox
import networkx as nx
from pathlib import Path

import numpy as np


repo_path = str(Path.home()) + "/dev/GraphRouteOptimizationRL/"
graph_path = repo_path +     "datasets/osmnx/houston_tx_usa_drive_2000_slope.graphml"

G = ox.load_graphml(graph_path)
G = nx.relabel.convert_node_labels_to_integers(G, first_label=0, ordering='default')

ox.write_graphml(G, graph_path)