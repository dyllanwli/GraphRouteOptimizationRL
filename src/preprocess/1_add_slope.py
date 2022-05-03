import osmnx as ox
import networkx as nx
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from copy import deepcopy


repo_path = str(Path.home()) + "/dev/GraphRouteOptimizationRL/"

graph_path = repo_path + "datasets/osmnx/houston_tx_usa_drive_2000_no_isolated_nodes.graphml"

G = ox.load_graphml(graph_path)

print("Loaded graph")

G = ox.add_edge_grades(G)

print("Getting slope")
edge_grades = [data['grade_abs'] for u, v, k, data in ox.get_undirected(G).edges(keys=True, data=True)]
avg_grade = np.mean(edge_grades)
print('Average street grade in this graph is {:.1f}%'.format(avg_grade*100))

med_grade = np.median(edge_grades)
print('Median street grade in this graph is {:.1f}%'.format(med_grade*100))

# get a color for each edge, by grade, then plot the network
ec = ox.plot.get_edge_colors_by_attr(G, 'grade_abs', cmap='plasma', num_bins=100)
fig, ax = ox.plot_graph(G, edge_color=ec, edge_linewidth=0.8, node_size=0, save=True, filepath=repo_path + "images/houston_tx_usa_drive_20000_slope.png")

print("plotting")

ox.save_graphml(G, repo_path + "datasets/osmnx/houston_tx_usa_drive_2000_slope.graphml")

print("Saved graph")