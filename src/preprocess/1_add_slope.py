import osmnx as ox
import networkx as nx
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from copy import deepcopy


repo_path = str(Path.home()) + "/dev/GraphRouteOptimizationRL/"

graph_path = repo_path + \
    "datasets/osmnx/houston_tx_usa_drive_2000_no_isolated_nodes.graphml"
output = repo_path + "datasets/osmnx/houston_tx_usa_drive_2000_slope_tmp.graphml"
image_output = repo_path + "images/houston_tx_usa_drive_2000_slope_tmp.png"
G = ox.load_graphml(graph_path)

print("Loaded graph")

nG = deepcopy(G)
for u, v, data in G.edges(data=True):
    if data['length'] <= 0 or data['speed_kph'] <= 0 or data['travel_time'] <= 0:
        print("Edge {} has attribute 0".format((u, v)))
        nG.remove_edge(u, v)

G = ox.add_edge_grades(nG, precision=5)


print("Getting slope")
edge_grades = [data['grade_abs']
               for u, v, k, data in ox.get_undirected(G).edges(keys=True, data=True)]
avg_grade = np.mean(edge_grades)
print('Average street grade in this graph is {:.1f}%'.format(avg_grade*100))

med_grade = np.median(edge_grades)
print('Median street grade in this graph is {:.1f}%'.format(med_grade*100))

# get a color for each edge, by grade, then plot the network
ec = ox.plot.get_edge_colors_by_attr(
    G, 'grade_abs', cmap='plasma', num_bins=100)
fig, ax = ox.plot_graph(G, edge_color=ec, edge_linewidth=0.8,
                        node_size=0, save=True, filepath=image_output)

print("plotting")

ox.save_graphml(G, output)

print("Saved graph")
