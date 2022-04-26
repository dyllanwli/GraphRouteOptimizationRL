"""
Use this script to analysis the reward function by sampling result from nx shortest path
The sample result of the script is saved at ~/dev/GraphRouteOptimizationRL/datasets/route_stat/df_{}.csv
load then by using env_stat.ipynb to analysis the random sample policy
"""
from pathlib import Path
from tqdm import tqdm
import osmnx as ox
import pandas as pd

import ray

from gym_graph_map.envs import GraphMapEnv

import torch

torch.cuda.empty_cache()

graph_path = str(Path.home()) + \
    "/dev/GraphRouteOptimizationRL/datasets/osmnx/houston_drive_20000.graphml"

# those files below are not used in this script
neg_df_path = str(Path.home()) + \
    "/dev/GraphRouteOptimizationRL/datasets/tx_flood.csv"
G = ox.load_graphml(graph_path)
# if those parameters are not set, uncomment the following line
# G = ox.speed.add_edge_speeds(G)
# G = ox.speed.add_edge_travel_times(G)
# G = ox.distance.add_edge_lengths(G)
# ox.save_graphml(G, graph_path)
# end of umcomment
print("Loaded graph")
neg_df = pd.read_csv(neg_df_path)
center_node = (29.764050, -95.393030)


@ray.remote
def test(x):
    sample_num = 2500
    df = pd.DataFrame(columns=["nx_shortest_path_length",
                      "nx_shortest_travel_time_length", "shortest_steps", "time_steps"])
    env = GraphMapEnv(G, neg_df, center_node=center_node, verbose=False)
    for i in range(sample_num):
        env.reset()
        env.get_default_route()
        # default_path = env.nx_shortest_path
        # env.render(plot_learned=False, default_path = default_path, save=False)
        # print(env.nx_shortest_path_length)
        # default_travel_path = env.nx_shortest_travel_time
        # env.render(plot_learned=False, default_path = default_travel_path, save=False)
        # print(env.nx_shortest_travel_time_length)
        df.loc[i] = [env.nx_shortest_path_length, env.nx_shortest_travel_time_length, len(
            env.nx_shortest_path), len(env.nx_shortest_travel_time)]
        # print("finished one sample")
    df.to_csv(str(Path.home()) +
              "/dev/GraphRouteOptimizationRL/datasets/route_stat/df_{}.csv".format(x), index=False)


refs = [test.remote(i) for i in range(20)]

while True:
    ready_refs, remaining_refs = ray.wait(refs)
    if len(remaining_refs) == 0:
        break
    refs = remaining_refs
