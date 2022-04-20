from pathlib import Path
from tqdm import tqdm
import osmnx as ox
import pandas as pd

import ray

from gym_graph_map.envs import GraphMapEnv

import torch

torch.cuda.empty_cache()


graph_path = str(Path.home()) + "/dev/GraphRouteOptimizationRL/houston_tx_usa_drive_2000.graphml"
neg_df_path = str(Path.home()) + "/dev/GraphRouteOptimizationRL/datasets/tx_flood.csv"
G = ox.load_graphml(graph_path)
print("Loaded graph")
neg_df = pd.read_csv(neg_df_path)
center_node = (29.764050, -95.393030)


@ray.remote
def test(x):
    sample_num = 10000
    df = pd.DataFrame(columns=["nx_shortest_path_length", "nx_shortest_travel_time_length"])
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
        df.loc[i] = [env.nx_shortest_path_length, env.nx_shortest_travel_time_length]
    df.to_csv( str(Path.home()) + "/dev/GraphRouteOptimizationRL/datasets/route_stat/df_{}.csv".format(x), index=False)

refs = [test.remote(i) for i in range(10)]

while True:
    ready_refs, remaining_refs = ray.wait(refs)
    if len(remaining_refs) == 0:
        break
    refs = remaining_refs


