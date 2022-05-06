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

from gym_graph_map.envs import GraphMapEnvV2 as GraphMapEnv

import torch

torch.cuda.empty_cache()

repo_path = str(Path.home()) + "/dev/GraphRouteOptimizationRL/"
graph_path = repo_path + "datasets/osmnx/houston_tx_usa_drive_2000_slope.graphml"

# those files below are not used in this script
neg_df_path = repo_path + "datasets/tx_flood.csv"

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

repo_path = str(Path.home()) + "/dev/GraphRouteOptimizationRL/"
embedding_dir = repo_path + "datasets/embeddings/"
graph_dir = repo_path + "datasets/osmnx/"
stats_dir = repo_path + "datasets/route_stat/"

@ray.remote
def test(x):
    sample_num = 2500
    df = pd.DataFrame(columns=["nx_shortest_path_length",
                      "nx_shortest_travel_time_length", "shortest_steps", "time_steps", "unique_node"])
    
    graph_path = graph_dir + "houston_tx_usa_drive_2000_slope.graphml"

    G = ox.load_graphml(graph_path)
    print("Loaded graph")
    env_config = {
        'graph': G,
        'verbose': False,
        'neg_df_path': repo_path + "datasets/tx_flood.csv",
        'center_node': (29.764050, -95.393030),  # sample
        # 'center_node': (29.72346214336903, -95.38599726549226), # houston
        'threshold': 2900,
        'embedding_path': embedding_dir + "houston_tx_usa_drive_2000_slope_node2vec.npy",
    }
    env = GraphMapEnv(config=env_config)
    for i in range(sample_num):
        env.reset()
        env.get_default_route()
        # default_path = env.nx_shortest_path
        # env.render(plot_learned=False, default_path = default_path, save=False)
        # print(env.nx_shortest_path_length)
        # default_travel_path = env.nx_shortest_travel_time
        # env.render(plot_learned=False, default_path = default_travel_path, save=False)
        # print(env.nx_shortest_travel_time_length)
        unique_node = len(env.nx_shortest_path) == len(list(set(env.nx_shortest_path)))
        df.loc[i] = [env.nx_shortest_path_length, env.nx_shortest_travel_time_length, len(
            env.nx_shortest_path), len(env.nx_shortest_travel_time), unique_node]
        # print("finished one sample")
    df.to_csv(stats_dir + "df_{}.csv".format(x), index=False)


refs = [test.remote(i) for i in range(20)]

while True:
    ready_refs, remaining_refs = ray.wait(refs)
    if len(remaining_refs) == 0:
        break
    refs = remaining_refs
