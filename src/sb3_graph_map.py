from pathlib import Path
import osmnx as ox
import pandas as pd

from sb3_contrib import MaskablePPO
from sb3_contrib.common.envs import InvalidActionEnvDiscrete
from gym_graph_map.envs import GraphMapEnv
from env_checker import check_env
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks
from wandb.integration.sb3 import WandbCallback
import wandb

import torch

torch.cuda.empty_cache()

neighbors = []
# use neighbors as global variable 

def train(env):
    # Train the agent
    run = wandb.init(project="rl_osmnx", group="ppo_mask", sync_tensorboard=True)

    model = MaskablePPO("MultiInputPolicy", env, gamma=0.99, seed=40,
                        batch_size=256, verbose=1, tensorboard_log=f"runs/{run.id}")
    model.learn(5000, callback=WandbCallback(verbose=1))

    evaluate_policy(model, env, n_eval_episodes=20,
                    reward_threshold=2, warn=False)

    model.save("ppo_mask")
    del model  # remove to demonstrate saving and loading

    model = MaskablePPO.load("ppo_mask")
    run.finish()

    return model

def main():
    home = str(Path.home())
    graph_path = home + "/dev/GraphRouteOptimizationRL/houston_tx_usa_drive_2000.graphml"
    neg_df_path = home + "/dev/GraphRouteOptimizationRL/datasets/tx_flood.csv"
    G = ox.load_graphml(graph_path)
    print("Loaded graph")
    neg_df = pd.read_csv(neg_df_path)
    center_node = (29.764050, -95.393030)
    env = GraphMapEnv(G, neg_df, center_node=center_node, verbose=False)

    check_env(env)


    model = train(env)

    obs = env.reset()
    while True:
        # Retrieve current action mask
        action_masks = get_action_masks(env)
        action, _states = model.predict(obs, action_masks=action_masks)
        # action = env.action_space.sample()
        # print("take action:", action, env.node_dict_reversed[action])
        obs, rewards, done, info = env.step(action)
        # print(env.render())
        if done:
            break

    print("final reward:", rewards)
    env.render(mode="human")


if __name__ == "__main__":
    main()
