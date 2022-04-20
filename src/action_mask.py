import pandas as pd

from sb3_contrib import MaskablePPO
from sb3_contrib.common.envs import InvalidActionEnvDiscrete
from gym_graph_map.envs import GraphMapEnv
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.env_checker import check_env
from wandb.integration.sb3 import WandbCallback
import wandb

import torch

torch.cuda.empty_cache()

import osmnx as ox

def train(env):
    # Train the agent
    wandb.init(project="rl_osmnx",group="maskable_ppo")

    model = MaskablePPO("MultiInputPolicy", env, gamma=0.99, seed=32, batch_size = 64, verbose=1)
    model.learn(5000, callback=WandbCallback(verbose=2))

    evaluate_policy(model, env, n_eval_episodes=20, reward_threshold=2, warn=False)

    model.save("ppo_mask")
    del model # remove to demonstrate saving and loading

    model = MaskablePPO.load("ppo_mask")

    return model


def main():
    graph_path = "/h/diya.li/dev/GraphRouteOptimizationRL/houston_tx_usa_drive_2000.graphml"
    neg_df_path = "/h/diya.li/dev/GraphRouteOptimizationRL/datasets/tx_flood.csv"
    G = ox.load_graphml(graph_path)
    print("Loaded graph")
    neg_df = pd.read_csv(neg_df_path)
    center_node = (29.764050, -95.393030)
    env = GraphMapEnv(G, neg_df, center_node=center_node ,verbose=False)

    # check_env(env)

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