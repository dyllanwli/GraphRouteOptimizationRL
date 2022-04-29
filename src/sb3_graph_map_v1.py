from pathlib import Path
import osmnx as ox
import pandas as pd

from sb3_contrib import MaskablePPO
# from sb3_contrib.common.envs import InvalidActionEnvDiscrete
# from ray.rllib.examples.env.parametric_actions_cartpole import ParametricActionsCartPole
# from ray.rllib.examples.env.action_mask_env import ActionMaskEnv
from gym_graph_map.envs import GraphMapEnv
from env_checker import check_env
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks
from wandb.integration.sb3 import WandbCallback
import wandb

import torch

from gym import spaces

torch.cuda.empty_cache()

neighbors = []
# use neighbors as global variable


def train(env):
    # Train the agent
    run = wandb.init(project="rl_osmnx", group="point_state",
                     sync_tensorboard=True)

    model = MaskablePPO("MultiInputPolicy", env, gamma=0.999, seed=40,
                        batch_size=512, verbose=1, tensorboard_log=f"runs/{run.id}")
    model.learn(1000, callback=WandbCallback(verbose=1))

    mean_std_reward = evaluate_policy(model, env, n_eval_episodes=20,
                                      reward_threshold=2, warn=False)
    print("mean_std_reward:", mean_std_reward)

    model.save("ppo_mask")
    del model  # remove to demonstrate saving and loading

    model = MaskablePPO.load("ppo_mask")
    run.finish()

    return model


def main():
    repo_path = str(Path.home()) + "/dev/GraphRouteOptimizationRL/"

    graph_path = repo_path + "datasets/osmnx/houston_tx_usa_drive_2000.graphml"
    neg_df_path = repo_path + "datasets/tx_flood.csv"
    G = ox.load_graphml(graph_path)
    print("Loaded graph")
    env_config = {
        'graph': G,
        'verbose': False,
        'neg_df': pd.read_csv(neg_df_path),
        'center_node': (29.764050, -95.393030),
        'threshold': 2900
    }
    env = GraphMapEnv(env_config)
    # env_config = { # ActionMaskEnv
    #     "action_space": spaces.Discrete(100),
    #     "observation_space": spaces.Box(-1.0, 1.0, (5,)),
    # }
    # env = ParametricActionsCartPole(10)

    check_env(env)

    model = train(env)

    obs = env.reset()
    while True:
        # Retrieve current action mask
        action_masks = get_action_masks(env)
        # action, _states = model.predict(obs, action_masks=action_masks)
        action = env.action_space_sample()
        # print("take action:", action, env.node_dict_reversed[action])
        obs, rewards, done, info = env.step(action)
        # print(env.render())
        if done:
            break

    print("final reward:", rewards)
    env.render(mode="human")


if __name__ == "__main__":
    main()
