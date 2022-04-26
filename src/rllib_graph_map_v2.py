from pathlib import Path
import os
from tracemalloc import stop

import osmnx as ox
import pandas as pd

from gym.spaces import Box, Discrete
import ray
from ray import tune
from ray.rllib.agents import ppo
# from ray.rllib.examples.env.action_mask_env import ActionMaskEnv
from gym_graph_map.envs.graph_map_env_v2 import GraphMapEnvV2
# from ray_models.models import TorchActionMaskModel as Model
from ray_models.models import ActionMaskModel as Model
# from ray_models.action_mask_model import ActionMaskModel as Model
from ray.tune.integration.wandb import WandbLoggerCallback


args = {
    'no_masking': False,
    'run': 'APPO',  # PPO, APPO
    'eager_tracing': False,
    'stop_iters': 100,
    'stop_timesteps': 10000,
    'stop_reward': 200,
    'no_tune': False,
}

if __name__ == "__main__":
    # Init Ray in local mode for easier debugging.
    ray.init(include_dashboard=False)
    home = str(Path.home())
    graph_path = home + \
        "/dev/GraphRouteOptimizationRL/datasets/osmnx/houston_tx_usa_drive_2000.graphml"
    neg_df_path = home + "/dev/GraphRouteOptimizationRL/datasets/tx_flood.csv"
    G = ox.load_graphml(graph_path)
    print("Loaded graph")

    config = {
        "env": GraphMapEnvV2,
        "env_config": {
            'graph': G,
            'verbose': False,
            'neg_df': pd.read_csv(neg_df_path),
            'center_node': (29.764050, -95.393030),
            'threshold': 2900
        },
        "model": {
            "custom_model": Model,
            # disable action masking according to CLI
            "custom_model_config": {"no_masking": args['no_masking']},
        },
        "horizon": 100,
        "framework": "tf2",
        "num_gpus": 1,
        "eager_tracing": args['eager_tracing'],
        "log_level": 'INFO'
    }

    stop = {
        "training_iteration": args['stop_iters'],
        "timesteps_total": args['stop_timesteps'],
        "episode_reward_mean": args['stop_reward'],
    }
    # run with tune for auto trainer creation, stopping, TensorBoard, etc.
    results = tune.run(args['run'], config=config, stop=stop, verbose=2,
                       #    callbacks=[WandbLoggerCallback(
                       #        project="graph_map_ray",
                       #        group="Test",
                       #        excludes=["perf"],
                       #        log_config=False)]
                       )

    print("Finished successfully without selecting invalid actions.", results)
    ray.shutdown()
