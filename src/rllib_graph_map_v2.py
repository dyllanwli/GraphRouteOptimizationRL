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
from ray.tune.registry import register_env

# render rgb_array
import imageio
import IPython
from PIL import Image
from pyvirtualdisplay import Display
display = Display(visible=0, size=(1400, 900))
display.start()


def create_policy_eval_video(env, trainer, filename="eval_video", num_episodes=200, fps=30):
    filename = home + "/dev/GraphRouteOptimizationRL/images/" + filename + ".gif"
    with imageio.get_writer(filename, fps=fps) as video:
        obs = env.reset()
        for _ in range(num_episodes):
            action = trainer.compute_single_action(obs)
            obs, reward, done, info = env.step(action)
            im = env.render(mode="rgb_array", )
            video.append_data(im)
            if done:
                obs = env.reset()
                break
        env.close()
    return filename


args = {
    'no_masking': False,
    'run': 'APPO',  # PPO, APPO
    'stop_iters': 100,  # stop iters for each step
    'stop_timesteps': 1e+7,
    'stop_reward': 1*200
}

if __name__ == "__main__":
    # Init Ray in local mode for easier debugging.
    ray.init(local_mode=True, include_dashboard=False)
    home = str(Path.home())
    graph_path = home + \
        "/dev/GraphRouteOptimizationRL/datasets/osmnx/houston_tx_usa_drive_2000.graphml"
    neg_df_path = home + "/dev/GraphRouteOptimizationRL/datasets/tx_flood.csv"
    G = ox.load_graphml(graph_path)
    print("Loaded graph")
    env_config = {
        'graph': G,
        'verbose': False,
        'neg_df': pd.read_csv(neg_df_path),
        'center_node': (29.764050, -95.393030),
        'threshold': 2900
    }
    config = {
        "env": GraphMapEnvV2,
        "env_config": env_config,
        "model": {
            "custom_model": Model,
            # disable action masking according to CLI
            "custom_model_config": {"no_masking": args['no_masking']},
        },
        "lambda": 0.9999,
        "horizon": 300,  # max steps per episode
        "framework": "tf2",
        "num_gpus": 1,
        "num_workers": 5,
        # For production workloads, set eager_tracing=True ; to match the speed of tf-static-graph (framework='tf'). For debugging purposes, `eager_tracing=False` is the best choice.
        "eager_tracing": False,
        "log_level": 'ERROR'
    }

    stop = {
        "training_iteration": args['stop_iters'],
        "timesteps_total": args['stop_timesteps'],
        "episode_reward_mean": args['stop_reward'],
    }

    # run with tune for auto trainer creation, stopping, TensorBoard, etc.
    results = tune.run(args['run'], config=config, stop=stop, verbose=0,
                       callbacks=[WandbLoggerCallback(
                           project="graph_map_ray",
                           group="tune_1",
                           excludes=["perf"],
                           log_config=False)],
                       keep_checkpoints_num=1,
                       checkpoint_at_end=True,
                       )

    print("Finished successfully without selecting invalid actions.", results)

    trial = results.get_best_trial(metric="episode_reward_mean", mode="max")
    checkpoints = results.get_trial_checkpoints_paths(
        trial, metric="episode_reward_mean")
    print("Best checkpoint:", checkpoints)

    # checkpoints = [('/h/diya.li/ray_results/APPO/APPO_GraphMapEnvV2_ca2bd_00000_0_2022-04-26_21-14-38/checkpoint_000005/checkpoint-5', 40.55825641636045)]
    if checkpoints:
        checkpoint_path = checkpoints[0][0]
        # ppo_config = ppo.DEFAULT_CONFIG.copy()
        # ppo_config.update(config)
        config['num_workers'] = 1
        trainer = ppo.APPOTrainer(config=config, env=GraphMapEnvV2)
        trainer.restore(checkpoint_path)
        env = GraphMapEnvV2(config["env_config"])
        print("run one iteration until done and render")
        filename = create_policy_eval_video(env, trainer)
        print(filename, "recorded")
    ray.shutdown()
