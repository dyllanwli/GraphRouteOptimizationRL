from tensorflow.python.ops.numpy_ops import np_config
from pathlib import Path
import os

# env
import osmnx as ox
import pandas as pd
import gym
from gym.spaces import Box, Discrete
from gym_graph_map.envs.graph_map_env_v3 import GraphMapEnvV3 as GraphMapEnv

# model
import ray
from ray import tune

from ray.rllib.agents import ppo

# tune
from ray.tune.integration.wandb import WandbLoggerCallback
# from ray.tune.suggest.hyperopt import HyperOptSearch

# render rgb_array
import imageio
import IPython
# from PIL import Image
from tqdm import tqdm
from pyvirtualdisplay import Display
display = Display(visible=0, size=(1400, 900))
display.start()

gym.logger.set_level(40)

# eager tensor debug
np_config.enable_numpy_behavior()

repo_path = str(Path.home()) + "/dev/GraphRouteOptimizationRL/"
embedding_dir = repo_path + "datasets/embeddings/"
graph_dir = repo_path + "datasets/osmnx/"


def create_policy_eval_video(env, trainer, filename="eval_video", num_episodes=200, fps=30):
    filename = repo_path + "images/" + filename + ".gif"
    with imageio.get_writer(filename, fps=fps) as video:
        obs = env.reset()
        for _ in tqdm(range(num_episodes)):
            action = trainer.compute_single_action(obs)
            obs, reward, done, info = env.step(action)
            im = env.render(mode="rgb_array", )
            video.append_data(im)
            if done:
                if info['arrived']:
                    print("Arrived at destination")
                    break
                else:
                    obs = env.reset()
        env.close()
    return filename


args = {
    'run': 'PPO',
    'stop_iters': 300,  # stop iters for each step
    'stop_timesteps': 1e+8,
    'stop_episode_reward_mean': 2.0,
    'train': True,
    'wandb': True,
    'checkpoint_path': '/h/diya.li/ray_results/PPO/PPO_GraphMapEnvV2_89ea7_00000_0_2022-04-30_12-47-01/checkpoint_000100/checkpoint-100'
}

if __name__ == "__main__":
    # Init Ray in local mode for easier debugging.
    ray.init(local_mode=True, include_dashboard=False)
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
        # 'wandb': args['wandb'],
    }
    config = {
        "env": GraphMapEnv,
        "env_config": env_config,
        "lambda": 0.999,
        "horizon": 2000,  # max steps per episode
        "framework": "tf2",
        "num_gpus": 0,
        "num_cpus_per_worker": 4,
        "num_envs_per_worker": 4,
        # "num_sgd_iter": 30, # Can not be tuned...
        # "sgd_minibatch_size": 128,
        "num_workers": 0,  # 0 for curiosity
        # For production workloads, set eager_tracing=True ; to match the speed of tf-static-graph (framework='tf'). For debugging purposes, `eager_tracing=False` is the best choice.
        "eager_tracing": True,
        "eager_max_retraces": None,
        "log_level": 'ERROR',
        "lr": 0.0007,  # 0.0003 or 0.0005 seem to work fine as well.
        'exploration_config': {
            "type": "Curiosity",
            "eta": 0.5,  # tune.grid_search([1.0, 0.5, 0.1]),  # curiosity
            "beta": 0.5,  # tune.grid_search([0.7, 0.5, 0.1]),
            "feature_dim": 256,  # curiosity
            # No actual feature net: map directly from observations to feature vector (linearly).
            # Hidden layers of the "inverse" model.
            "inverse_net_hiddens": [256],
            # Activation of the "inverse" model.
            "inverse_net_activation": "relu",
            # Hidden layers of the "forward" model.
            "forward_net_hiddens": [256],
            # Activation of the "forward" model.
            "forward_net_activation": "relu",
            "feature_net_config": {  # curiosity
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
            },
            "sub_exploration": {
                "type": "StochasticSampling",
            },
        }
    }

    stop = {
        "training_iteration": args['stop_iters'],
        "timesteps_total": args['stop_timesteps'],
        "episode_reward_mean": args['stop_episode_reward_mean'],
    }

    # hyperopt_search = HyperOptSearch(metric="episode_reward_mean", mode="max")

    checkpoints = None
    if args['wandb']:
        cb = [WandbLoggerCallback(
            project="graph_map_ray",
            group="ppo_cur_v3_1",
            excludes=["perf"],
            log_config=False)]
    else:
        cb = None
    if args['train']:
        # run with tune for auto trainer creation, stopping, TensorBoard, etc.
        results = tune.run(args['run'], config=config, stop=stop, verbose=0,
                           callbacks=cb,
                           keep_checkpoints_num=1,
                           checkpoint_at_end=True,
                           # search_alg=hyperopt_search,
                           )

        print("Finished successfully without selecting invalid actions.", results)

        trial = results.get_best_trial(
            metric="episode_reward_mean", mode="max")
        checkpoints = results.get_trial_checkpoints_paths(
            trial, metric="episode_reward_mean")
        print("Best checkpoint:", checkpoints)

    if checkpoints:
        checkpoint_path = checkpoints[0][0]
    else:
        checkpoint_path = args['checkpoint_path']
    ppo_config = ppo.DEFAULT_CONFIG.copy()
    # ppo_config.update(config)
    # config['num_workers'] = 0
    config['num_envs_per_worker'] = 1
    # trainer = ppo.PPOTrainer(config=config, env=GraphMapEnvV2)
    trainer = ppo.PPOTrainer(config=config, env=GraphMapEnv)
    trainer.restore(checkpoint_path)
    env = GraphMapEnv(config["env_config"])
    print("run one iteration until arrived and render")
    filename = create_policy_eval_video(env, trainer)
    print(filename, "recorded")
    ray.shutdown()
