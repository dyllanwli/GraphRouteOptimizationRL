from pathlib import Path
import os

# env
import osmnx as ox
import pandas as pd
import gym
from gym.spaces import Box, Discrete
from gym_graph_map.envs.graph_map_env_v2 import GraphMapEnvV2

# model
import ray
from ray import tune
from ray.rllib.agents import ppo
from ray_models.models import ActionMaskModel as Model
# from ray_models.models import TorchActionMaskModel as Model
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

repo_path = str(Path.home()) + "/dev/GraphRouteOptimizationRL/"


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
    'no_masking': False,
    'run': 'PPO',  # PPO, APPO
    'stop_iters': 70,  # stop iters for each step
    'stop_timesteps': 1e+7,
    'stop_episode_reward_mean': 2.0,
    'train': True,
    'checkpoint_path': ''
}

if __name__ == "__main__":
    # Init Ray in local mode for easier debugging.
    ray.init(local_mode=True, include_dashboard=False)
    graph_path = repo_path + \
        "datasets/osmnx/houston_tx_usa_drive_2000_no_isolated_nodes.graphml"
    neg_df_path = repo_path + "datasets/tx_flood.csv"
    G = ox.load_graphml(graph_path)
    print("Loaded graph")
    env_config = {
        'graph': G,
        'verbose': False,
        'neg_df': pd.read_csv(neg_df_path),
        'center_node': (29.764050, -95.393030),  # sample
        # 'center_node': (29.72346214336903, -95.38599726549226), # houston
        'threshold': 2900
    }
    config = {
        "env": GraphMapEnvV2,
        "env_config": env_config,
        "model": {
            "custom_model": Model,
            # disable action masking according to CLI
            "custom_model_config": {"no_masking": args['no_masking']}
        },
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
        "log_level": 'ERROR',
        "lr": 0.0005,  # 0.0003 or 0.0005 seem to work fine as well.
        'exploration_config': {
            "type": "Curiosity",
            # in the policy model).
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
    if args['train']:
        # run with tune for auto trainer creation, stopping, TensorBoard, etc.
        results = tune.run(args['run'], config=config, stop=stop, verbose=0,
                        #    callbacks=[WandbLoggerCallback(
                        #        project="graph_map_ray",
                        #        group="ppo_cur_6",
                        #        excludes=["perf"],
                        #        log_config=False)],
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
    # ppo_config = ppo.DEFAULT_CONFIG.copy()
    # ppo_config.update(config)
    # config['num_workers'] = 0
    config['num_envs_per_worker'] = 1
    trainer = ppo.PPOTrainer(config=config, env=GraphMapEnvV2) if args['run'] == "PPO" else ppo.PPOTrainer(
        config=config, env=GraphMapEnvV2)
    trainer.restore(checkpoint_path)
    env = GraphMapEnvV2(config["env_config"])
    print("run one iteration until arrived and render")
    filename = create_policy_eval_video(env, trainer)
    print(filename, "recorded")
    ray.shutdown()
