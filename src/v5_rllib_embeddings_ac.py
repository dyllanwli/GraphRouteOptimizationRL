from tensorflow.python.ops.numpy_ops import np_config
from pathlib import Path

# env
import osmnx as ox
import pandas as pd
import gym
from gym.spaces import Box, Discrete
from gym_graph_map.envs.graph_map_env_v5 import GraphMapEnvV5 as GraphMapEnv

# model
import ray
from ray import tune


from ray.rllib.agents import a3c
from ray_models.models import ActionMaskModel, TorchActionMaskModel
from ray.rllib.agents.callbacks import RE3UpdateCallbacks

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

# ignore UserWarning: Box bound precision
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
        env._reset_render()
        for _ in tqdm(range(num_episodes)):
            action = trainer.compute_single_action(obs)
            obs, reward, done, info = env.step(action)
            im = env.render(mode="human", show=True)
            video.append_data(im)
            if done:
                if info['arrived']:
                    print("Arrived at destination")
                    break
                else:
                    obs = env.reset()
                    env._reset_render()
        env.close()
    return filename


args = {
    'no_masking': False,
    'run': 'A3C',
    'stop_iters': 500,  # stop iters for each step
    'stop_timesteps': 1e+8,
    'stop_episode_reward_mean': 2.0,
    'train': True,
    'checkpoint_path': '/h/diya.li/ray_results/PPO/PPO_GraphMapEnvV4_15115_00000_0_2022-05-06_12-43-49/checkpoint_001000/checkpoint-1000',
    'wandb': True,
    'framework': 'tf2',
    'resume_name': '',
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
        'nblock': 2,
        'embedding_path': embedding_dir + "houston_tx_usa_drive_2000_slope_node2vec.npy",
        'envolving': False,  # neg envolving
        'envolving_freq': 50,  # every nth step
    }
    config = {
        "env": GraphMapEnv,
        "env_config": env_config,
        "model": {
            "custom_model": TorchActionMaskModel if args["framework"] == "torch" else ActionMaskModel,
            "fcnet_hiddens": [256, 512, 256],
            "fcnet_activation": "tanh",
            "use_lstm": False,
            "max_seq_len": 100,
            "lstm_cell_size": 256,
            "dim": 256,
            # "use_attention": False,
            # "attention_head_dim": 64,
            # "attention_dim": 256,
            "custom_model_config": {
                "no_masking": args['no_masking']
            }
        },
        "lambda": 0.9,
        "horizon": 2000,  # max steps per episode
        "framework": args['framework'],
        "num_gpus": 0,
        "num_cpus_per_worker": 2,
        "num_envs_per_worker": 1,
        "simple_optimizer": True,
        "num_workers": 20,  # 0 for curiosity

        "eager_tracing": True,
        "eager_max_retraces": None,
        "log_level": 'ERROR',
        "lr": 0.0005,  # 0.0003 or 0.0005
        "seed": 42,
        "callbacks": RE3UpdateCallbacks,
        'exploration_config': {
            "type": "RE3",
            # the dimensionality of the observation embedding vectors in latent space.
            "embeds_dim": 128,
            "rho": 0.1,  # Beta decay factor, used for on-policy algorithm.
            # Number of neighbours to set for K-NN entropy estimation.
            "k_nn": 50,
            # Configuration for the encoder network, producing embedding vectors from observations.
            # This can be used to configure fcnet- or conv_net setups to properly process any
            # observation space. By default uses the Policy model configuration.
            "encoder_net_config": {
                "fcnet_hiddens": [256, 512, 256],
                "fcnet_activation": "relu",
            },
            # Hyperparameter to choose between exploration and exploitation. A higher value of beta adds
            # more importance to the intrinsic reward, as per the following equation
            # `reward = r + beta * intrinsic_reward`
            "beta": 1,
            # Schedule to use for beta decay, one of constant" or "linear_decay".
            "beta_schedule": 'linear_decay',
            # Specify, which exploration sub-type to use (usually, the algo's "default"
            # exploration, e.g. EpsilonGreedy for DQN, StochasticSampling for PG/SAC).
            "sub_exploration": {
                "type": "StochasticSampling",
            }
        }
    }

    stop = {
        "training_iteration": args['stop_iters'],
        "timesteps_total": args['stop_timesteps'],
        "episode_reward_mean": args['stop_episode_reward_mean'],
    }

    # hyperopt_search = HyperOptSearch(metric="episode_reward_mean", mode="max")

    if args['wandb']:
        cb = [WandbLoggerCallback(
            project="graph_map_ray",
            group="ac_cur_nx_8",
            excludes=["perf"],
            log_config=False)]
    else:
        cb = None

    if args['train']:
        if args['resume_name'] != "":
            results = tune.run(args['run'], stop=stop, verbose=0, callbacks=cb, keep_checkpoints_num=1,
                               checkpoint_at_end=True, resume=True, name=args['resume_name'])
        else:
            results = tune.run(args['run'], config=config, stop=stop, verbose=0, callbacks=cb,
                               keep_checkpoints_num=1, checkpoint_at_end=True,
                               # search_alg=hyperopt_search,
                               )

        print("Finished successfully without selecting invalid actions.", results)

        trial = results.get_best_trial(
            metric="episode_reward_mean", mode="max")
        checkpoints = results.get_trial_checkpoints_paths(
            trial, metric="episode_reward_mean")
        print("Best checkpoint:", checkpoints)
        args['checkpoint_path'] = checkpoints[0][0]

    checkpoint_path = args['checkpoint_path']
    # ppo_config = ppo.DEFAULT_CONFIG.copy()
    # ppo_config.update(config)
    # config['num_workers'] = 0
    config['num_envs_per_worker'] = 1
    trainer = a3c.A3CTrainer(config=config, env=GraphMapEnv)

    trainer.restore(checkpoint_path)
    env = GraphMapEnv(config["env_config"])
    print("run one iteration until arrived and render")
    filename = create_policy_eval_video(env, trainer)
    print(filename, "recorded")
    ray.shutdown()
