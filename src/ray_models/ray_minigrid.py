from pathlib import Path
from collections import deque
import gym
import gym_minigrid
import numpy as np
import sys
import os
import unittest

import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
import ray.rllib.agents.ppo as ppo
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.utils.numpy import one_hot
from ray.tune import register_env
from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune.integration.wandb import WandbLoggerCallback


# render rgb_array
import imageio
import IPython
from PIL import Image
from pyvirtualdisplay import Display
display = Display(visible=0, size=(1400, 900))
display.start()


class MyCallBack(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self.deltas = []

    def on_postprocess_trajectory(
        self,
        *,
        worker,
        episode,
        agent_id,
        policy_id,
        policies,
        postprocessed_batch,
        original_batches,
        **kwargs
    ):
        pos = np.argmax(postprocessed_batch["obs"], -1)
        x, y = pos % 8, pos // 8
        self.deltas.extend((x ** 2 + y ** 2) ** 0.5)

    def on_sample_end(self, *, worker, samples, **kwargs):
        print("mean. distance from origin={}".format(np.mean(self.deltas)))
        self.deltas = []


class OneHotWrapper(gym.core.ObservationWrapper):
    def __init__(self, env, vector_index, framestack):
        super().__init__(env)
        self.framestack = framestack
        # 49=7x7 field of vision; 11=object types; 6=colors; 3=state types.
        # +4: Direction.
        self.single_frame_dim = 49 * (11 + 6 + 3) + 4
        self.init_x = None
        self.init_y = None
        self.x_positions = []
        self.y_positions = []
        self.x_y_delta_buffer = deque(maxlen=100)
        self.vector_index = vector_index
        self.frame_buffer = deque(maxlen=self.framestack)
        for _ in range(self.framestack):
            self.frame_buffer.append(np.zeros((self.single_frame_dim,)))

        self.observation_space = gym.spaces.Box(
            0.0, 1.0, shape=(self.single_frame_dim * self.framestack,), dtype=np.float32
        )

    def observation(self, obs):
        # Debug output: max-x/y positions to watch exploration progress.
        if self.step_count == 0:
            for _ in range(self.framestack):
                self.frame_buffer.append(np.zeros((self.single_frame_dim,)))
            if self.vector_index == 0:
                if self.x_positions:
                    max_diff = max(
                        np.sqrt(
                            (np.array(self.x_positions) - self.init_x) ** 2
                            + (np.array(self.y_positions) - self.init_y) ** 2
                        )
                    )
                    self.x_y_delta_buffer.append(max_diff)
                    # print(
                    #     "100-average dist travelled={}".format(
                    #         np.mean(self.x_y_delta_buffer)
                    #     )
                    # )
                    self.x_positions = []
                    self.y_positions = []
                self.init_x = self.agent_pos[0]
                self.init_y = self.agent_pos[1]

        # Are we carrying the key?
        # if self.carrying is not None:
        #    print("Carrying KEY!!")

        self.x_positions.append(self.agent_pos[0])
        self.y_positions.append(self.agent_pos[1])

        # One-hot the last dim into 11, 6, 3 one-hot vectors, then flatten.
        objects = one_hot(obs[:, :, 0], depth=11)
        colors = one_hot(obs[:, :, 1], depth=6)
        states = one_hot(obs[:, :, 2], depth=3)
        # Is the door we see open?
        # for x in range(7):
        #    for y in range(7):
        #        if objects[x, y, 4] == 1.0 and states[x, y, 0] == 1.0:
        #            print("Door OPEN!!")

        all_ = np.concatenate([objects, colors, states], -1)
        all_flat = np.reshape(all_, (-1,))
        direction = one_hot(np.array(self.agent_dir),
                            depth=4).astype(np.float32)
        single_frame = np.concatenate([all_flat, direction])
        self.frame_buffer.append(single_frame)
        return np.concatenate(self.frame_buffer)

def env_maker(config):
    name = config.get("name")
    framestack = config.get("framestack", 4)
    env = gym.make(name)
    # Only use image portion of observation (discard goal and direction).
    env = gym_minigrid.wrappers.ImgObsWrapper(env)
    env = OneHotWrapper(
        env,
        config.vector_index if hasattr(config, "vector_index") else 0,
        framestack=framestack,
    )
    return env

def test_curiosity_on_partially_observable_domain(train=True):
    config = ppo.DEFAULT_CONFIG.copy()
    config["env"] = "mini-grid"
    config["env_config"] = {
        # Also works with:
        # - MiniGrid-MultiRoom-N4-S5-v0
        # - MiniGrid-MultiRoom-N2-S4-v0
        # - MiniGrid-Dynamic-Obstacles-16x16-v0
        "name": "MiniGrid-Dynamic-Obstacles-16x16-v0",
        "framestack": 1,  # seems to work even w/o framestacking
    }
    config["horizon"] = 500  # Make it impossible to reach goal by chance. # also known as batch size is n_steps * n_env
    config["num_envs_per_worker"] = 9
    config['lambda'] = 0.999
    config["model"]["fcnet_hiddens"] = [256, 256]
    config["model"]["fcnet_activation"] = "relu"
    config["num_sgd_iter"] = 100
    config["sgd_minibatch_size"] = 512
    config["num_workers"] = 0 # only 0 works for curiosity
    config["num_gpus"] = 1
    # config["framework"] = "torch"
    config["log_level"] = "ERROR"
    config['output'] = "./outputs"
    config["exploration_config"] = {
        "type": "Curiosity",
        # For the feature NN, use a non-LSTM fcnet (same as the one
        # in the policy model).
        "eta": 0.1,
        "lr": 0.0005,  # 0.0003 or 0.0005 seem to work fine as well.
        "feature_dim": 128,
        # No actual feature net: map directly from observations to feature
        # vector (linearly).
        "feature_net_config": {
            "fcnet_hiddens": [],
            "fcnet_activation": "relu",
        },
        "sub_exploration": {
            "type": "StochasticSampling",
        },
    }

    episode_reward_mean = 0.9
    stop = {
        "training_iteration": 50,
        "episode_reward_mean": episode_reward_mean,
    }
    checkpoint_path = str(Path.home()) + "/ray_sync/PPO/PPO_mini-grid_cdc9a_00000_0_2022-04-04_13-19-45/checkpoint_000011/checkpoint-11"
    if train:
        analysis = tune.run("PPO",
                            config=config,
                            stop=stop,
                            verbose=0,
                            local_dir= str(Path.home()) + "/ray_sync",

                            # checkpoints mode-metric
                            # checkpoint_score_attr="max-episode_reward_mean",
                            # keep_checkpoints_num=2,
                            checkpoint_freq=1,
                            checkpoint_at_end=True,
                            # a very useful trick! this will resume from the last run specified by
                            # sync_config (if one exists), otherwise it will start a new tuning run
                            # resume="AUTO",
                            sync_config=tune.SyncConfig(),
                            # resources_per_trial={"cpu": 1},
                            callbacks=[WandbLoggerCallback(
                                project="mini_grid_rllib",
                                group="Curiosity_Obstacles_16_02",
                                excludes=["perf"],
                                log_config=False)]
                                )
                                
        check_learning_achieved(analysis, "min_reward")
        best_config = analysis.get_best_config(metric="episode_reward_mean", mode="max")
        print("Best Config:", best_config)
        iters = analysis.trials[0].last_result["training_iteration"]
        print("Reached in {} iterations.".format(iters))
        best_trial = analysis.get_best_trial('episode_reward_mean', mode="max")
        checkpoints = analysis.get_trial_checkpoints_paths(trial=best_trial, metric='episode_reward_mean')
        checkpoint_path = checkpoints[0][0]
        print("checkpoint_path", checkpoint_path)

    if checkpoint_path:
        trainer = ppo.PPOTrainer(config=config)
        trainer.restore(checkpoint_path)
        env = env_maker(config["env_config"])

        filename = create_policy_eval_video(env, trainer)
        print(filename, "created.")
    # config_wo = config.copy()
    # config_wo["exploration_config"] = {"type": "StochasticSampling"}
    # stop_wo = stop.copy()
    # stop_wo["training_iteration"] = iters
    # analysis = tune.run(
    #     "PPO", config=config_wo, stop=stop_wo, verbose=1)
    # try:
    #     check_learning_achieved(analysis, min_reward)
    # except ValueError:
    #     print("Did not learn w/o curiosity (expected).")
    # else:
    #     raise ValueError("Learnt w/o curiosity (not expected)!")

def create_policy_eval_video(eval_env, policy_model, filename="eval_video", num_episodes=1000, fps=30):
    filename = filename + ".mp4"
    with imageio.get_writer(filename, fps=fps) as video:
        obs = eval_env.reset()
        for _ in range(num_episodes):
            obs, reward, done, info = eval_env.step(policy_model.compute_single_action(obs))
            im = eval_env.render(mode="rgb_array")
            video.append_data(im)
            if done:
                obs = eval_env.reset()
                break
        eval_env.close()
    return filename


register_env("mini-grid", env_maker)
CONV_FILTERS = [[16, [11, 11], 3], [32, [9, 9], 3], [64, [5, 5], 3]]


ray.init(num_gpus=1, ignore_reinit_error=True, log_to_driver=False)

test_curiosity_on_partially_observable_domain(train=False)
ray.shutdown()
