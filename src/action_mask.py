from sb3_contrib import MaskablePPO
from sb3_contrib.common.envs import InvalidActionEnvDiscrete
from gym_graph_map.envs import GraphMapEnv
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks

import osmnx as ox


def main():
    graph_path = "/h/diya.li/dev/GraphRouteOptimizationRL/houston_tx_usa_drive_500.graphml"
    G = ox.load_graphml(graph_path)
    origin = 0
    goal = 1
    env = GraphMapEnv(G, origin, goal)
    model = MaskablePPO("MlpPolicy", env, gamma=0.4, seed=32, verbose=1)
    model.learn(5000)

    evaluate_policy(model, env, n_eval_episodes=20, reward_threshold=90, warn=False)

    model.save("ppo_mask")
    del model # remove to demonstrate saving and loading

    model = MaskablePPO.load("ppo_mask")

    obs = env.reset()
    while True:
        # Retrieve current action mask
        action_masks = get_action_masks(env)
        action, _states = model.predict(obs, action_masks=action_masks)
        obs, rewards, dones, info = env.step(action)
        env.render()

if __name__ == "__main__":
    main()