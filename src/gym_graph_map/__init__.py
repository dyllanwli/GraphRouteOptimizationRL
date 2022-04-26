
from gym.envs.registration import register

register(
        id = "graph-map-v0",
        entry_point="gym_graph_map.envs:GraphMapEnv"
        )
register(
        id = "graph-map-v1",
        entry_point="gym_graph_map.envs:GraphMapEnvV2"
        )