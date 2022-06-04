
from gym.envs.registration import register

register(
        id = "graph-map-v1",
        entry_point="gym_graph_map.envs:GraphMapEnvV1"
        )
register(
        id = "graph-map-v2",
        entry_point="gym_graph_map.envs:GraphMapEnvV2"
        )

register(
        id = "graph-map-v3",
        entry_point="gym_graph_map.envs:GraphMapEnvV3"
        )
register(
        id = "graph-map-v4",
        entry_point="gym_graph_map.envs:GraphMapEnvV4"
        )
register(
        id = "graph-map-v5",
        entry_point="gym_graph_map.envs:GraphMapEnvV5"
        )
register(
        id = "graph-map-v6",
        entry_point="gym_graph_map.envs:GraphMapEnvV6"
        )
register(
        id = "graph-map-v7",
        entry_point="gym_graph_map.envs:GraphMapEnvV7"
        )