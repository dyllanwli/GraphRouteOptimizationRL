from collections import defaultdict
from typing import Tuple

import numpy as np

import osmnx as ox
from osmnx import distance

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import networkx as nx
from soupsieve import closest

def set_neg_weights(G, df):
    return ox.distance.nearest_nodes(
        G, X=df.loc[0, "Longitude"], Y=df.loc[0, "Latitude"])


class GraphMapEnv(gym.Env):
    """
    Custom Environment that follows gym interface
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, networkx_graph: nx.MultiDiGraph, origin, goal, neg_df, center_node: Tuple, verbose=False) -> None:
        """
        Initializes the environment
        origin: the origin node (151820557)
        goal: the goal node (151820557)
        neg_weights: name of negative weights
        """
        super(gym.Env, self).__init__()
        self.graph = networkx_graph
        self.origin = origin
        self.goal = goal
        self.verbose = verbose
        self.neg_points = self.get_neg_points(neg_df, center_node)
        self.avoid_threshold = 500.0
        self.ep_length = self.graph.number_of_nodes() * 2
        # Constant values

        self.seed()
        self.reset(networkx_graph, origin, goal)
        # Reset the environment

    def reindex_graph(self, graph):
        """
        Reindexes the graph
        """
        self.reindexed_graph = nx.relabel.convert_node_labels_to_integers(
            graph, first_label=0, ordering='default')

        self.node_dict = {node: i for i,
                          node in enumerate(graph.nodes(data=False))}
        # node_dict: {151820557: 0}

        self.node_dict_reversed = {
            i: node for node, i in self.node_dict.items()}
        # node_dict_reversed: {0: 151820557}

        self.nodes = self.reindexed_graph.nodes()

    def reset(self):
        """
        Resets the environment
        """
        self.reindex_graph(self.graph)
        self.current = self.node_dict[self.origin]
        self.current_step = 0
        self.done = False
        # Reset values

        self.adj_shape = self.graph.number_of_nodes()

        self.action_space = spaces.Discrete(
            self.reindexed_graph.number_of_nodes())

        self.observation_space = spaces.Dict({
            "current": spaces.Discrete(self.reindexed_graph.number_of_nodes()),
            "adj": spaces.Box(low=0, high=1, shape=(self.adj_shape, self.adj_shape), dtype=np.float32),
            "length": spaces.Box(low=0, high=float("inf"), shape=(self.adj_shape, self.adj_shape), dtype=np.float32),
            "speed_kph": spaces.Box(low=0, high=120, shape=(self.adj_shape, self.adj_shape), dtype=np.float32),
            "travel_time": spaces.Box(low=0, high=float("inf"), shape=(self.adj_shape, self.adj_shape), dtype=np.float32),
        })
        self._update_state()
        return self.state

    def action_mask(self):
        """
        Computes the action mask
        Returns:
            action_mask: [1, 0, ...]
        """
        neighbors = self.reindexed_graph.neighbors(self.current)
        mask = np.isin(self.reindexed_graph.nodes(), neighbors,
                       assume_unique=True).astype(int)
        return mask

    # @property
    # def action_space(self):
    #   return self.graph.neighbors(self.current)

    def _update_state(self):
        self.state = spaces.Dict({
            "current": self.current,
            "adj": nx.adjacency_matrix(self.reindexed_graph, weight=None),
            "length": nx.adjacency_matrix(self.reindexed_graph, weight="length"),
            "speed_kph": nx.adjacency_matrix(self.reindexed_graph, weight="speed_kph"),
            "travel_time": nx.adjacency_matrix(self.reindexed_graph, weight="travel_time"),
        })

    def step(self, action):
        """
        Executes one time step within the environment
        """
        self.current = action
        self.current_step += 1
        self.reward = self._reward()
        self._update_state()
        if self.current == self.node_dict[self.goal] or self.current_step >= self.ep_length:
            self.done = True
        return self.state, self.reward, self.done, {"current_step": self.current_step}

    def _great_circle_vec(self, node1, node2):
        """
        Computes the euclidean distance between two nodes
        Input:
            node1: (lat, lon)
            node2: (lat, lon)
        Returns:
            distance: float (meters)
        """
        x1, y1 = node1
        x2, y2 = node2
        return distance.great_circle_vec(x1, y1, x2, y2)

    def _reward(self):
        """
        Computes the reward
        neg_factor range: [-inf, 1.0]
        """
        current_node = (self.nodes[self.current]['y'], self.nodes[self.current]['x'])
        neg_factor = 1.0
        closest_dist = 0.0
        for node in self.neg_points:
            dist = self._great_circle_vec(current_node, node)
            if dist <= self.avoid_threshold:
                closest_dist = min(closest_dist, dist)
        # too close to a negative point will cause a negative reward
        neg_factor = 1 + np.log(closest_dist / self.avoid_threshold)

        return neg_factor * (1 - 0.9 * 1)

    def render(self, mode='human', close=False):
        """
        Renders the environment
        """
        fig, ax = ox.plot_graph_route(self.graph.ngraph, self.graph.path)
        return np.array(fig.canvas.buffer_rgba())

    def seed(self, seed=None):
        """
        Seeds the environment
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self) -> None:
        return super().close()

    def get_neg_points(self, df, center_node, threshold = 2000.0):
        """
        Computes the negative weights
        Inputs: 
            df: pandas dataframe with the following columns: ['Longitude', 'Latitude', 'neg_weights', ...]
            center_node: (29.764050, -95.393030) # buffalo bayou park
            threshold: the threshold for the negative weights (meters)
        Returns:
            neg_weights: {(Longitude, Longitude): weight}
        """
        neg_list = defaultdict(float)
        for x in df.values:
            neg_list[(x[0], x[1])] = x[2]
        neg_nodes = []

        for node in neg_list.keys():
            # caculate the distance between the node and the center node
            dist = self._great_circle_vec(node, center_node)
            if dist < threshold:
                neg_nodes.append(node)

        return neg_nodes

