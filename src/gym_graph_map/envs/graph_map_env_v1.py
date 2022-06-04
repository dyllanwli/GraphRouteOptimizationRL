# from collections import defaultdict
from typing import Tuple

from pathlib import Path

import numpy as np
import networkx as nx

# for mapping
import osmnx as ox
from osmnx import distance
import pandas as pd

# for plotting
import geopandas
import matplotlib.pyplot as plt

# for reinforcement learning environment
import gym
from gym import error, spaces, utils
from gym.utils import seeding

INF = 100000000
home = str(Path.home())


class GraphMapEnvV1(gym.Env):
    """
    Custom Environment that follows gym interface
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, config) -> None:
        """
        Initializes the environment
        config: {
            'graph': graph,
            'verbose': True,
            'neg_df': neg_df,
            'center_node': (29.764050, -95.393030),
            'threshold': 2900,
        }
        """
        super(gym.Env, self).__init__()

        self._set_config(config)

        self._set_utility_function()

        self.seed(1)
        self._reindex_graph(self.graph)

        self.adj_shape = (self.number_of_nodes, self.number_of_nodes)
        self.action_space = spaces.Discrete(
            self.number_of_nodes,)

        self.observation_space = spaces.Dict({
            "current": spaces.Discrete(self.action_space.n),
            "action_mask": spaces.Box(low=0, high=1, shape=(self.action_space.n,), dtype=np.int64),
            # "adj": spaces.Box(low=0, high=float("inf"), shape=self.adj_shape, dtype=np.float32),
            # "length": spaces.Box(low=0, high=float("inf"), shape=self.adj_shape, dtype=np.float32),
            # "speed_kph": spaces.Box(low=0, high=float("inf"), shape=self.adj_shape, dtype=np.float32),
            # "travel_time": spaces.Box(low=0, high=float("inf"), shape=self.adj_shape, dtype=np.float32),
        })
        # self.reset()
        # Reset the environment

        print("EP_LENGTH:", self.EP_LENGTH)
        print("action_space:", self.action_space)
        print("num of neg_points:", len(self.neg_points))
        # print("origin:", self.origin)
        # print("goal:", self.goal)

    def _set_config(self, config):
        """
        Sets the config
        """
        # Constant values
        self.graph = config['graph']
        self.verbose = config['verbose'] if config['verbose'] else False
        self.neg_df = config['neg_df']
        self.center_node = config['center_node']
        self.threshold = config['threshold']
        # graph radius or average short path in this graph, sampled by env stat
        self.threshold = 2900
        self.neg_points = self._get_neg_points(self.neg_df, self.center_node)

        if self.neg_points == []:
            raise ValueError("Negative weights not found")

        self.avoid_threshold = self.threshold / 4
        self.number_of_nodes = self.graph.number_of_nodes()
        self.EP_LENGTH = self.graph.number_of_nodes()/2  # based on the stats of the graph

        # utility values
        self.render_img_path = home + "/dev/GraphRouteOptimizationRL/images/render_img.png"

    def _set_utility_function(self):
        """
        Sets the utility function
        """
        self.sigmoid1 = lambda x: 1 / (1 + np.exp(-x/self.threshold))

        parameters = "%e" % self.threshold
        sp = parameters.split("e")
        a = -float(sp[0])
        b = 10**(-float(sp[1]))
        self.sigmoid2 = lambda x, a=a, b=b: 1 / (1 + np.exp(a + b * x))

    def _reindex_graph(self, graph):
        """
        Reindexes the graph
        node_dict: {151820557: 0}
        node_dict_reversed: {0: 151820557}
        """
        self.reindexed_graph = nx.relabel.convert_node_labels_to_integers(
            graph, first_label=0, ordering='default')

        self.node_dict = {node: i for i,
                          node in enumerate(graph.nodes(data=False))}

        self.node_dict_reversed = {
            i: node for node, i in self.node_dict.items()}

        self.nodes = self.reindexed_graph.nodes()

    def _update_state(self):
        """
        Updates: 
            self.neighbors
            self.action_space
            self.state
        """
        self.neighbors = list(self.reindexed_graph.neighbors(self.current))
        if self.verbose:
            print(self.current, "'s neighbors: ", self.neighbors)
        self.state = {
            "current": np.int64(self.current),
            "action_mask": self.action_masks(),
            # "adj": nx.to_numpy_array(self.reindexed_graph, weight="None", dtype=np.float32),
            # "length": nx.to_numpy_array(self.reindexed_graph, weight="length", dtype=np.float32),
            # "speed_kph": nx.to_numpy_array(self.reindexed_graph, weight="speed_kph", dtype=np.float32),
            # "travel_time": nx.to_numpy_array(self.reindexed_graph, weight="travel_time", dtype=np.float32),
        }

    def _get_neg_points(self, df, center_node):
        """
        Computes the negative weights
        Inputs: 
            df: pandas dataframe with the following columns: ['Latitude', 'Longitude', 'neg_weights', ...]
            center_node: (29.764050, -95.393030) # buffalo bayou park
            threshold: the threshold for the negative weights (meters)
        Returns:
            neg_nodes: {'x': Latitude, 'y': Longitude, 'weight': neg_weights}
        """
        neg_nodes = []
        center_node = {'x': center_node[0], 'y': center_node[1]}
        self.df = pd.DataFrame(columns=df.columns)
        index = 0
        for _, row in df.iterrows():
            node = {'x': row[0], 'y': row[1], 'weight': row[2]}
            dist = self._great_circle_vec(center_node, node)
            # caculate the distance between the node and the center node
            if dist <= self.threshold:
                neg_nodes.append(node)
                self.df.loc[index] = row
                index += 1
        return neg_nodes

    def _great_circle_vec(self, node1, node2):
        """
        Computes the euclidean distance between two nodes
        Input:
            node1: (lat, lon)
            node2: (lat, lon)
        Returns:
            distance: float (meters)
        """
        x1, y1 = node1['x'], node1['y']
        x2, y2 = node2['x'], node2['y']
        return distance.great_circle_vec(x1, y1, x2, y2)

    def _update_attributes(self):
        """
        Updates the path length and travel time
        """
        self.path_length += ox.utils_graph.get_route_edge_attributes(
            self.reindexed_graph, self.path[-2:], "length")[0]

        self.travel_time += ox.utils_graph.get_route_edge_attributes(
            self.reindexed_graph, self.path[-2:], "travel_time")[0]

    def _reward(self):
        """
        Computes the reward
        """
        neg_factor = 1.0
        closest_dist = self.avoid_threshold
        for node in self.neg_points:
            dist = self._great_circle_vec(self.current_node, node)
            if dist <= self.avoid_threshold:
                closest_dist = min(closest_dist, dist)
        # too close to a negative point will cause a negative reward
        neg_factor = max(-INF, 1 + np.log(closest_dist / self.avoid_threshold))
        # r1 = np.log(- self.current_step + self.EP_LENGTH + 1) - 2
        r2 = 1.0 if self.current == self.goal else 0.0
        r3 = r2 * self.sigmoid2(self.path_length)

        # r4 = np.log2(- (self.travel_time /
        #              self.nx_shortest_travel_time_length) + 2) + 1
        r5 = self.sigmoid1(self._great_circle_vec(
            self.current_node, self.goal_node))
        r = np.mean([r2, r3, r5]) * neg_factor
        return r

    def step(self, action):
        """
        Executes one time step within the environment
        self.current: the current node
        self.current_step: the current step
        self.done: whether the episode is done:
            True: the episode is done OR the current node is the goal node OR the current node is a dead end node
        self.state: the current state
        self.reward: the reward
        Returns:
            self.state: the next state
            self.reward: the reward
            self.done: whether the episode is done
            self.info: the information
        """
        self.current = action
        self.current_node = self.nodes[self.current]
        self.current_step += 1
        self.path.append(self.current)
        if self.verbose:
            print("self.path:", self.path)

        self._update_state()
        if self.current == self.goal or self.current_step >= self.EP_LENGTH or self.neighbors == []:
            self.done = True
        self._update_attributes()
        self.reward = self._reward()
        return self.state, self.reward, self.done, self.info

    def action_space_sample(self):
        """
        Samples an action from the action space
        """
        if self.neighbors == []:
            return self.action_space.sample()
        else:
            return np.int64(np.random.choice(self.neighbors))

    def get_default_route(self):
        """
        Set the default route by tratitional method
        """
        try:
            self.nx_shortest_path = nx.shortest_path(
                self.reindexed_graph, source=self.origin, target=self.goal, weight="length")
            self.nx_shortest_path_length = sum(ox.utils_graph.get_route_edge_attributes(
                self.reindexed_graph, self.nx_shortest_path, "length"))

            self.nx_shortest_travel_time = nx.shortest_path(
                self.reindexed_graph, source=self.origin, target=self.goal, weight="travel_time")
            self.nx_shortest_travel_time_length = sum(ox.utils_graph.get_route_edge_attributes(
                self.reindexed_graph, self.nx_shortest_travel_time, "travel_time"))
        except nx.exception.NetworkXNoPath:
            if self.verbose:
                print("No path found for default route. Restting...")
            self.reset()

    def reset(self):
        """
        Resets the environment
        """

        self.origin = self.action_space.sample()
        self.goal = self.action_space.sample()
        self.goal_node = self.nodes[self.goal]

        self.current = self.origin
        self.current_node = self.nodes[self.current]
        self.current_step = 0
        self.done = False
        self.path = [self.current]
        self.path_length = 0.0
        self.travel_time = 0.0
        self.neighbors = []
        self.info = {}

        self._update_state()
        if self.neighbors == []:
            # make sure the first step is not a dead end node
            self.reset()
        return self.state

    def action_masks(self):
        """
        Computes the action mask
        Returns:
            action_mask: [1, 0, ...]
        """
        self.mask = np.isin(self.nodes, self.neighbors,
                            assume_unique=True).astype(np.int64)
        return self.mask

    def render(self, mode='human', default_path=None, plot_learned=True, plot_neg=True, save=True):
        """
        Renders the environment
        """
        if self.verbose:
            print("Get path", self.path)
        if default_path is not None:
            ox.plot_graph_route(self.reindexed_graph, default_path,
                                save=save, filepath=home + "/dev/GraphRouteOptimizationRL/images/default_image.png")
        if plot_learned:
            save = False if plot_neg else save
            fig, ax = ox.plot_graph_route(
                self.reindexed_graph, self.path, save=False, filepath=self.render_img_path, show=False, close=False)
            if plot_neg:
                gdf = geopandas.GeoDataFrame(
                    self.df, geometry=geopandas.points_from_xy(self.df['Longitude'], self.df['Latitude']))
                gdf.plot(ax=ax, markersize=10, color="blue", alpha=1, zorder=7)
                plt.savefig(self.render_img_path)
                plt.show()
                plt.close(fig)

        if mode == 'human':
            pass
        else:
            return np.array(fig.canvas.buffer_rgba())

    def seed(self, seed=None):
        """
        Seeds the environment
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self) -> None:
        return super().close()


# class MaskedDiscreteAction(spaces.Discrete):
#     def __init__(self, n):
#         super().__init__(n)
#         self.neighbors = None

#     def super_sample(self):
#         return np.int64(super().sample())

#     def sample(self):
#         # The type need to be the same as Discrete
#         return np.int64(np.random.choice(self.neighbors))
