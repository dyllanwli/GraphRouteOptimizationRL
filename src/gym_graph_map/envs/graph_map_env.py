# from collections import defaultdict
from typing import Tuple

import numpy as np
import networkx as nx

# for mapping
import osmnx as ox
from osmnx import distance

# for plotting
import geopandas
import matplotlib.pyplot as plt

# for reinforcement learning environment
import gym
from gym import error, spaces, utils
from gym.utils import seeding

INF = 100000000


class MaskedDiscreteAction(spaces.Discrete):
    def __init__(self, n):
        super().__init__(n)
        self.neighbors = None

    def super_sample(self):
        return int(super().sample())

    def sample(self):
        # The type need to be the same as Discrete
        return int(np.random.choice(self.neighbors))


class GraphMapEnv(gym.Env):
    """
    Custom Environment that follows gym interface
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, networkx_graph: nx.MultiDiGraph, neg_df, center_node: Tuple = (29.764050, -95.393030), verbose=False) -> None:
        """
        Initializes the environment
        origin: the origin node (151820557)
        goal: the goal node (151820557)
        neg_weights: name of negative weights
        center_node: the center node (29.764050, -95.393030)
        """
        super(gym.Env, self).__init__()
        self.graph = networkx_graph
        self.verbose = verbose
        # graph radius or average short path in this graph, sampled by env stat
        self.threshold = 2900
        self.neg_points = self._get_neg_points(neg_df, center_node)
        if self.neg_points == []:
            raise ValueError("Negative weights not found")
        self.avoid_threshold = self.threshold / 4
        self.number_of_nodes = self.graph.number_of_nodes()
        self.EP_LENGTH = 100
        # Constant values

        # utility values
        self.render_img_path = "./render_img.png"

        self._set_utility_function()

        self.seed(1)
        self._reindex_graph(self.graph)
        self.reset()
        # Reset the environment

        print("EP_LENGTH:", self.EP_LENGTH)
        print("origin:", self.origin)
        print("goal:", self.goal)
        print("num of neg_points:", len(self.neg_points))
        print("action_space:", self.action_space)

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
        Updates the state
        TODO: update dynamical state here
        """
        self.state['current'] = self.current
        self.neighbors = list(self.reindexed_graph.neighbors(self.current))
        if self.verbose:
            print(self.current, "'s neighbors: ", self.neighbors)
        self.action_space.neighbors = self.neighbors

    def _reward(self):
        """
        Computes the reward
        neg_factor range: [-inf, 1.0] the closer to the negative point, the less the overall reward
        r1 range: [0, 1.0] the more the steps, the less the r1, this effect will be even more stronger when the steps are close to EP_LENGTH
        r2 range: {0.0, 1.0} if reached the goal, the r2 is 1.0 else 0.0
        r3 range: [0.0, 1.0] if reached the goal, the r3 will caculate the path length with sigmoid function, 
            the sigmod funtion is constructed by the self.threshold, if the path length is less than the threshold, the r3 is less than 0.5 else greater than 0.5
        TODO: add r4
        r5 range: [0, 1.0] the closer to the goal the higher the r5 for each step
        """
        current_node = self.nodes[self.current]
        neg_factor = 1.0
        closest_dist = self.avoid_threshold
        for node in self.neg_points:
            dist = self._great_circle_vec(current_node, node)
            if dist <= self.avoid_threshold:
                closest_dist = min(closest_dist, dist)
        # too close to a negative point will cause a negative reward
        neg_factor = max(-INF, 1 + np.log(closest_dist / self.avoid_threshold))
        r1 = np.log(- self.current_step + self.EP_LENGTH + 1) - 2
        r2 = 1.0 if self.state['current'] == self.goal else 0.0
        r3 = r2 * self.sigmoid2(self.path_length)

        # r4 = np.log2(- (self.travel_time /
        #              self.nx_shortest_travel_time_length) + 2) + 1
        r5 = self.sigmoid1(self._great_circle_vec(
            current_node, self.goal_node))
        r = np.mean([r1, r2, r3, r5]) * neg_factor
        return r

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

        def condition(row):
            node = {'x': row[0], 'y': row[1], 'weight': row[2]}
            dist = self._great_circle_vec(center_node, node)
            # caculate the distance between the node and the center node
            if dist <= self.threshold:
                neg_nodes.append(node)
                return row
        self.df = df.apply(condition, axis=1)
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

    def _update_path_length(self):
        """
        Updates the path length
        """
        self.path_length += ox.utils_graph.get_route_edge_attributes(
            self.reindexed_graph, self.path[-2:], "length")[0]

    def _update_travel_time(self):
        """
        Updates the travel time
        """
        self.travel_time += ox.utils_graph.get_route_edge_attributes(
            self.reindexed_graph, self.path[-2:], "travel_time")[0]

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

        self.adj_shape = (self.number_of_nodes, self.number_of_nodes)
        self.action_space = MaskedDiscreteAction(
            self.number_of_nodes,)

        self.observation_space = spaces.Dict({
            "current": spaces.Discrete(self.number_of_nodes),
            "adj": spaces.Box(low=0, high=float("inf"), shape=self.adj_shape, dtype=np.float32),
            # "length": spaces.Box(low=0, high=float("inf"), shape=self.adj_shape, dtype=np.float32),
            # "speed_kph": spaces.Box(low=0, high=float("inf"), shape=self.adj_shape, dtype=np.float32),
            # "travel_time": spaces.Box(low=0, high=float("inf"), shape=self.adj_shape, dtype=np.float32),
        })

        self.origin = self.action_space.super_sample()
        self.goal = self.action_space.super_sample()
        self.goal_node = self.nodes[self.goal]

        self.current = self.origin
        self.current_step = 0
        self.done = False
        self.path = [self.current]
        self.path_length = 0.0
        self.travel_time = 0.0
        self.neighbors = []
        self.info = {}

        self.state = {
            "current": self.current,
            "adj": nx.to_numpy_array(self.reindexed_graph, weight="None", dtype=np.float32),
            # "length": nx.to_numpy_array(self.reindexed_graph, weight="length", dtype=np.float32),
            # "speed_kph": nx.to_numpy_array(self.reindexed_graph, weight="speed_kph", dtype=np.float32),
            # "travel_time": nx.to_numpy_array(self.reindexed_graph, weight="travel_time", dtype=np.float32),
        }

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
                            assume_unique=True).astype(int)
        return self.mask

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
        self.current_step += 1
        self.path.append(self.current)
        if self.verbose: 
            print("self.path:", self.path)

        self._update_state()
        if self.current == self.goal or self.current_step >= self.EP_LENGTH or self.neighbors == []:
            self.done = True
        self._update_path_length()
        self._update_travel_time()
        self.reward = self._reward()
        return self.state, self.reward, self.done, self.info

    def render(self, mode='human', default_path=None, plot_learned=True, plot_neg=True, save=True):
        """
        Renders the environment
        """
        if self.verbose:
            print("Get path", self.path)
        if default_path is not None:
            ox.plot_graph_route(self.reindexed_graph, default_path,
                                save=save, filepath="./default_image.png")
        if plot_learned:
            fig, ax = ox.plot_graph_route(
                self.reindexed_graph, self.path, save=save, filepath=self.render_img_path)
            if plot_neg:
                gdf = geopandas.GeoDataFrame(
                    self.df, geometry=geopandas.points_from_xy(self.df.Longitude, self.df.Latitude))
                gdf.plot(ax=ax, markersize = 1, color="blue" , alpha=1, zorder=7)
                plt.savefig(self.render_img_path)
                plt.show()

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
