# from collections import defaultdict
"""
Version 4 of the Graph Map Environment with the following changes:
1. The environment now has a config dict that can be passed to the constructor.
2. The environment now has a reset_config method that can be called to reset the
    config dict to its default values.
3. The environment now has a new state as a sequence as observation space so the model won't have to learn to
    handle the wrapper stuff.
4. The environment now has a envolving environment based on a marsh attribute and elevation attribute.
5. The environment now has a a better render mode.
"""
from copy import deepcopy
from pathlib import Path

import numpy as np
import networkx as nx

# for mapping
import osmnx as ox
from osmnx import distance
import pandas as pd
import wandb
# for plotting
import matplotlib.pyplot as plt

# for reinforcement learning environment
import gym
from gym import spaces
from gym.utils import seeding

repo_path = str(Path.home()) + "/dev/GraphRouteOptimizationRL/"


class GraphMapEnvV2(gym.Env):
    """
    Custom Environment that follows gym interface
    V2 is the version that uses the compatible with rllib
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, config) -> None:
        """
        Initializes the environment
        config: {
            'graph': graph,
            'verbose': True,
            'neg_df_path': neg_df_path,
            'center_node': (29.764050, -95.393030),
            'threshold': 2900,
        }
        """
        self._skip_env_checking = True
        super(gym.Env, self).__init__()

        self._set_config(config)

        self.seed(1)
        self._reindex_graph(self.graph)

        self._set_action_observation()

        self.reset()
        self._set_utility_function()
        # Reset the environment

        print("action_space:", self.action_space)
        print("num of neg_points:", len(self.neg_points))
        print("origin:", self.origin)
        print("goal:", self.goal)
        # print("origin_goal_distance", self.origin_goal_distance)

    def _set_action_observation(self):
        """
        Sets the observation
            action_space:
                type: Discrete
                shape: (number_of_nodes)
            observation_space:
                type: ndarray
                shape: (self.embedding_dim, 3)
                first line is the state
                second line is the reference
        """

        self.adj_shape = (self.number_of_nodes, self.number_of_nodes)
        self.action_space = spaces.Discrete(
            self.number_of_nodes)

        # set observation space low
        state_low = np.zeros(8)
        assert self.embedding_dim == 8

        current_embedding_low = np.full(
            shape=(self.embedding_dim,), fill_value=-np.inf)

        diff_embedding_low = np.full(
            shape=(self.embedding_dim,), fill_value=-np.inf)
        # nx_path_references_low = np.zeros(self.number_of_nodes)
        nx_path_embedding_low = np.full(
            shape=(self.embedding_dim,), fill_value=-np.inf)

        current_path_embedding_low = np.full(
            shape=(self.embedding_dim,), fill_value=-np.inf)

        obs_low = np.hstack((
            state_low,
            current_embedding_low,
            diff_embedding_low,
            nx_path_embedding_low,
            current_path_embedding_low,
        ))

        # set observation space high
        state_high = np.full(shape=(8,), fill_value=np.inf)

        current_embedding_high = np.full(
            shape=(self.embedding_dim,), fill_value=np.inf)

        diff_embedding_high = np.full(
            shape=(self.embedding_dim,), fill_value=np.inf)

        # nx_path_referenes_high = np.full(
        #     shape=(self.number_of_nodes, ), fill_value=self.number_of_nodes)

        nx_path_embedding_high = np.full(
            shape=(self.embedding_dim,), fill_value=np.inf)

        current_path_embedding_high = np.full(
            shape=(self.embedding_dim,), fill_value=np.inf)

        obs_high = np.hstack((
            state_high,
            current_embedding_high,
            diff_embedding_high,
            nx_path_embedding_high,
            current_path_embedding_high,
        ))

        self.observation_space = spaces.Dict({
            "observations": spaces.Box(low=obs_low, high=obs_high, dtype=np.float32),
            "action_mask": spaces.Box(low=0, high=1, shape=(self.number_of_nodes,), dtype=np.int64),
        })

        # self.adj = nx.to_numpy_array(self.graph, weight="None", dtype=np.float32)

    def _update_state(self):
        """
        Updates:
            self.neighbors
            self.action_space
            self.state
            self.current_distance2goal
            self.current_distance2neg
        Uncomment self.action_space to update the neighbors if use MaskedSpace
        """

        # forward neighbors
        self.neighbors = list(x for x in self.graph.neighbors(
            self.current) if x not in self.passed_nodes_ids)

        # for n in self.neighbors:

        # self.action_space.neighbors = self.neighbors

        # self.neighbors_

        self.current_distance2goal = self._great_circle_vec(
            self.current_node, self.goal_node)

        self.current_distance2neg = max(10, self._get_closest_distance_neg(
            self.current_node))  # make sure log won't be 0

        if self.verbose:
            print(self.current, "'s neighbors: ", self.neighbors)
            print("current_distance_goal:", self.current_distance2goal)
            print("nx_shortest_path_length:", self.nx_shortest_path_length)

        # set state
        state = np.array([
            self.current,
            self.goal,
            self.current_distance2goal,
            self.current_distance2neg,
            self.path_length,
            self.travel_time,
            self.current_step,
            self.nx_shortest_path_length,
        ], dtype=np.float32)

        # set references
        # nx_path_references = np.pad(np.array(self.nx_shortest_path), pad_width=(
        #     0, self.number_of_nodes - len(self.nx_shortest_path)), mode='constant', constant_values=0)
        nx_path_embedding = np.sum(
            [self.embedding[n] for n in np.array(self.nx_shortest_path)], axis=0)

        self.current_path_embedding += self.embedding[self.current]

        self.state = {
            "observations": np.hstack((
                state,
                self.embedding[self.current],
                self.embedding[self.goal] - self.embedding[self.current],
                nx_path_embedding,
                self.current_path_embedding,
            )),
            "action_mask": self.action_masks(),
        }

    def _set_config(self, config):
        """
        Sets the config
        """
        # Constant values
        self.graph = config['graph']
        self.verbose = config['verbose'] if config['verbose'] else False
        center_node = config['center_node']
        self.threshold = config['threshold']
        # graph radius or average short path in this graph, sampled by env stat
        self.threshold = 2900
        self.neg_points_reset = self._get_neg_points(
            pd.read_csv(config['neg_df_path']), center_node)

        self.envolving = config['envolving']
        self.envolving_freq = config['envolving_freq']

        self.avoid_threshold = self.threshold / 7
        self.number_of_nodes = self.graph.number_of_nodes()

        # load embeddings
        self.embedding = np.load(config['embedding_path'])
        assert self.embedding.shape[0] == self.number_of_nodes
        self.embedding_dim = self.embedding.shape[1]

        # utility values
        self.render_img_path = repo_path + "images/render_img.png"

    def _get_neighbors_embeddings(self):
        """
        Returns the embeddings of the neighbors
        """
        pass

    def _set_utility_function(self):
        """
        Sets the utility function (sigmoid)
        e.g.
            self.path_length = 2900.0
            p = 2.9e+03
            a = -2.9
            b = 1e-03 (0.001)
        """
        self.sigmoid1 = lambda x: 2 / (1 + np.exp(x/self.threshold))

        p = "%e" % self.nx_shortest_path_length
        sp = p.split("e")
        a = float(sp[0])
        b = 10**(-float(sp[1]))

        self.sigmoid2 = lambda x, a=a, b=b: 1 / (1 + np.exp(-a + b * x))

        self.tanh = lambda x: np.tanh(a - b * x)

    def _reindex_graph(self, graph):
        """
        Reindexes the graph
        node_dict: {151820557: 0}
        node_dict_reversed: {0: 151820557}
        """
        self.graph = nx.relabel.convert_node_labels_to_integers(
            graph, first_label=0, ordering='default')

        # self.node_dict = {node: i for i, node in enumerate(graph.nodes(data=False))}

        # self.node_dict_reversed = {i: node for node, i in self.node_dict.items()}

        self.nodes = self.graph.nodes()

    def _get_neg_points(self, df, center_node):
        """
        Computes the negative weights
        Inputs:
            df: pandas dataframe with the following columns: ['Latitude', 'Longitude', 'neg_weights', ...]
            center_node: (29.764050, -95.393030) # buffalo bayou park
            threshold: the threshold for the negative weights (meters)
        Returns:
            neg_nodes: {'y': Latitude, 'x': Longitude, 'weight': neg_weights}
        """
        neg_nodes = []
        center_node = {'y': center_node[0], 'x': center_node[1]}
        index = 0

        for _, row in df.iterrows():
            node = {'y': row[0], 'x': row[1], 'weight': row[2]}
            dist = self._great_circle_vec(center_node, node)

            # caculate the distance between the node and the center node
            if dist <= self.threshold:
                neg_nodes.append(node)
                index += 1
        return neg_nodes

    def _great_circle_vec(self, node1, node2):
        """
        Computes the euclidean distance between two nodes
        Input:
            node1: {y: lat, x: lon}
            node2: {y: lat, x: lon}
        Returns:
            distance: float (meters)
        """
        x1, y1 = node1['y'], node1['x']
        x2, y2 = node2['y'], node2['x']
        return distance.great_circle_vec(x1, y1, x2, y2)

    def _update_attributes(self):
        """
        Updates the path length and travel time
        """
        self.path_length += ox.utils_graph.get_route_edge_attributes(
            self.graph, self.path[-2:], "length")[0]

        self.travel_time += ox.utils_graph.get_route_edge_attributes(
            self.graph, self.path[-2:], "travel_time")[0]

    def _get_closest_distance_neg(self, node):
        """
        Computes the closest distance to the negative points
        Input:
            node: (lat, lon);  e.g. self.current_node
        Returns:
            closest_dist: float (meters)
        """
        closest_dist = np.inf
        for neg_node in self.neg_points:
            dist = self._great_circle_vec(node, neg_node)
            if dist < closest_dist:
                closest_dist = dist
        return closest_dist

    def _envolving(self):
        """
        Envolving
        """
        # TODO: choice with elevation
        if self.envolving and self.current_step % self.envolving_freq == 0:
            envolving_node = self.graph.nodes[np.random.choice(
                self.number_of_nodes)]
            node = {'y': envolving_node['y'],
                    'x': envolving_node['x'], 'weight': 5}
            self.neg_points.append(node)

    def _reward(self):
        """
        Computes the reward
        """
        # too close to a negative point will cause a negative reward
        neg = min(
            0, max(-10, np.log2(self.current_distance2neg / self.avoid_threshold)))

        r2 = 1.0 if self.info['arrived'] else 0.0
        # r3 = r2 * self.tanh(self.path_length) # v0
        r3 = r2 * (self.nx_shortest_path_length / self.path_length)  # v1

        if self.current in self.reference_path_set:
            r4 = self.reference_reward
            self.reference_path_set.remove(self.current)
        else:
            r4 = 0.0

        return r2 + r3 + r4 + neg

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
        self.last_current = self.current
        self.current = action
        self.current_node = self.nodes[self.current]
        self.current_step += 1
        self.path.append(self.current)
        self.passed_nodes_ids.add(self.current)

        self._envolving()

        if self.verbose:
            print("self.path:", self.path)

        self._update_attributes()
        self._update_state()

        self.info['arrived'] = self.current == self.goal
        if self.info['arrived'] or self.neighbors == []:
            self.done = True

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
                self.graph, source=self.origin, target=self.goal, weight="length")
            self.nx_shortest_path_length = sum(ox.utils_graph.get_route_edge_attributes(
                self.graph, self.nx_shortest_path, "length"))

            self.nx_shortest_travel_time = nx.shortest_path(
                self.graph, source=self.origin, target=self.goal, weight="travel_time")
            self.nx_shortest_travel_time_length = sum(ox.utils_graph.get_route_edge_attributes(
                self.graph, self.nx_shortest_travel_time, "travel_time"))
        except nx.exception.NetworkXNoPath:
            if self.verbose:
                print("No path found for default route. Restting...")
            self.reset()

    def _check_origin_goal_distance(self):
        """
        Check the distance between origin and goal for test
        """
        self.origin_goal_distance = self._great_circle_vec(
            self.current_node, self.goal_node)

        if self.origin_goal_distance > self.avoid_threshold:
            if self.verbose:
                print("The distance between origin and goal is too close, resetting...")
            self.reset()

    def _reset_render(self):
        """
        Reset the render
        """
        # plotting base map
        self.fig, self.ax = ox.plot_graph(self.graph, show=False, close=False)
        self.origin_node = self.nodes[self.origin]
        self.ax.plot(
            self.origin_node['x'], self.origin_node['y'], c='yellow', marker='o', markersize=5)
        self.ax.plot(self.goal_node['x'], self.goal_node['y'],
                     c='yellow', marker='o', markersize=5)

    def reset(self):
        """
        Resets the environment
        """

        self.origin = self.action_space.sample()
        self.goal = self.action_space.sample()
        self.goal_node = self.nodes[self.goal]

        self.current = self.origin
        self.current_node = self.nodes[self.current]

        self.passed_nodes_ids = {self.current}
        self.current_step = 0
        self.done = False
        self.path = [self.current]
        self.path_length = 0.0
        self.travel_time = 0.0
        self.neighbors = []
        self.current_path_embedding = np.zeros(8, np.float32)
        self.info = {'arrived': False}
        self.neighbors_embedding = np.zeros(self.number_of_nodes)
        self.neg_points = deepcopy(self.neg_points_reset)

        # self._check_origin_goal_distance()
        # self._reset_render()

        self.get_default_route()
        self.reference_path_set = set(self.nx_shortest_path)
        self.reference_reward = 1/len(self.reference_path_set)

        self._update_state()
        if self.neighbors == [] or \
                self._get_closest_distance_neg(self.current_node) < self.avoid_threshold or \
                self._get_closest_distance_neg(self.goal_node) < self.avoid_threshold:
            # make sure the first step is not a dead end node or origin node sample in avoid area
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

    def render(self, mode='human', plot_default=False, save=False, show=False):
        """
        Renders the environment
        """
        if self.verbose:
            print("Get path", self.path)
        if self.done and plot_default:
            ox.plot_graph_route(self.graph, self.nx_shortest_path,
                                save=save, filepath=repo_path + "images/default_image.png")

        # plot negative points
        for node in self.neg_points:
            self.ax.plot(node['x'], node['y'], c='blue',
                         marker='o', markersize=5)

        self.plot_line(self.path[-2:])

        if save:
            self.fig.savefig(self.render_img_path)

        if show and mode == 'human':
            self.fig.canvas.draw()

        return np.array(self.fig.canvas.buffer_rgba())

    def plot_line(self, node_pair):
        """
        Plot the line between two nodes
        """
        self.ax.plot([self.nodes[node_pair[0]]['x'], self.nodes[node_pair[1]]['x']],
                     [self.nodes[node_pair[0]]['y'], self.nodes[node_pair[1]]['y']], c='red', marker='o', markersize=1)

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
