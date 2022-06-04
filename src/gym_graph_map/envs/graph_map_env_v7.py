# Version 4 of the Graph Map Environment with the following changes:
# 1. The environment now has a config dict that can be passed to the constructor.
# 2. The environment now has a reset_config method that can be called to reset the
#     config dict to its default values.
# 3. The environment now has a new state as a sequence as observation space so the model won't have to learn to
#     handle the wrapper stuff.
# 4. The environment now has a envolving environment based on a marsh attribute and elevation attribute.
# 5. The environment now has a a better render mode.
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from collections import deque

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
directions = ("Turn Left", "Proceed", "Turn Right", "Turn Back")
reverse_directions_dict = {i: d for i, d in enumerate(directions)}
directions_dict = {d: i for i, d in enumerate(directions)}


class GraphMapEnvV7(gym.Env):
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

        self._get_optimal_graph()
        self._set_action_observation()

        self.reset()
        # self._set_utility_function()
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

        self.action_space = spaces.Discrete(3)
        self.graph_space = spaces.Discrete(self.number_of_nodes)

        # set observation space low
        state_dim = 2
        state_low = np.zeros(state_dim)

        goal_embeddings_low = np.zeros(1)
        nx_embeddings_low = np.zeros(2)
        neg_embeddings_low = np.zeros(self.action_space.n)
        # length_embeddings_low = np.zeros(self.action_space.n)

        obs_low = np.hstack((
            state_low,
            goal_embeddings_low,
            nx_embeddings_low,
            neg_embeddings_low,
            # length_embeddings_low,
        ))

        state_high = np.full(shape=(state_dim,), fill_value=np.inf)

        goal_embeddings_high = np.full(1, np.inf)
        nx_embeddings_high = np.full(2, np.inf)
        neg_embeddings_high = np.full(self.action_space.n, np.inf)
        # length_embeddings_high = np.full(self.action_space.n, np.inf)

        obs_high = np.hstack((
            state_high,
            goal_embeddings_high,
            nx_embeddings_high,
            neg_embeddings_high,
            # length_embeddings_high,
        ))

        self.observation_space = spaces.Dict({
            "observations": spaces.Box(low=obs_low, high=obs_high, dtype=np.float32),
            "action_mask": spaces.Box(low=0, high=1, shape=(self.action_space.n,), dtype=np.int64),
        })

    def _update_state(self):
        """
        Updates state
        """

        action_masks = self.action_masks()

        # forward neighbors
        self.neighbors = list(x for x in self.graph.neighbors(
            self.current) if x not in self.passed_nodes_ids)

        # observation_set = get_observation(
        #     self.current, self.graph, size=self.nblock, return_subgraph=False)

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
            # self.current,
            # self.goal,
            self.current_distance2goal,
            self.current_distance2neg,
            # self.path_length,
            # self.travel_time,
            # self.current_step,
        ], dtype=np.float32)

        embeddings = self._get_observation_embeddings(
            None, u=self.current)

        self.state = {
            "observations": np.hstack((
                state,
                embeddings['goal_embeddings'],
                embeddings['nx_embeddings'],
                embeddings['neg_embeddings'],
                # embeddings['length_embeddings'],
            )),
            "action_mask": action_masks,
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
        self.nblock = config['nblock']

        self.envolving = config['envolving']
        self.envolving_freq = config['envolving_freq']
        self.init_envolving = config['init_envolving']
        # graph radius or average short path in this graph, sampled by env state
        self.neg_points_reset = self._get_neg_points(
            pd.read_csv(config['neg_df_path']), center_node)
        self.neg_points = deepcopy(self.neg_points_reset)

        self.avoid_threshold = config['avoid_threshold']
        self.number_of_nodes = self.graph.number_of_nodes()

        # load embeddings
        # self.embedding = np.load(config['embedding_path'])
        # assert self.embedding.shape[0] == self.number_of_nodes
        # self.embedding_dim = self.embedding.shape[1]

        # utility values
        self.render_img_path = repo_path + "images/render_img.png"

    def _get_direction(self):
        """
        Gets the direction
        """

    def _get_direction_bearing(self, u, v):
        bearing = calculate_bearing(
            self.nodes[u]['y'], self.nodes[u]['x'], self.nodes[v]['y'], self.nodes[v]['x'])
        direction = self._get_direction(bearing)
        return bearing, direction

    def _check_edges(self, direction):
        if direction not in self.avail_directions:
            u, v = self.current, self.reference_path[0]
            self.graph.add_edge(u, v)
            self.avail_directions[direction] = v

    def _get_observation_embeddings(self, observation_set, u):
        """
        Returns embeddings of observation_set
        """
        # observation_set.remove(u)

        neg_embeddings = np.zeros(self.action_space.n)
        # length_embeddings = np.zeros(self.action_space.n)

        nx = self._get_cloest_node(self.reference_path, u)

        _, nx_direction_cloest = self._get_direction_bearing(
            u, nx)
        _, nx_direction = self._get_direction_bearing(
            u, self.reference_path[0])
        _, goal_direction = self._get_direction_bearing(u, self.goal)

        for d in self.avail_directions:
            v = self.avail_directions[d]
            # length_embeddings[d] = self.graph.edges[u, v, 0]['length']
            neg_embeddings[d] = self._get_closest_distance_neg(self.nodes[v])

        return {
            "goal_embeddings": np.array([goal_direction]),
            "nx_embeddings": np.array([nx_direction, nx_direction_cloest]),
            "neg_embeddings": neg_embeddings,
            # "length_embeddings": length_embeddings,
        }

    def _get_cloest_node(self, node_list, u):
        """
        Returns the cloest node from node_list to u
        """
        min_dist = np.inf
        nx = u
        for v in node_list:
            dist = self._great_circle_vec(self.nodes[u], self.nodes[v])
            if dist < min_dist:
                min_dist = dist
                nx = v
        return nx

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

        self.current_bearing = calculate_bearing(
            self.nodes[self.path[-2]], self.nodes[self.path[-1]])

        self.ref_idx = index_finder(self.reference_path, self.current)
        if self.ref_idx > -1:
            self.reference_path = self.reference_path[self.ref_idx + 1:]

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

    def _init_envolving(self):
        """
        Envolving
        """
        while self.init_envolving > 0:
            envolving_node = self.graph.nodes[np.random.choice(
                self.number_of_nodes)]

            node = {'y': envolving_node['y'],
                    'x': envolving_node['x'], 'weight': 5}
            self.neg_points_reset.append(node)
            self.init_envolving -= 1

    def _envolving(self):
        # TODO: choice with elevation
        if self.envolving and self.current_step % self.envolving_freq == 0:
            envolving_node = self.nodes[self.goal]
            d1 = 0
            d2 = 0
            while d1 < self.avoid_threshold + 300 or d2 < self.avoid_threshold + 300:
                # choose the node until the distance is greater than the threshold
                envolving_node = self.graph.nodes[np.random.choice(
                    self.number_of_nodes)]
                d1 = self._great_circle_vec(
                    self.nodes[self.goal], envolving_node)
                d2 = self._great_circle_vec(
                    self.nodes[self.origin], envolving_node)

            node = {'y': envolving_node['y'],
                    'x': envolving_node['x'], 'weight': 5}
            self.neg_points.append(node)

    def _check_origin_goal_distance(self):
        """
        Check the distance between origin and goal for test
        """
        self.origin_goal_distance = self._great_circle_vec(
            self.current_node, self.goal_node)

        if self.origin_goal_distance < self.avoid_threshold:
            if self.verbose:
                print("The distance between origin and goal is too close, resetting...")
            self.reset()

    def _reset_render(self):
        """
        Reset the render
        """
        # plotting base map
        self.fig, self.ax = ox.plot_graph(
            self.graph, show=False, close=False, figsize=(30, 30))
        self.origin_node = self.nodes[self.origin]
        self.ax.plot(
            self.origin_node['x'], self.origin_node['y'], c='yellow', marker='o', markersize=15)
        self.ax.plot(self.goal_node['x'], self.goal_node['y'],
                     c='yellow', marker='o', markersize=15)

    def _reset_embeddings(self):
        """
        Reset the embeddings
        """
        self.current_path_embedding = np.zeros(8, np.float32)

    def _get_optimal_graph(self):
        """
        Get the optimal graph
        """
        self._init_envolving()
        self.neg_points = deepcopy(self.neg_points_reset)
        self.optimal_graph = deepcopy(self.graph)
        # remove node that too close to neg points
        for x in self.graph.nodes():
            if self._get_closest_distance_neg(self.nodes[x]) < self.avoid_threshold + 500:
                self.optimal_graph.remove_node(x)

    def _reward(self):
        """
        Computes the reward
        """
        # too close to a negative point will cause a negative reward
        neg = min(
            0, max(-15, np.log2(self.current_distance2neg / self.avoid_threshold)))

        # r3 = r2 * self.tanh(self.path_length) # v0
        # r3 = r2 * (self.nx_shortest_path_length / self.path_length)  # v1

        if self.ref_idx > -1:
            r4 = self.reference_reward
        else:
            r4 = -self.reference_reward/3

        return r4 + neg

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
        self.current = self.avail_directions[action]
        self.current_node = self.nodes[self.current]
        self.current_step += 1
        self.path.append(self.current)
        self.passed_nodes_ids.add(self.current)

        self._envolving()

        if self.verbose:
            print("self.path:", self.path)

        self._update_attributes()

        self.info['arrived'] = self.current == self.goal
        if self.info['arrived'] or self.reference_path == []:
            return self.state, 3 - self.reward, True, self.info

        self._update_state()
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

    def get_optimal_route(self):
        """
        Returns the optimal path
        """
        try:
            self.optimal_path = nx.shortest_path(
                self.optimal_graph, source=self.origin, target=self.goal, weight="travel_time")
            self.optimal_path_length = sum(ox.utils_graph.get_route_edge_attributes(
                self.optimal_graph, self.optimal_path, "travel_time"))
        except nx.exception.NetworkXNoPath:
            if self.verbose:
                print("No path found for default route. Restting...")
            self.reset()
        except nx.exception.NodeNotFound:
            if self.verbose:
                print("Node not in Graph Restting...")
            self.reset()

    def get_default_route(self, travel_time=False):
        """
        Set the default route by tratitional method
        """
        try:
            self.nx_shortest_path = nx.shortest_path(
                self.graph, source=self.origin, target=self.goal, weight="length")
            self.nx_shortest_path_length = sum(ox.utils_graph.get_route_edge_attributes(
                self.graph, self.nx_shortest_path, "length"))
            if travel_time:
                self.nx_shortest_travel_time = nx.shortest_path(
                    self.graph, source=self.origin, target=self.goal, weight="travel_time")
                self.nx_shortest_travel_time_length = sum(ox.utils_graph.get_route_edge_attributes(
                    self.graph, self.nx_shortest_travel_time, "travel_time"))
        except nx.exception.NetworkXNoPath:
            if self.verbose:
                print("No path found for default route. Restting...")
            self.reset()

    def reset(self):
        """
        Resets the environment
        """
        self.origin = np.random.choice(list(self.optimal_graph.nodes()))
        self.goal = np.random.choice(list(self.optimal_graph.nodes()))
        self.goal_node = self.nodes[self.goal]

        self.current = self.origin
        self.current_node = self.nodes[self.current]

        self.passed_nodes_ids = {self.current}
        self.current_step = 0
        self.done = False
        self.path = [self.current]
        self.current_bearing = None
        self.path_length = 0.0
        self.travel_time = 0.0
        self.neighbors = []
        self.info = {'arrived': False}
        self.reward = 0.0

        self.neg_points = deepcopy(self.neg_points_reset)

        self._check_origin_goal_distance()
        self._reset_embeddings()

        # self.get_default_route()
        self.get_optimal_route()
        self.reference_reward = 3/len(self.optimal_path)
        self.reference_path = deepcopy(self.optimal_path)[1:]
        if self.reference_path == []:
            self.reset()

        self._update_state()
        if self.neighbors == []:
            # make sure the first step is not a dead end node or origin node sample in avoid area
            self.reset()
        return self.state

    def action_masks(self):
        """
        Computes the action mask
        Returns:
            action_mask: [1, 0, ...]
        """
        self.avail_bearing = {data['bearing']: v for u, v,
                              data in self.graph.edges(self.current, data=True)}
        abk = sorted(list(self.avail_bearing.keys()))

        if len(abk) == 3:
            self.avail_directions = {0: self.avail_bearing[abk[0]],
                                     1: self.avail_bearing[abk[1]],
                                     2: self.avail_bearing[abk[2]]}
        elif len(abk) == 2:
            if self.avail_bearing[abk[0]] >= self.current_bearing:
                self.avail_directions = {1: self.avail_bearing[abk[0]],
                                         2: self.avail_bearing[abk[1]]}
            elif self.avail_bearing[abk[1]] <= self.current_bearing:
                self.avail_directions = {0: self.avail_bearing[abk[0]],
                                         1: self.avail_bearing[abk[1]]}
            else:
                self.avail_directions = {0: self.avail_bearing[abk[0]],
                                         2: self.avail_bearing[abk[1]]}
        elif len(abk) == 1:
            self.avail_directions = {0: self.avail_bearing[abk[0]]}
        else:
            self.avail_directions = {}

        self.mask = np.isin(range(self.action_space.n), list(
            self.avail_directions.keys()), assume_unique=True).astype(np.int64)
        return self.mask

    def render(self, mode='human', route=[], save=False, show=False, figsize=(30, 30)):
        """
        Renders the environment
        """
        if self.verbose:
            print("Get path", self.path)

        if route != []:
            ox.plot_graph_route(self.graph, route,
                                save=save, filepath=repo_path + "images/default_image.png", figsize=figsize)

        # plot negative points
        for node in self.neg_points:
            self.ax.plot(node['x'], node['y'], c='blue',
                         marker='o', markersize=15)

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


def calculate_bearing(lat1, lng1, lat2, lng2):
    """
    Calculate the compass bearing(s) between pairs of lat-lng points.

    Vectorized function to calculate (initial) bearings between two points'
    coordinates or between arrays of points' coordinates. Expects coordinates
    in decimal degrees. Bearing represents angle in degrees (clockwise)
    between north and the geodesic line from point 1 to point 2.

    Parameters
    ----------
    lat1 / y : float or numpy.array of float
        first point's latitude coordinate
    lng1 / x : float or numpy.array of float
        first point's longitude coordinate
    lat2 / y: float or numpy.array of float
        second point's latitude coordinate
    lng2 / x: float or numpy.array of float
        second point's longitude coordinate

    Returns
    -------
    bearing : float or numpy.array of float
        the bearing(s) in decimal degrees
    """
    # get the latitudes and the difference in longitudes, in radians
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    d_lng = np.radians(lng2 - lng1)

    # calculate initial bearing from -180 degrees to +180 degrees
    y = np.sin(d_lng) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * \
        np.cos(lat2) * np.cos(d_lng)
    initial_bearing = np.degrees(np.arctan2(y, x))

    # normalize to 0-360 degrees to get compass bearing
    return initial_bearing % 360


def get_observation(u, G, size=10, return_subgraph=False):
    dq = deque([u])
    visited = set()
    visited.add(u)

    while len(dq) != 0:
        v = dq.popleft()
        for i in G.neighbors(v):
            if i not in visited:
                dq.append(i)
                visited.add(i)
            if len(visited) == size + 1:
                # make sure the current not did not count in the size
                if return_subgraph:
                    H = G.subgraph(list(visited))
                    return H, visited
                else:
                    return visited
    return visited


def index_finder(lst, item):
    try:
        return lst.index(item, 0)
    except ValueError:
        return -1

# class MaskedDiscreteAction(spaces.Discrete):
#     def __init__(self, n):
#         super().__init__(n)
#         self.neighbors = None

#     def super_sample(self):
#         return np.int64(super().sample())

#     def sample(self):
#         # The type need to be the same as Discrete
#         return np.int64(np.random.choice(self.neighbors))
