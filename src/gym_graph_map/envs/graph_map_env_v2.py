# from collections import defaultdict

from pathlib import Path

import numpy as np
import networkx as nx

# for mapping
import osmnx as ox
from osmnx import distance
import pandas as pd
import wandb
# for plotting
import geopandas
import matplotlib.pyplot as plt

# for reinforcement learning environment
import gym
from gym import spaces
from gym.utils import seeding
from ray import tune

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
                shape: (number_of_nodes, number_of_nodes+2)
                first line is the state
                second line is the reference
        """

        self.adj_shape = (self.number_of_nodes, self.number_of_nodes)
        self.action_space = spaces.Discrete(
            self.number_of_nodes)

        # set observation space low
        state_low = np.full(shape=(self.number_of_nodes,), fill_value=0)
        nx_path_references_low = np.zeros(self.number_of_nodes)
        adj_low = np.full(shape=self.adj_shape, fill_value=-np.inf)
        obs_low = np.vstack((
            state_low,
            nx_path_references_low,
            # adj_low
        ))

        # set observation space high
        state_high = np.full(shape=(self.number_of_nodes,), fill_value=np.inf)
        nx_path_referenes_high = np.full(
            shape=(self.number_of_nodes, ), fill_value=self.number_of_nodes)
        adj_high = np.full(shape=self.adj_shape, fill_value=np.inf)
        obs_high = np.vstack((
            state_high,
            nx_path_referenes_high,
            # adj_high
        ))

        self.observation_space = spaces.Dict({
            "observations": spaces.Box(low=obs_low, high=obs_high, dtype=np.float32),
            "action_mask": spaces.Box(low=0, high=1, shape=(self.number_of_nodes,), dtype=np.int64),
            # "length": spaces.Box(low=0, high=float("inf"), shape=self.adj_shape, dtype=np.float32),
        })

        # self.adj = nx.to_numpy_array(self.reindexed_graph, weight="None", dtype=np.float32)

    def _set_config(self, config):
        """
        Sets the config
        """
        # Constant values
        self.graph = config['graph']
        self.verbose = config['verbose'] if config['verbose'] else False
        neg_df = pd.read(config['neg_df_path'])
        center_node = config['center_node']
        self.threshold = config['threshold']
        # graph radius or average short path in this graph, sampled by env stat
        self.threshold = 2900
        self.neg_points = self._get_neg_points(neg_df, center_node)

        if self.neg_points == []:
            raise ValueError("Negative weights not found")

        self.avoid_threshold = self.threshold / 5
        self.number_of_nodes = self.graph.number_of_nodes()

        # utility values
        self.render_img_path = repo_path + "images/render_img.png"

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
        self.reindexed_graph = nx.relabel.convert_node_labels_to_integers(
            graph, first_label=0, ordering='default')

        # self.node_dict = {node: i for i, node in enumerate(graph.nodes(data=False))}

        # self.node_dict_reversed = {i: node for node, i in self.node_dict.items()}

        self.nodes = self.reindexed_graph.nodes()

    def _update_state(self):
        """
        Updates:
            self.neighbors
            self.action_space
            self.state
            self.current_distance_goal
            self.current_closest_distance
        Uncomment self.action_space to update the neighbors if use MaskedSpace
        """
        self.neighbors = list(x for x in self.reindexed_graph.neighbors(
            self.current) if x not in self.passed_nodes_ids)

        # self.action_space.neighbors = self.neighbors

        self.current_distance_goal = self._great_circle_vec(
            self.current_node, self.goal_node)

        self.current_closest_distance = self._get_closest_distance_neg(
            self.current_node)

        if self.verbose:
            print(self.current, "'s neighbors: ", self.neighbors)
            print("current_distance_goal:", self.current_distance_goal)
            print("nx_shortest_path_length:", self.nx_shortest_path_length)

        # set state
        state = np.array([
            self.current,
            self.goal,
            self.current_distance_goal,
            self.current_closest_distance,
            self.path_length,
            self.travel_time,
        ], dtype=np.float32)
        state = np.pad(state, pad_width=(0, self.number_of_nodes - state.shape[0]),
                       mode='constant', constant_values=0)

        # set references
        nx_path_references = np.pad(np.array(self.nx_shortest_path), pad_width=(
            0, self.number_of_nodes - len(self.nx_shortest_path)), mode='constant', constant_values=0)

        self.state = {
            "observations": np.vstack((
                state,
                nx_path_references,
                # self.adj
            )),
            "action_mask": self.action_masks(),
        }

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
        self.df = pd.DataFrame(columns=df.columns)
        index = 0

        for _, row in df.iterrows():
            node = {'y': row[0], 'x': row[1], 'weight': row[2]}
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
            self.reindexed_graph, self.path[-2:], "length")[0]

        self.travel_time += ox.utils_graph.get_route_edge_attributes(
            self.reindexed_graph, self.path[-2:], "travel_time")[0]

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

    def _reward(self):
        """
        Computes the reward
        Overall reward, [-inf, 3.0]
        neg: [-inf, 0.0]: when closest_dist >= self.avoid_threshold, neg = 0.0; when closest_dist / self.avoid_threshold == 0.5, neg = -1
            the closer the node is to the negative point, the less the reward
        r2: {0.0, 1.0} (comfired) if reached the goal, the r2 is 1.0 else 0.0
        r3_v1: [0.0, 1.1] (comfired) if reached the goal, the r3 will caculate the path length with simple ratio
        r4: cumulative reward, [0.0, 0.001]. It aims to reward the agent who follow the references

        r5 range: [0, 1.0] (deprecated) the closer to the goal the higher the r5 for each done

        r3_v0: [0.0, 1.0] (deprecated) if reached the goal, the r3 will caculate the path length with sigmoid function,
            if aims to reward the shorter path, the sigmod funtion is constructed by the self.threshold,
            if the path length is less than the threshold the r3 is greater than 0.5 else less than 0.5
            r3_v0 deprecated because of overflow encountered inreduct issues in using sigmod and tanh
        """
        # too close to a negative point will cause a negative reward
        neg = min(
            0, np.log2(self.current_closest_distance / self.avoid_threshold))

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
                self.reindexed_graph, source=self.origin, target=self.goal, weight="length")
            self.nx_shortest_path_length = sum(ox.utils_graph.get_route_edge_attributes(
                self.reindexed_graph, self.nx_shortest_path, "length"))

            # self.nx_shortest_travel_time = nx.shortest_path(
            #     self.reindexed_graph, source=self.origin, target=self.goal, weight="travel_time")
            # self.nx_shortest_travel_time_length = sum(ox.utils_graph.get_route_edge_attributes(
            #     self.reindexed_graph, self.nx_shortest_travel_time, "travel_time"))
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
        self.info = {'arrived': False}

        # self._check_origin_goal_distance()

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

    def render(self, mode='human', plot_default=False, plot_learned=True, plot_neg=True, save=True, show=False):
        """
        Renders the environment
        """
        if self.verbose:
            print("Get path", self.path)
        if self.done and plot_default:
            ox.plot_graph_route(self.reindexed_graph, self.nx_shortest_path,
                                save=save, filepath=repo_path + "images/default_image.png")
        if plot_learned:
            save = False if plot_neg else save
            fig, ax = ox.plot_graph_route(
                self.reindexed_graph, self.path, save=False, filepath=self.render_img_path, show=False, close=False)
            if plot_neg:
                self.origin_node = self.nodes[self.origin]
                goal_df = pd.DataFrame({
                    "Longitude": [self.goal_node['x'], self.origin_node['x']],
                    "Latitude": [self.goal_node['y'], self.origin_node['y']]
                })
                # plot the origin and goal
                gdf_goal = geopandas.GeoDataFrame(
                    goal_df, geometry=geopandas.points_from_xy(goal_df['Longitude'], goal_df['Latitude']))
                gdf_goal.plot(ax=ax, color='yellow', markersize=20)

                # plot negative points
                gdf_neg = geopandas.GeoDataFrame(
                    self.df, geometry=geopandas.points_from_xy(self.df['Longitude'], self.df['Latitude']))
                gdf_neg.plot(ax=ax, markersize=10,
                             color="blue", alpha=1, zorder=7)

                plt.savefig(self.render_img_path)
                if show:
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
