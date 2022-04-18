import copy
import math
import time

import gym
import numpy as np
import pyglet

from .graph import Graph

import osmnx as ox




REWARD_INVALID = -1000

WINDOW_W = WINDOW_H = 1000

# TODO allow cop to move faster than robber should it be needed (see paper). basically just give it d turns before the
# turn bool switches
class ShortestRouteEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, networkx_graph, origin, goal, weights=None, random_weights=(0, 10), make_horizon=False):
        super(gym.Env, self).__init__()

        # Define action and observation space
        # They must be gym.spaces objects
        self.viewer = None
        self.reset(networkx_graph, origin, goal, weights, random_weights, make_horizon)

    def step(self, action):
        reward, done = self.graph.transition(action)
        if str(reward) == "False":  # action was invalid. Python is dumb and testing for "not reward" catches 0...
            reward = self.max_r * REWARD_INVALID
            print("INVALID!")

        return self.graph.position, reward, done, {}

    def reset(self, networkx_graph=None, origin=None, goal=None, weights=None, random_weights=(0, 10), make_horizon=False):
        if networkx_graph is None:
            self.graph.reset(origin, goal)
        else:
            self.layout = None
            self.graph = Graph(networkx_graph, origin, goal, weights, random_weights, make_horizon)
            self._make_PRAlpha()
            if make_horizon:
                self._make_horizons()
            self.action_space = gym.spaces.Discrete(len(networkx_graph.nodes))
            self.observation_space = gym.spaces.Tuple((
                gym.spaces.Discrete(len(networkx_graph.nodes)),
                gym.spaces.Discrete(len(networkx_graph.nodes))
            ))
        return self.graph.position

    def _make_R(self):
        self.R = self.graph.adj_mat.copy().astype("float64")

        max_r = np.max(self.R)
        self.max_r = max_r
        # Make non-existent transitions very bad to use
        self.R[self.R == -1] = max_r * 10

        self.R = -1 * self.R  # make rewards negative

        # The incentive to go to the goal is that once there, the agent can "spin" on it, gaining no negative rewards
        if not self.graph.made_horizon:
            self.R[self.graph.goal, self.graph.goal] = 0 + 1 / (1000 * len(self.graph.ngraph.nodes))

    def _make_PRAlpha(self):
        adj_mat = self.graph.adj_mat.copy()

        # Completely deterministic. So, really, we duplicate the adj mat len(nodes) times
        adj_mat[adj_mat >= 0] = 1
        adj_mat[adj_mat < 0] = 0

        def range_except_i(n, i):
            t = []
            for j in range(n):
                if j != i:
                    t.append(j)
            return t

        P = []
        for n in range(len(self.graph.ngraph.nodes)):
            prob_mat_for_n = adj_mat.copy()
            # p of switching to non-connected node is 0
            prob_mat_for_n[:, range_except_i(len(self.graph.ngraph.nodes), n)] = 0

            # if action's node is not connected, stay in current node
            for i in range(len(self.graph.ngraph.nodes)):
                if 1 in prob_mat_for_n[i]:
                    continue
                else:
                    prob_mat_for_n[i, i] = 1

            P.append(prob_mat_for_n)

        self.P = np.array(P)
        self._make_R()
        self.alpha = np.array(range(len(self.graph.ngraph.nodes)))
        self.alpha[self.graph.origin] = 1  # initial distribution is just the origin

    def get_dijkstra(self):
        """
        Returns the optimal path and reward for the current problem.

        @return: dijkstra path, dijkstra reward
        """
        return self.graph.dijkstra_path, self.graph.dijkstra_rew

    def render(self, mode='human', human_reading_delay=1, layout_function=None):
        """
        Renders the state of the env for a human

        @param mode: only "human" is implemented
        @param human_reading_delay: delay for which the render stays visible (in seconds)
        @param layout_function: Any networkx layout works here https://networkx.org/documentation/stable/reference/drawing.html#module-networkx.drawing.layout
        """
        if mode == "osmnx":
            fig, ax = ox.plot_graph_route(self.graph.ngraph, self.graph.path)
            return np.array(fig.canvas.buffer_rgba())

        if layout_function == None:
            layout_function = "spring"
            if self.graph.was_directed:
                layout_function = "shell"
            elif self.graph.made_horizon:
                layout_function = "horizon"

        def is_in_ipython():
            try:
                return hasattr(__builtins__, '__IPYTHON__')
            except NameError:
                return False

        if not is_in_ipython():
            from gym.envs.classic_control import rendering
        import matplotlib.pyplot as plt
        import networkx as nx

        if self.viewer is None:
            if not is_in_ipython():
                self.viewer = rendering.SimpleImageViewer()
                self.viewer.width = WINDOW_W
                self.viewer.height = WINDOW_H
                self.viewer.window = pyglet.window.Window(width=self.viewer.width, height=self.viewer.height,
                                                          display=self.viewer.display, vsync=False, resizable=True)
        if self.layout is None:
            if layout_function == "horizon":
                self.layout = hierarchy_pos(self.graph.ngraph, self.graph.origin)
            else:
                fun = getattr(nx, f"{layout_function}_layout")
                if layout_function == "spring":
                    self.layout = fun(self.graph.ngraph, k=10/math.sqrt(self.graph.ngraph.order()))
                else:
                    self.layout = fun(self.graph.ngraph)

        fig, ax = plt.subplots(figsize=(15, 15))
        fig.tight_layout()
        ax.axis("off")

        colors = np.array(["cyan"] * len(self.graph.ngraph.nodes))
        colors[list(self.graph.ngraph.nodes).index(self.graph.goal)] = "blue"
        colors[list(self.graph.ngraph.nodes).index(self.graph.origin)] = "pink"
        colors[list(self.graph.ngraph.nodes).index(self.graph.position)] = "red"

        optimal_path = self.graph.dijkstra_bigram

        edge_colors = []
        for edge in self.graph.ngraph.edges:
            if edge in self.graph.path_bigram:
                edge_colors.append("pink")
            elif edge in optimal_path:
                edge_colors.append("blue")
            else:
                edge_colors.append("black")


        nx.draw_networkx_nodes(self.graph.ngraph, self.layout, cmap=plt.get_cmap('jet'), node_color=colors, node_size=400)
        nx.draw_networkx_labels(self.graph.ngraph, self.layout)
        nx.draw_networkx_edges(self.graph.ngraph, self.layout, edge_color=edge_colors,
                               edgelist=[edge for edge in self.graph.ngraph.edges()], arrows=True,
                               connectionstyle='arc3, rad = 0.01')
        weights = nx.get_edge_attributes(self.graph.ngraph, "weight")

        weight_dict = {}
        for i in range(self.graph.adj_mat.shape[0]):
            for j in range(self.graph.adj_mat.shape[1]):
                w = self.graph.adj_mat[i,j]
                if w != -1:
                    weight_dict[(i,j)] = w

        nx.draw_networkx_edge_labels(self.graph.ngraph, self.layout,
                                     edge_labels=weight_dict, label_pos=0.25,
                                     font_size=12)

        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        if is_in_ipython():
            plt.show()
        else:
            plt.close(fig)
            self.viewer.imshow(image_from_plot)
        import time
        time.sleep(human_reading_delay)

    def close(self):
        try:
            self.viewer.window.close()
        except:
            pass

    def _make_horizons(self):
        self.horizon_states = self.graph.horizon_states[:-1]
        self.horizon_acts = self.graph.horizon_acts[:-1]
        self.horizon_rews = copy.deepcopy(self.horizon_acts)

        for i, epoch in enumerate(self.horizon_acts):
            for j, state_acts in enumerate(epoch):
                for k, act in enumerate(state_acts):
                    rew = self.graph.adj_mat[self.horizon_states[i][j],act]
                    if rew == -1:
                        rew = np.max(self.graph.adj_mat)

                    self.horizon_rews[i][j][k] = -1 * rew

import random
import networkx as nx
# https://stackoverflow.com/questions/29586520/can-one-get-hierarchical-graphs-from-networkx-with-python-3/29597209#29597209
def hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    '''
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723.
    Licensed under Creative Commons Attribution-Share Alike

    If the graph is a tree this will return the positions to plot this in a
    hierarchical layout.

    G: the graph (must be a tree)

    root: the root node of current branch
    - if the tree is directed and this is not given,
      the root will be found and used
    - if the tree is directed and this is given, then
      the positions will be just for the descendants of this node.
    - if the tree is undirected and not given,
      then a random choice will be used.

    width: horizontal space allocated for this branch - avoids overlap with other branches

    vert_gap: gap between levels of hierarchy

    vert_loc: vertical location of root

    xcenter: horizontal location of root
    '''

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  # allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''

        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)
        if len(children) != 0:
            dx = width / len(children)
            nextx = xcenter - width / 2 - dx / 2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap,
                                     vert_loc=vert_loc - vert_gap, xcenter=nextx,
                                     pos=pos, parent=root)
        return pos

    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)
