import copy
from xml.dom import INDEX_SIZE_ERR

import networkx as nx
import numpy as np
from tqdm import tqdm


def make_bigram(list):
    return [(list[i], list[i + 1]) for i in range(len(list) - 1)]


def get_correct_adj_mat(ngaph):
    initial_adjacency_matrix = np.squeeze(
        np.asarray(
            nx.adjacency_matrix(ngaph, nodelist=sorted(
                ngaph.nodes()), weight=None).todense()
        )
    )
    adjacency_matrix = np.squeeze(
        np.asarray(
            nx.adjacency_matrix(
                ngaph, nodelist=sorted(ngaph.nodes())).todense()
        )
    )
    adjacency_matrix[initial_adjacency_matrix == 0] = - \
        1  # replace non-edges by -1 instead of 0
    return adjacency_matrix


class Graph:
    def __init__(self, networkx_graph, origin, goal, weights=None, random_weights=(0, 10), make_horizon=False):
        networkx_graph = networkx_graph.copy()
        self.was_directed = nx.is_directed(networkx_graph)
        networkx_graph = networkx_graph.to_directed()

        networkx_graph = nx.convert_node_labels_to_integers(
            networkx_graph, label_attribute="old_name")

        if weights is not None and nx.is_weighted(networkx_graph, weight=weights):
            print("Using weights:", weights)
        else:
            print("Using random weights")
            dico = {}
            key = False
            if networkx_graph.is_multigraph():
                key = True
            for e in tqdm(networkx_graph.edges(keys=key)):
                dico[e] = np.random.randint(
                    random_weights[0], random_weights[1], size=1)[0]
            nx.set_edge_attributes(networkx_graph, dico, name="weights")

        if make_horizon:
            networkx_graph, origin, goal = self._make_horizons(
                networkx_graph, origin, goal)

        # must rename to make nodes from 0 to order-1:
        # networkx_graph, origin, goal = self._reorder(networkx_graph, origin, goal)
        # print("Setting origin and goal", origin, goal)
        # set attributes
        self.made_horizon = make_horizon
        self.ngraph = networkx_graph
        self.adj_mat = get_correct_adj_mat(networkx_graph)
        self.adjacent_indices = [np.nonzero(
            self.adj_mat[i] != -1)[0] for i in range(self.adj_mat.shape[0])]

        print("Setting problem")
        self._set_problem(origin, goal)

    def _reorder(self, networkx_graph, origin, goal):
        print("Reordering nodes")
        dico = {}
        for idx, node in enumerate(networkx_graph.nodes):
            print(node)
            dico[node] = idx
            if node == goal:
                goal = idx
            elif node == origin:
                origin = idx
        return nx.relabel_nodes(networkx_graph, dico), origin, goal

    def _set_problem(self, origin, goal):
        self._set_position(origin)
        self.origin = origin
        self.goal = goal
        self.path = [origin]
        self.path_bigram = []

        print("Getting dijkstra path:")
        self.dijkstra_path = nx.shortest_path(self.ngraph, source=origin, target=goal, weight="weight")
        self.dijkstra_bigram = make_bigram(self.dijkstra_path)
        self.dijkstra_rew = sum([self.adj_mat[(e1, e2)]
                                for e1, e2 in self.dijkstra_bigram])
        
        print(self.dijkstra_path)
        # can't get the longest path with dijkstra
        # print("Getting Longest Path")
        # all_simple_paths = list(nx.all_simple_paths(self.ngraph, origin, goal))
        # ws = []
        # for path in tqdm(all_simple_paths):
        #     big = make_bigram(path)
        #     weight = 0
        #     for e1, e2 in big:
        #         weight += self.adj_mat[e1, e2]
        #     ws.append(weight)
        # i = np.argmax(np.array(ws))

        # self.longest_path_rew = ws[i]
        # self.longest_path = all_simple_paths[i]
        # self.longest_path_bigram = make_bigram(self.longest_path)

    def reset(self, origin=None, goal=None):
        if origin is None:
            origin = self.origin
        if goal is None:
            goal = self.goal
        self._set_problem(origin, goal)

    def _set_position(self, pos):
        self.position = pos

    def transition(self, new_pos):
        self.path.append(new_pos)
        self.path_bigram = make_bigram(self.path)

        if new_pos not in self.adjacent_indices[self.position]:
            print(f"{new_pos} not in {self.adjacent_indices[self.position]}")
            return False, False

        reward = self.adj_mat[self.position, new_pos]
        self._set_position(new_pos)

        done = self.position == self.goal

        return reward, done
    
    def _make_horizons(self, ngraph, origin, goal):
        pass
