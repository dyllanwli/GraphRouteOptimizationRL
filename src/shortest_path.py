import osmnx as ox
import networkx as nx
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
from src.utils import plot_path
import wandb
from tqdm import tqdm


class ShortestPath:
    def __init__(self, G):
        self.G = G
        self.node_num = G.number_of_nodes()
        self.R = np.matrix(np.ones(shape = (self.node_num, self.node_num))) #reward matrix
        self.R *= -1
        self.gamma = 0.8
        self.initial_state = 1
        self.node_dict = dict()
        self.reverse_nodes_dict = dict()
        self.new_points_list = list(G.edges())
        self.build_node_lists(self.G)

    def build_node_lists(self, G):
        self.node_num = G.number_of_nodes()
        print("nodesNumber:", self.node_num)
        for idx, n in enumerate(G.nodes()):
            self.node_dict[n] = idx
            self.reverse_nodes_dict[idx] = n
        
    def check_nodes(self, origin_node, destination_node):
        print("origin_node:", self.node_dict[origin_node])
        print("destination_node:", self.node_dict[destination_node])

    def get_route_length(self, route):
        edge_lengths = ox.utils_graph.get_route_edge_attributes(self.G, route, "length")
        return round(sum(edge_lengths))
    
    def get_route_lat_long(self, route):
        long = [] 
        lat = []  
        for i in route:
            point = self.G.nodes[i]   
            long.append(point['x'])
            lat.append(point['y'])
        return lat, long
        
    def networkx_shortest_path(self, origin, destination, plot = True):
        route = ox.shortest_path(self.G, origin, destination, weight="travel_time")
        if plot:
            fig, ax = ox.plot_graph_route(self.G, route, node_size=0)
            fig.show()
        return route

    def available_actions(self, state):
        startingNode_row = self.R[state, ]
        av_act = np.where(startingNode_row >= 0)[1]
        return av_act

    def sample_next_action(self, available_act):
        next_action = int(np.random.choice(available_act, 1))
        return next_action

    def update(self, Q,startingNode, action):
        max_index = np.where(Q[action,] == np.max(Q[action,]))[1]
        if max_index.shape[0] > 1:
            max_index = int(np.random.choice(max_index, size = 1))
        else:
            max_index = int(max_index)
        max_value = Q[action, max_index]
        Q[startingNode, action] = self.R[startingNode, action] + self.gamma *max_value
        if np.max(Q > 0):
            return(np.sum(Q/np.max(Q)*100))
        else:
            return (0)

    def shortest_path(self, origin, destination, verbose=False):
        origin = self.node_dict[origin]
        destination = self.node_dict[destination]
        for point in self.new_points_list:
            p0, p1 = self.node_dict[point[0]], self.node_dict[point[1]]
            self.R[(p0, p1)] = 150 if p1 == destination else 0
            self.R[(p1, p0)] = 150 if p0 == destination else 0
            
        Q = np.matrix(np.zeros([self.node_num, self.node_num])) # row: current state; col: next state
        available_act = self.available_actions(self.initial_state)
        action = self.sample_next_action(available_act)
        self.update(Q, self.initial_state, action)
        scores = []

        for _ in tqdm(range(self.node_num*100)):
            startingNode = np.random.randint(0, int(Q.shape[0]))
            action = self.sample_next_action(self.available_actions(startingNode))
            score = self.update(Q, startingNode, action)
            scores.append(score)

        steps = [origin] # set steps from origin
        current = origin
        print("start finding path...")
        if verbose:
            wandb.init(project="osmnx-rl")
        while current != destination:
            next_step_index = np.where(Q[current, ] == np.max(Q[current, ]))[1]
            if next_step_index.shape[0] > 1:
                next_step_index = int(np.random.choice(next_step_index, size = 1))
            else:
                next_step_index = int(next_step_index)
            steps.append(next_step_index)
            current = next_step_index
            if verbose:
                wandb.log({"next_step_index": next_step_index})
        return score, steps