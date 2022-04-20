from pathlib import Path
import random
import imageio
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
import networkx as nx
from rx import catch
import wandb
from wandb import config
from tqdm import tqdm

def plot_graph(adjacency_matrix, figure_title=None, print_shortest_path=False, src_node=None, filename=None,
               added_edges=None, pause=False):
    adjacency_matrix = np.array(adjacency_matrix)
    rows, cols = np.where(adjacency_matrix > 0)
    edges = list(zip(rows.tolist(), cols.tolist()))
    values = [adjacency_matrix[i][j] for i, j in edges]
    weighted_edges = [(e[0], e[1], values[idx]) for idx, e in enumerate(edges)]
    plt.cla()
    fig = plt.figure(1)
    if figure_title is None:
        plt.title("The shortest path for every node to the target")
    else:
        plt.title(figure_title)
    G = nx.Graph()
    G.add_weighted_edges_from(weighted_edges)
    # plot
    labels = nx.get_edge_attributes(G, 'weight')
    pos = nx.kamada_kawai_layout(G)
    # set with_labels to False if use node labels
    nx.draw(G, pos=pos, with_labels=True, font_size=15)
    nodes = nx.draw_networkx_nodes(G, pos, node_color="y")
    nodes.set_edgecolor('black')
    nodes = nx.draw_networkx_nodes(
        G, pos, nodelist=[0, src_node] if src_node else [0], node_color="g")
    nodes.set_edgecolor('black')
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=labels, font_size=15)
    if print_shortest_path:
        print("The shortest path (dijkstra) is showed below: ")
        added_edges = []
        for node in range(1, num_nodes):
            shortest_path = nx.dijkstra_path(G, node, 0)  # [1,0]
            print("{}: {}".format("->".join([str(v) for v in shortest_path]),
                                  nx.dijkstra_path_length(G, node, 0)))
            added_edges += list(zip(shortest_path, shortest_path[1:]))
    if added_edges is not None:
        nx.draw_networkx_edges(
            G, pos, edgelist=added_edges, edge_color='r', width=2)

    if filename is not None:
        plt.savefig(filename)

    if pause:
        plt.pause(0.3)
    else:
        plt.show()

    # return img for video generation
    img = None
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    w, h = fig.canvas.get_width_height()
    img = img.reshape((h, w, 3))
    return img

def get_best_actions(D, states):
    best_actions = []
    for node in range(1, num_nodes):
        actions = [(idx, states[idx])
                   for idx, weight in enumerate(D[node]) if weight > 0]
        actions, scores = zip(*actions)
        best_actions.append((node, actions[scores.index(max(scores))]))
    return best_actions

def print_best_actions(best_actions):
    best_actions_info = [
        "{}->{}".format(item[0], item[1]) for item in best_actions]
    return ", ".join(best_actions_info)

def value_iteration(threshold=0.001, gamma=1, visualize=True):
    print("-" * 20)
    print("value iteration begins ...")
    states = [0] * num_nodes  # max the negative of the length
    states_new = [0] + [float("-inf")] * (num_nodes - 1)
    epoch = 0
    while True:
        for j in tqdm(range(1, num_nodes)):
            state = [- weight + gamma * states[idx] for idx, weight in enumerate(D[j]) if weight > 0]
            states_new[j] = max(state) 

        epoch += 1
        print("epoch {}: values：{}\tpolicies：{}".format(epoch, list(np.around(states_new, decimals=2)),
                                                        print_best_actions(get_best_actions(D, states_new))))
        if max([abs(states_new[idx] - states[idx]) for idx in range(num_nodes)]) <= threshold:
            states[:] = states_new[:]
            break
        states[:] = states_new[:]

    best_actions = get_best_actions(D, states)
    if visualize:
        # not save figure when filename is None
        plot_graph(D, print_shortest_path=False, added_edges=best_actions, filename="value_iteration.png",
                   figure_title="Value iteration for the shortest paths")
    print("-" * 20)
    print("The best action for every node (value_iteration) is showed below: ")
    print(print_best_actions(get_best_actions(D, states_new)))
    print("-" * 20)


def policy_iteration(gamma=1, visualize=True):
    print("-" * 20)
    print("policy iteration begins ...")
    states = [0] * num_nodes  # max the negative of the length
    actions = [0] * num_nodes
    states_new = [0] + [float("-inf")] * (num_nodes - 1)
    actions_new = [0] * num_nodes
    epoch = 0
    while True:
        epoch += 1
        # policy evaluation
        for j in range(1, num_nodes):
            if epoch == 0:  # the initial policy
                states_new[j] = np.mean(
                    [-weight + gamma * states[idx] for idx, weight in enumerate(D[j]) if weight > 0])
            else:
                states_new[j] = -D[j][actions[j]] + gamma * states[actions[j]]
        # policy improvement
        for j in range(1, num_nodes):
            action_score_pair = [(idx, -weight + gamma * states_new[idx]) for idx, weight in enumerate(D[j]) if
                                 weight > 0]
            actions_, scores_ = zip(*action_score_pair)
            actions_new[j] = actions_[scores_.index(max(scores_))]
        print("epoch {}: values：{}\tpolicies：{}".format(epoch, list(np.around(states_new, decimals=2)),
                                                        print_best_actions(get_best_actions(D, states_new))))
        if actions_new == actions:
            print("The best action for every node (policy_iteration) is showed below: ")
            best_actions = [(idx, tar)
                            for idx, tar in enumerate(actions_new[1:], start=1)]
            if visualize:
                # not save figure when filename is None
                plot_graph(D, print_shortest_path=False, added_edges=best_actions, filename="policy_iteration.png",
                           figure_title="Policy iteration for shortest paths")
            best_actions = [(idx, tar) for idx, tar in best_actions]
            print("-" * 20)
            print(print_best_actions(best_actions))
            print("-" * 20)
            break
        states[:] = states_new[:]
        actions[:] = actions_new[:]
    return 0


def epsilon_greedy(s_curr, q, epsilon):
    # find the potential next states(actions) for current state
    potential_next_states = np.where(np.array(D[s_curr]) > 0)[0]
    # if len(potential_next_states) == 0:
    #     return s_curr
    if random.random() > epsilon:  # greedy
        q_of_next_states = q[s_curr][potential_next_states]
        s_next = potential_next_states[np.argmax(q_of_next_states)]
    else:  # random select
        s_next = random.choice(potential_next_states)
    return s_next


def sarsa(start_state=3, num_epoch=200, gamma=0.8, epsilon=0.05, alpha=0.1, visualize=True, save_video=False):
    print("-" * 20)
    print("sarsa begins ...")
    if start_state == 0:
        raise Exception("start node(state) can't be target node(state)!")
    imgs = []  # useful for gif/video generation
    len_of_paths = []
    # init all q(s,a)
    q = np.zeros((num_nodes, num_nodes))  # num_states * num_actions
    for i in tqdm(range(1, num_epoch + 1)):
        s_cur = start_state
        s_next = epsilon_greedy(s_cur, q, epsilon=epsilon)
        path = [s_cur]
        len_of_path = 0
        while True:
            s_next_next = epsilon_greedy(s_next, q, epsilon=epsilon)
            # update q
            reward = -D[s_cur][s_next]
            delta = reward + gamma * q[s_next, s_next_next] - q[s_cur, s_next]
            q[s_cur, s_next] = q[s_cur, s_next] + alpha * delta
            # update current state
            s_cur = s_next
            s_next = s_next_next
            len_of_path += -reward
            path.append(s_cur)
            if s_cur == 0:
                break
        len_of_paths.append(len_of_path)
        if visualize:
            img = plot_graph(D, print_shortest_path=False, src_node=start_state,
                             added_edges=list(zip(path[:-1], path[1:])), pause=True,
                             figure_title="sarsa\n {}th epoch: {}".format(i, cal_distance(path)))
            imgs.append(img)

    if visualize:
        plt.show()
    if visualize and save_video:
        print("begin to generate gif/mp4 file...")
        # creates video/gif out of list of images
        imageio.mimsave("sarsa.gif", imgs, fps=5)
    # print the best path for start state to target state
    strs = "best path for node {} to node 0: ".format(start_state)
    strs += "->".join([str(i) for i in path])
    print(strs)
    return 0


def sarsa_lambda(start_state=3, num_epoch=200, gamma=0.8, epsilon=0.05, alpha=0.1, lamda=0.9, visualize=True,
                 save_video=False):
    print("-" * 20)
    print("sarsa(lamda) begins ...")
    imgs = []  # useful for gif/video generation
    len_of_paths = []
    # init all q(s,a)
    q = np.zeros((num_nodes, num_nodes))  # num_states * num_actions
    for i in tqdm(range(1, num_epoch + 1)):
        s_cur = start_state
        s_next = epsilon_greedy(s_cur, q, epsilon=epsilon)
        e = np.zeros((num_nodes, num_nodes))  # eligibility traces
        path = [s_cur]  # save the path for every event
        len_of_path = 0
        while True:
            s_next_next = epsilon_greedy(s_next, q, epsilon=epsilon)
            # update q
            e[s_cur, s_next] = e[s_cur, s_next] + 1
            reward = -D[s_cur][s_next]
            delta = reward + gamma * q[s_next, s_next_next] - q[s_cur, s_next]
            q = q + alpha * delta * e
            # update e
            e = gamma * lamda * e
            # update current state
            s_cur = s_next
            s_next = s_next_next
            len_of_path += -reward
            path.append(s_cur)
            if s_cur == node_dict[destination_node]:  # if current state is target state, finish the current event
                break
        len_of_paths.append(len_of_path)
        if visualize:
            img = plot_graph(D, print_shortest_path=False, src_node=start_state,
                             added_edges=list(zip(path[:-1], path[1:])),
                             pause=True,
                             figure_title="sarsa(lambda)\n {}th epoch: {}".format(i, cal_distance(path)))
            imgs.append(img)
    if visualize:
        plt.show()
    if visualize and save_video:
        print("begin to generate gif/mp4 file...")
        # generate video/gif out of list of images
        imageio.mimsave("sarsa(lambda).gif", imgs, fps=5)
    # print the best path for start state to target state
    strs = "best path for node {} to node {}: ".format(start_state, node_dict[destination_node])
    strs += "->".join([str(i) for i in path])
    print(strs)
    return 0


def q_learning(start_state=3, num_epoch=200, gamma=0.8, epsilon=0.05, alpha=0.1, visualize=True, save_video=False):
    print("-" * 20)
    print("q_learning begins ...")
    imgs = []  # useful for gif/video generation
    len_of_paths = []
    # init all q(s,a)
    q = np.zeros((num_nodes, num_nodes))  # num_states * num_actions
    for i in tqdm(range(1, num_epoch + 1)):
        s_cur = start_state
        path = [s_cur]
        len_of_path = 0
        while True:
            s_next = epsilon_greedy(s_cur, q, epsilon=epsilon)

            # greedy policy
            # epsilon<0, greedy policy
            s_next_next = epsilon_greedy(s_next, q, epsilon=-0.2)
            while s_cur == s_next_next:
                s_next_next = epsilon_greedy(s_next, q, epsilon=-0.2)

            # update q
            reward = -D[s_cur][s_next]
            delta = reward + gamma * q[s_next, s_next_next] - q[s_cur, s_next]
            q[s_cur, s_next] = q[s_cur, s_next] + alpha * delta
            # update current state
            s_cur = s_next
            len_of_path += -reward
            path.append(s_cur)
            if s_cur == node_dict[destination_node]:
                break
        len_of_paths.append(len_of_path)
        if visualize:
            img = plot_graph(D, print_shortest_path=False, src_node=start_state,
                             added_edges=list(zip(path[:-1], path[1:])), pause=True,
                             figure_title="q-learning\n {}th epoch: {}".format(i, cal_distance(path)))
            imgs.append(img)

    if visualize:
        plt.show()
    if visualize and save_video:
        print("begin to generate gif/mp4 file...")
        # generate video/gif out of list of images
        imageio.mimsave("q-learning.gif", imgs, fps=5)
    # print the best path for start state to target state
    strs = "best path for node {} to node {}: ".format(start_state, node_dict[destination_node])
    strs += "->".join([str(i) for i in path])
    print(strs)
    return 0

class ShortestPath:
    def __init__(self, G, config):
        self.G = G
        self.number_of_nodes = self.G.number_of_nodes()
        self.config = config

def print_best_actions(best_actions):
    best_actions_info = [
        "{}->{}".format(x[0], x[1]) for x in best_actions]
    return ", ".join(best_actions_info)
    
def cal_distance(path):
    dis = 0
    for i in range(len(path) - 1):
        dis += D[path[i]][path[i + 1]]
    return dis

def remove_de_node(D, nodes_dict):
    # remove dead-end road
    de_nodes = [nodes_dict[idx] for idx, n in enumerate(D) if sum(n) <= 0]
    return de_nodes

random.seed(100)
np.random.seed(50)

if __name__ == '__main__':
    # adjacent matrix
    # the target node is 0
    center_point = (29.72346214336903, -95.38599726549226) # houston museum/houston center pointsorigin_point = center_point
    origin_point = center_point
    destination_point = (29.714630473243457, -95.37716122309068) # lat long
    # get the nearest nodes to the locations 
    path = Path("~/dev/GraphRouteOptimizationRL/houston_tx_usa_drive_500.graphml").expanduser()
    G = ox.load_graphml(path)

    origin_node, origin_node_error = ox.distance.nearest_nodes(G, X = origin_point[1], Y = origin_point[0], return_dist = True)
    destination_node, destination_node_error = ox.distance.nearest_nodes(G, X = destination_point[1], Y = destination_point[0], return_dist = True)
    # printing the closest node id to origin and destination points 
    print("origin_node id:", origin_node, "origin_node_error: ", origin_node_error)
    print("destination_node id:", destination_node, "destination_node_error: ", destination_node_error)
    # ox.save_graphml(G, "./houston_tx_usa_drive_500.graphml")
    A = nx.adjacency_matrix(G, weight="length")
    D = A.toarray()
    node_dict = dict()
    reverse_nodes_dict = dict()
    for idx, n in enumerate(G.nodes()):
        node_dict[n] = idx
        reverse_nodes_dict[idx] = n
    
    A = nx.adjacency_matrix(G, weight="length")
    D = A.toarray()
    de_nodes = remove_de_node(D, reverse_nodes_dict)
    print("nodes need to be removed:", de_nodes)
    G.remove_nodes_from(de_nodes)
    A = nx.adjacency_matrix(G, weight="length")
    D = A.toarray()
    num_nodes = len(D)
    print(num_nodes)

    # wandb.init()
    # config.threshold = 0.001
    # config.gamma = 1
    # config.visulize = True
    # config.start_state - 3
    # config.num_epoch = 150
    # config.epsilon = 0.05
    # config.alpha = 0.1
    # config.save_video = True

    value_iteration(threshold=0.001, gamma=1, visualize=True)

    # policy_iteration(gamma=1, visualize=True)

    # sarsa(start_state=3, num_epoch=150, gamma=0.8, epsilon=0.05,
    #         alpha=0.1, visualize=True, save_video=True)

    # sarsa_lambda(start_state=node_dict[origin_node], num_epoch=120, gamma=0.8, epsilon=0.05, alpha=0.1, lamda=0.9,
    #                 visualize=False, save_video=True)
    # q_learning(start_state=node_dict[origin_node], num_epoch=150, gamma=0.8,
    #             epsilon=0.05, alpha=0.1, visualize=False)
