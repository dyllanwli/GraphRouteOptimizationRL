from collections import namedtuple, Counter
import numpy as np
import networkx as nx
# from geopy.distance import great_circle
import plotly.graph_objects as go

# hyperparameters
CONFIG = {
    "PWD": "../",
    "STATE_DIM": 256,
    "HISTORY_DIM": 680,
    "ACTION_SPACE": 8,
    "EPS_START": 1,
    "EPS_END": 0.1,
    "EPE_DECAY": 1000,
    "REPLAY_MEMORY_SIZE": 10000,
    "BATCH_SIZE": 128,
    "EMBEDDING_DIM": 128,
    "GAMMA": 0.99,
    "TARGET_UPDATE_FREQ": 1000,
    "MAX_STEP": 50,
    "MAX_STEP_TEST": 50,
}


Transition = namedtuple(
    'Transition', ('state', 'action', 'history', 'next_state', 'reward'))


def distance(e1, e2):
    return np.sqrt(np.sum(np.square(e1 - e2)))


def compare(v1, v2):
    return np.sum(v1 == v2)

def path_clean(path):
    rel_ents = path.split(' -> ')
    relations = []
    entities = []
    for idx, item in enumerate(rel_ents):
        if idx % 2 == 0:
            relations.append(item)
        else:
            entities.append(item)
    entity_stats = Counter(entities).items()
    duplicate_ents = [item for item in entity_stats if item[1] != 1]
    duplicate_ents.sort(key=lambda t: t[1], reverse=True)
    for item in duplicate_ents:
        ent = item[0]
        ent_idx = [i for i, x in enumerate(rel_ents) if x == ent]
        if len(ent_idx) != 0:
            min_idx = min(ent_idx)
            max_idx = max(ent_idx)
            if min_idx != max_idx:
                rel_ents = rel_ents[:min_idx] + rel_ents[max_idx:]
    return ' -> '.join(rel_ents)


def prob_norm(probs):
    return probs / sum(probs)


if __name__ == '__main__':
    print(prob_norm(np.array([1, 1, 1])))

def plot_path(lat, long, origin_point, destination_point):

    flooded_point = (29.719234316403067, -95.37637299903416)
    # adding the lines joining the nodes
    fig = go.Figure(go.Scattermapbox(
        name = "Path",
        mode = "markers+lines",
        lon = long,
        lat = lat,
        marker = {'size': 10},
        line = dict(width = 4.5, color = 'blue')))
        
    # adding destination marker
    fig.add_trace(go.Scattermapbox(
        name = "Destination",
        mode = "markers",
        lon = [destination_point[1]],
        lat = [destination_point[0]],
        marker = {'size': 12, 'color':'green'}))

    # adding source marker
    fig.add_trace(go.Scattermapbox(
        name = "Source",
        mode = "markers",
        lon = [origin_point[1]],
        lat = [origin_point[0]],
        marker = {'size': 12, 'color':"red"}))

    # adding flood marker
    fig.add_trace(go.Scattermapbox(
        name = "Flood Points",
        mode = "markers",
        lon = [flooded_point[1]],
        lat = [flooded_point[0]],
        marker = {'size': 12, 'color':"blue"}))
     
    
    # getting center for plots:
    lat_center = np.mean(lat)
    long_center = np.mean(long)
    # defining the layout using mapbox_style
    fig.update_layout(mapbox_style="stamen-terrain",
        mapbox_center_lat = 30, mapbox_center_lon=-80)
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0},
                      mapbox = {
                          'center': {'lat': lat_center, 'lon': long_center},
                          'zoom': 13})
    fig.show()