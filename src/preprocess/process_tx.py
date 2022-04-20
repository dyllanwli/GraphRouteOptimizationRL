from pathlib import Path
import osmnx as ox
import networkx as nx
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
from src.utils import plot_path
import wandb
from tqdm import tqdm
from src.shortest_path import ShortestPath
from collections import defaultdict
import pandas as pd

# process tx flood
df = pd.read_csv(str(Path.home()) + "/dev/GraphRouteOptimizationRL/datasets/tx_flood_record.csv")
df = df[df['governing_location'] == 'TX, USA']
df = df[['Latitude', 'Longitude', 'stage_ft', 'name']]
df[df['stage_ft'].notnull()]
df.to_csv("tx_flood.csv", index=None)