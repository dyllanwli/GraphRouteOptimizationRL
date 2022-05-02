
"""Get node elevations and calculate edge grades."""

import multiprocessing as mp
import time

import networkx as nx
import numpy as np
import pandas as pd
import requests

from osmnx import downloader

def add_node_elevations_opentopo(
    G, max_locations_per_batch=100, pause_duration=0, precision=3
):  # pragma: no cover
    """
    Add `elevation` (meters) attribute to each node using a web service.

    This uses the Opentopodata Maps Elevation API and requires an API key. For a
    free, local alternative, see the `add_node_elevations_raster` function.
    See also the `add_edge_grades` function.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        input graph

    Returns
    -------
    G : networkx.MultiDiGraph
        graph with node elevation attributes
    """
    # different elevation API endpoints formatted ready for use
    endpoints = {
        "self": "http://10.42.1.0:5000/v1/srtm90m?locations={}",
        # "aster30m": "https://api.opentopodata.org/v1/aster30m?locations={}",
        # "srtm30m": "https://api.opentopodata.org/v1/srtm30m?locations={}",
        # "airmap": "https://api.airmap.com/elevation/v1/ele?points={}",
    }
    provider = ["self"]

    # make a pandas series of all the nodes' coordinates as 'lat,lng'
    # round coordinates to 5 decimal places (approx 1 meter) to be able to fit
    # in more locations per API call
    node_points = pd.Series(
        {node: f'{data["y"]:.5f},{data["x"]:.5f}' for node, data in G.nodes(data=True)}
    )
    n_calls = int(np.ceil(len(node_points) / max_locations_per_batch))
    print(f"Requesting node elevations from the API in {n_calls} calls")

    # break the series of coordinates into chunks of size max_locations_per_batch
    # API format is locations=lat,lng|lat,lng|lat,lng|lat,lng...
    results = []
    for i in range(0, len(node_points), max_locations_per_batch):
        chunk = node_points.iloc[i : i + max_locations_per_batch]
        elevation_provider = np.random.choice(provider)

        url_template = endpoints[elevation_provider]

        locations = "|".join(chunk)
        url = url_template.format(locations)

        # check if this request is already in the cache (if global use_cache=True)
        cached_response_json = downloader._retrieve_from_cache(url)
        if cached_response_json is not None:
            response_json = cached_response_json
        else:
            try:
                # request the elevations from the API
                print(f"Requesting node elevations", i)
                time.sleep(pause_duration)
                if elevation_provider == "airmap":
                    headers = {
                        # "X-API-Key": api_key,
                        "Content-Type": "application/json; charset=utf-8",
                    }
                    response = requests.get(url, headers=headers)
                else:
                    response = requests.get(url)
                response_json = response.json()
                downloader._save_to_cache(url, response_json, response.status_code)
            except Exception as e:
                print(e)
                print(f"Server responded with {response.status_code}: {response.reason}")

        # append these elevation results to the list of all results
        if elevation_provider == "airmap":
            results.extend(response_json["data"])
        else:
            results.extend([r['elevation'] for r in response_json["results"]])

    # sanity check that all our vectors have the same number of elements
    if not (len(results) == len(G) == len(node_points)):
        raise Exception(
            f"Graph has {len(G)} nodes but we received {len(results)} results from elevation API"
        )
    else:
        print(
            f"Graph has {len(G)} nodes and we received {len(results)} results from elevation API"
        )

    # add elevation as an attribute to the nodes
    df = pd.DataFrame(node_points, columns=["node_points"])

    df["elevation"] = results
        
    df["elevation"] = df["elevation"].round(precision)
    nx.set_node_attributes(G, name="elevation", values=df["elevation"].to_dict())
    print("Added elevation data from web service to all nodes.")

    return G