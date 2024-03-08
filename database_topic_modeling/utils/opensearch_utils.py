from typing import Optional, Union, Any

import os
import json
import requests
import torch
import numpy as np

from datasets import Dataset
from datetime import date, datetime
from dotenv import load_dotenv
from requests import Session
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util import Retry
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

def get_nearest_oldids_without_neighbors(
    model: SentenceTransformer, 
    search_str: str,
    opensearch_url: str,
    opensearch_user: str,
    opensearch_pass: str,
    index_name: str,
    num_nearest_neighbors: int=5000,
    device: torch.device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
) -> tuple[np.ndarray, Dataset]:
    """
    Gets a small representation of the nearest neighbor embeddings as well as the dataset

    Args:
        model - the SentenceTransformer to use for encoding. Must be the same sentence transformer
                that was used to produce the weaviate cluster.
        search_str - the string to search 
        opensearch_url - the opensearch url for use as the database
        opensearch_user - the opensearch username
        opensearch_pass - the opensearch password
        index_name - the index_name (database name) of the vectors in the opensearch database
        num_nearest_neighbors - the number of nearest neighbors to query for

    Returns:
        tuple[np.ndarray, Dataset] -
            np.ndarray - a smaller representation of embeds with just the embeds from oldids
            Dataset - a smaller representation of the original dataset with just the dataset
                      values from oldids
    """

    vector = [float(f) for f in model.encode(search_str)]

    payload = {
        "size": num_nearest_neighbors,
        "query": {
            "knn": {
                "vector": {
                "vector": vector,
                "k": num_nearest_neighbors,
                }
            }
        }
    }
    json_payload = json.dumps(payload)

    response = requests.get(opensearch_url+'/'+index_name+'/_search', data=json_payload, headers={'Content-Type': 'application/json'}, auth=(opensearch_user, opensearch_pass))
    if response.status_code != 200:
        raise RuntimeError(response.text)

    results = json.loads(response.text)

    if len(results["hits"]["hits"]) == 0:
        raise RuntimeError("Returned no vectors from query, unable to perform search.")

    cols = [k for k in results["hits"]["hits"][0]["_source"].keys() if k != "vector"]

    dict_ = {}
    for col in cols:
        dict_[col] = []
    dict_['id'] = []

    vector_dimension = len(vector)
    embeds = np.zeros((len(results["hits"]["hits"]), vector_dimension))

    for i, d in enumerate(results["hits"]["hits"]):
        for k, v in d["_source"].items():
            if k == 'vector':  # TODO: Change back to oldid
                continue
            dict_[k].append(v)
        embeds[i] = d["_source"]['vector']
        dict_['id'].append(d['_id'])

    ds = Dataset.from_dict(dict_)

    return embeds, ds
