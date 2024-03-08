from typing import Optional, Union, Any

import datetime
import dateutil
import hdbscan
import json
import matplotlib.pyplot as plt
import mysql.connector
import numpy as np
import openai
import os
import pandas as pd
import plotly.express as px
import pytz
import re
import sklearn
import time
import torch
import umap

from collections import Counter
from collections import defaultdict
from datasets import Dataset
from dotenv import load_dotenv
from numpy.linalg import norm
from mpl_toolkits.axes_grid1.axes_divider import make_axes_area_auto_adjustable
from scipy.cluster.hierarchy import dendrogram, to_tree
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
from tqdm.auto import tqdm

def batch_encode_texts(
    model: SentenceTransformer, 
    dataset: Dataset, 
    batch_size: int=32,
    device: torch.device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
) -> np.ndarray:
    """
    Returns the batch encoded texts in the form of a numpy array for the dataset's sentencetext
    column given a SentenceTransformer model.

    Args:
        model - the SentenceTransformer to use for encoding
        dataset - the Huggingface dataset containing the sentencetext column to encode.
        batch_size - the batch size to use for encoding

    Returns:
        np.ndarray - the embeddings of the encoded text
    """

    n = len(dataset['oldid'])
    embeds = np.zeros((n, model.get_sentence_embedding_dimension()))

    # Batch encode sentences
    for i in tqdm(range(0, n, batch_size)):
        i_end = min(i+batch_size, n)
        batch = dataset['sentencetext'][i:i_end]
        batch_embed = model.encode(batch, device=device)
        embeds[i:i_end, :] = batch_embed

    return embeds

def clustering(model_name: str, embeds: np.ndarray, **kwargs) -> sklearn.base.ClusterMixin:
    """
    The purpose of this function is to provide some sensible defaults for clustering.

    Args:
        model_name - one of 'agglomerative' or 'hdbscan'
        embeds - np.ndarray of the embeddings as input into clustering, can be three dimensional
                 or the dimensionality of the text embeddings
        kwargs - keyword arguments for the model

    Returns:
        sklearn.base.ClusterMixin - the clustering model to return
    """
    if model_name == "agglomerative":
        agg_defaults = {'n_clusters': 6, 'metric': "euclidean", 'linkage': "ward"}
        # n_clusters=5, metric='cosine', linkage='average'
        for k, v in kwargs.items():
            agg_defaults[k] = v
        clusterer = AgglomerativeClustering(**agg_defaults)
        clusterer.fit(embeds)
    elif model_name == "hdbscan":
        hdbscan_defaults = {'min_cluster_size': 80, 'min_samples': 40}
        for k, v in kwargs.items():
            hdbscan_defaults[k] = v
        clusterer = hdbscan.HDBSCAN(**hdbscan_defaults)
        clusterer.fit(embeds)

    return clusterer

def create_dendrogram_json(
    linkage_data: np.ndarray, 
    dend: dict, 
    leave_dict: Optional[dict] = None,
    additional_dendro_info: Optional[dict] = None,
    openai_api_key: Optional[str] = None,
    search_str: Optional[str] = None,
) -> dict:
    """
    Gets a json representation of the dendrogram representing the data in linkage_data and dend

    Args:
        linkage_data - np.ndarray of the linkage matrix contained in linkage_data
        dend - dictionary containing icoords, leaves, and such representing the dendrogram
        leave_dict - dictionary of leaves to the titles of the corresponding leaves
        additional_dendro_info - dictionary of leaf labels to additional info, for producing inner nodes of dendrogram
        openai_api_key - the open ai api key, must be set if additional_dendro_info is set
        search_str - the search string to use to search concepts and produce output on the graph

    Returns:
        dict - a json output file of the dendrogram
    """
    start_time = time.perf_counter()

    def add_node(node, parent ):
        # if root, just create node with name
        if "name" not in parent:
            parent["name"] = node.id 
            newNode = parent
        else:
            newNode = dict( name=node.id )
            if "children" not in parent:
                parent["children"] = [newNode]
            else:
                parent["children"].append( newNode )
    
        # Recursively add the current node's children
        if node.left: add_node( node.left, newNode )
        if node.right: add_node( node.right, newNode )
    
    T = to_tree( linkage_data , rd=False )
    dendro_json = dict()
    add_node( T, dendro_json )

    def flatten(l):
        return [item for sublist in l for item in sublist]
    X = flatten(dend['icoord'])
    Y = flatten(dend['dcoord'])
    leave_coords = [(x,y) for x,y in zip(X,Y) if y==0]
    
    # in the dendogram data structure,
    # leave ids are listed in ascending order according to their x-coordinate
    order = np.argsort([x for x,y in leave_coords])
    id_to_coord = dict(zip(dend['leaves'], [leave_coords[idx] for idx in order])) # <- main data structure

    all_ids = []

    def delete_children(dendro_json):
        all_ids.append(dendro_json["name"])
        if dendro_json["name"] in id_to_coord:
            del dendro_json["children"]
        if "children" in dendro_json:
            for node in dendro_json["children"]:
                delete_children(node)
    
    delete_children(dendro_json)

    children_to_parent_coords = dict()
    for i, d in zip(dend['icoord'], dend['dcoord']):
        x = (i[1] + i[2]) / 2
        y = d[1] # or d[2]
        parent_coord = (x, y)
        left_coord = (i[0], d[0])
        right_coord = (i[-1], d[-1])
        children_to_parent_coords[(left_coord, right_coord)] = parent_coord
    
    # traverse tree from leaves upwards and populate mapping ID -> (x,y)
    root_node, node_list = to_tree(linkage_data, rd=True)
    ids_left = all_ids
    
    while len(ids_left) > 0:
    
        for ii, node_id in enumerate(ids_left):
            node = node_list[node_id]
            if (node.left.id in id_to_coord) and (node.right.id in id_to_coord):
                left_coord = id_to_coord[node.left.id]
                right_coord = id_to_coord[node.right.id]
                id_to_coord[node_id] = children_to_parent_coords[(left_coord, right_coord)]
    
        ids_left = [node_id for node_id in all_ids if not node_id in id_to_coord]

    max_y = max(Y)

    openai.api_key = openai_api_key

    def add_height_to_node(dendro_json, depth, list_prompts):
        dendro_json["y"] = (max_y - id_to_coord[dendro_json["name"]][1])/max_y*100.0
        dendro_json["x"] = id_to_coord[dendro_json["name"]][0]
        tags_or_titles = []
        if "children" in dendro_json:
            for node in dendro_json["children"]:
                tags_or_titles.extend(add_height_to_node(node, depth+1, list_prompts))

        if not (depth == 0 and search_str) and additional_dendro_info and leave_dict and dendro_json["name"] not in leave_dict:
            if depth <= 1:
                prompt = "Summarize the following items into a category title less than 4 words:"
            elif depth > 1:
                prompt = "Summarize the following items into a category title less than 6 words:"

            for t in tags_or_titles:
                prompt+=(" " + t)

            list_prompts.append(prompt)

        if "children" in dendro_json:
            return tags_or_titles
        else:
            if additional_dendro_info:
                return additional_dendro_info[leave_dict[dendro_json["name"]]]
            else:
                return dendro_json["name"]

    list_prompts = []

    add_height_to_node(dendro_json, 0, list_prompts)

    responses = []
    if list_prompts:
        responses = openai.Completion.create(
            model="gpt-3.5-turbo-instruct",
            prompt=list_prompts,
            temperature=0.0,
        ).choices

    def add_labels_to_node(dendro_json, depth):
        if "children" in dendro_json:
            for node in dendro_json["children"]:
                add_labels_to_node(node, depth+1)

        if depth == 0 and search_str:
            dendro_json["name"] = search_str
        elif additional_dendro_info and leave_dict and dendro_json["name"] not in leave_dict:
            if depth <= 1:
                num_items = 4
            elif depth > 1:
                num_items = 6

            response = responses.pop(0)

            message = response.text
            message = message.replace('"', '')
            message = re.sub('\s+', ' ', message);
            items = message.split(" ")
            items = [item for item in items if item]
            items = items[:num_items]
            message = " ".join(items)
            message = message.strip(',')

            dendro_json["name"] = message
        elif leave_dict and dendro_json["name"] in leave_dict:
            dendro_json["name"] = leave_dict[dendro_json["name"]]

    add_labels_to_node(dendro_json, 0)

    # print(dendro_json)
    # print(f"Time to run dendrogram = {time.perf_counter() - start_time}")

    return dendro_json

def create_linkage_matrix(model: AgglomerativeClustering) -> np.ndarray:
    """
    Creates linkage matrix representing the agglomerative clustering on the fit model

    Args:
        model - Agglomerative clustering model fit on the data

    Returns:
        np.ndarray - linkage matrix representing the data in the agglomerative clustering
    """

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    return linkage_matrix

def create_tokens_per_class_label_and_vocab(dataset: Dataset) -> tuple[dict, set]:
    """
    Creates a dict with class labels as keys to the tokens in the class and a vocab set.

    Args:
        dataset - Huggingface dataset that contains the tokens as well as a class label per row

    Returns:
        tuple[dict, set] - the dict with class labels to tokens in the class as well as the vocab set
    """
    classes = {label: defaultdict(lambda: 0) for label in set(dataset['class'])}
    # add tokenized sentences to respective class
    for row in dataset:
        for token in row['tokens']:
            classes[row['class']][token] += 1

    # Create the vocabs (both overall and class)
    vocab = set()
    for c in classes.keys():
        vocab = vocab.union(set(classes[c].keys()))

    return classes, vocab

def dim_reduction_3d(model_name: str, embeds: np.ndarray, **kwargs) -> np.ndarray:
    """
    The purpose of this function is to provide some sensible defaults for dimensionality
    reduction.

    Args:
        model_name - one of 'TSNE' or 'umap'
        embeds - np.ndarray of the embeddings as input into dimensionality reduction
        kwargs - keyword arguments for the model

    Returns:
        np.ndarray - the 3d dimensionality reduction of the embeddings
    """
    if model_name == "TSNE":
        TSNE_defaults = {'n_components': 3, 'perplexity': 30.0, 'metric': "cosine", 'random_state': 42}
        for k, v in kwargs.items():
            TSNE_defaults[k] = v
        fit = TSNE(**TSNE_defaults)
        u = fit.fit_transform(embeds)
    elif model_name == "umap":
        umap_defaults = {'n_neighbors': 5, 'n_components': 3, 'min_dist': 0.05, 'random_state': 42}
        for k, v in kwargs.items():
            umap_defaults[k] = v
        fit = umap(**umap_defaults)
        u = fit.fit_transform(embeds)

    return u

def extract_articles_per_date(dataset: Dataset, date_column_name: str) -> dict:
    """
    Extracts the number of articles per date range from the given dataset.

    Args:
        dataset - Huggingface dataset that contains the date column of the items to extract
        date_column_name - column name of the date column

    Returns:
        dict - json that contains the starts of the date ranges and the number of articles
               for that date
    """
    dates = dataset[date_column_name]

    if len(dates) > 0 and type(dates[0]) is str:
        for i, d in enumerate(dates):
            dates[i] = dateutil.parser.parse(d).replace(tzinfo=pytz.UTC)

    articles_per_date = {"date": [], "num_articles": []}

    dates.sort()
    if len(dates) > 0:
        startdate = datetime.datetime(dates[0].year, dates[0].month, 1).replace(tzinfo=pytz.UTC)
        articles_per_date["date"].append(startdate.isoformat())
        articles_per_date["num_articles"].append(0)
    for d in dates:
        border_date = startdate + dateutil.relativedelta.relativedelta(months=1)
        if d >= border_date:
            startdate = border_date
            articles_per_date["date"].append(startdate.isoformat())
            articles_per_date["num_articles"].append(0)
        articles_per_date["num_articles"][-1] += 1

    return articles_per_date

def get_cols_and_rows(
    client_or_connection: mysql.connector.connection_cext.CMySQLConnection,
    tablename: str,
) -> tuple[list[tuple], list[tuple]]:
    """
    Gets the cols and rows from a database client or connection object.

    Args:
        client_or_connection - the mysql connection object or the Clickhouse client object from which
                               to extract rows and cols
        tablename - the tablename of the database to query

    Returns:
        tuple[list[tuple], list[tuple]] - 
            list of tuples of the cols, list of tuples of the rows
    """

    cursor = client_or_connection.cursor()
    cursor.execute(f"SHOW columns FROM {tablename}")
    cols = [col[0] for col in cursor.fetchall()]

    cursor.execute(f"SELECT * from {tablename}")
    rows = cursor.fetchall()
    return cols, rows

def get_nearest_oldids(
    sentence_transformer_name: str, 
    search_str: str,
    embeds: np.ndarray, 
    dataset: Dataset,
    id_column_name: str,
    num_nearest_neighbors: int = 100,
) -> set[Any]:
    """
    Gets the nearest neighbor oldids from the embeddings in the dataset

    Args:
        sentence_transformer_name - the name of the sentence transformer to use for encoding 
                                    texts. Must be the same sentence transformer that was
                                    used to produce the weaviate cluster.
        search_str - the string to search 
        embeds - np.ndarray of the embeddings produced from the sentencetransformer
        dataset - huggingface dataset that contains the oldids of the embeddings to return
        id_column_name - the id column of the oldids to retrieve from the dataset
        num_nearest_neighbors - the number of nearest neighbors to query for

    Returns:
        set(Any) - the oldids which represent the nearest neighbors
    """
    model = SentenceTransformer(sentence_transformer_name)
    vector = model.encode(search_str)
    cosine = np.dot(embeds,vector)/(norm(embeds, axis=1)*norm(vector))

    nearest_values = np.argsort(cosine)[-num_nearest_neighbors:]
    oldids = set([dataset[int(i)][id_column_name] for i in nearest_values])

    return oldids

def get_avg_similarity(vector: np.ndarray, embeds: np.ndarray) -> float:
    """
    The purpose of this function is to get the average similarity between a vector and a
    matrix of embeddings.

    Args:
        vector - vector to compare similarity of
        embeds - np.ndarray of the embeddings

    Returns:
        float - the average similarity of the vector to embeds
    """
    cosine = np.dot(embeds, vector)/(norm(embeds, axis=1)*norm(vector))
    return float(cosine.mean())

def get_sentence_summaries(
    dict_titles: dict,
    openai_api_key: str,
) -> dict:
    """
    Returns a sentence summary of the titles given the input sentences

    Args:
        dict_titles - a dictionary of the class to titles per class
        openai_api_key - the open api key

    Returns:
        dict - a dictionary with classes mapped to their sentence summaries
    """

    openai.api_key = openai_api_key

    sentence_summaries = {}
    stringified_prompts_array = ""

    list_prompts = []

    for d_key, list_titles in dict_titles.items():
        prompt = "Summarize the following collection of items into one phrase less than 15 words without listing items:"

        for title in list_titles:
            prompt+=(" " + title)

        list_prompts.append(prompt)

    responses = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=list_prompts,
        temperature=0.0,
        max_tokens=20,
    ).choices

    responses = [elem.text for elem in responses]

    for message, d_key in zip(responses, list(dict_titles.keys())):
        message = re.sub('\s+', ' ', message)
        items = message.split(" ")
        items = [item for item in items if item]
        items.insert(7, '<br>')
        last_item = 16
        if len(items) > 16:
            for i in range(16):
                if "," in items[i]:
                    last_item = i+1
            items[last_item-1].replace(",", "")
        message = " ".join(items[:last_item])
        message = message.replace('"', '')
        sentence_summaries[d_key] = message

    return sentence_summaries


def get_small_representations_from_oldids(
    oldids: set[Any],
    u: np.ndarray,
    embeds: np.ndarray,
    dataset: Dataset,
    id_column_name: str,
) -> tuple[np.ndarray, np.ndarray, Dataset, dict]:
    """
    Get smaller representations of the embeddings matrix and the dataset to match the oldids
    in the input

    Args:
        oldids - the oldids representing the documents that we want to select specific information
                 for
        u - np.ndarray of the 3d vectors that represent the embeddings 
        embeds - np.ndarray of the embeddings produced from the sentencetransformer
        dataset - huggingface dataset that contains the oldids of the embeddings to return
        id_column_name - the id column of the oldids to retrieve from the dataset

    Returns:
        tuple[np.ndarray, np.ndarray, Dataset, dict] - 
            np.ndarray - a smaller representation of u with just the u from oldids
            np.ndarray - a smaller representation of embeds with just the embeds from oldids
            Dataset - a smaller representation of the original dataset with just the dataset
                      values from oldids
            dict - a dictionary with oldids mapped to the index into the small representations
    """
    small_u = np.zeros((len(oldids), 3))
    small_embeds = np.zeros((len(oldids), embeds.shape[1]))
    small_dataset = []
    small_oldid_index_dict = {}

    j = 0
    for i, d in enumerate(dataset):
        if d[id_column_name] in oldids:
            small_u[j] = u[i]
            small_embeds[j] = embeds[i]
            small_dataset.append(d)
            small_oldid_index_dict[d[id_column_name]] = j
            j+=1

    small_dataset = Dataset.from_pandas(pd.DataFrame(data=small_dataset))

    return small_u, small_embeds, small_dataset, small_oldid_index_dict

def plot_dendrogram(
    embeds: np.ndarray, 
    dataset: Dataset,
    list_labels: list[str], 
    custom_data: list[str],
    additional_dendro_info: Optional[dict]=None,
    openai_api_key: Optional[str]=None,
    search_str: Optional[str]=None,
    **kwargs,
) -> dict:
    """
    The purpose of this function is to create a dendrogram using agglomerative clustering, yes, 
    this function only works with agglomerative clustering right now.

    Args:
        embeds - np.ndarray of the embeddings as input into clustering, can be three dimensional
                 or the dimensionality of the text embeddings
        dataset - Huggingface dataset that contains all the articles
        list_labels - list of labels in list format for the embeddings
        custom_data - the list of columns in dataset, the original database, that will also be included
                      in the figure
        additional_dendro_info - dictionary of leaf labels to additional info, for producing inner nodes of dendrogram
        openai_api_key - the open api key, must be provided if additional_dendro_info is passed
        search_str - the search string to use to search concepts and produce output on the graph
        kwargs - keyword arguments for the agglomerative clustering method

    Returns:
        dict - a json output file of the dendrogram
    """
    if additional_dendro_info and not openai_api_key:
        raise ValueError("openai_api_key must be passed if additional_dendro_info is set.")

    model_name = 'agglomerative'

    n_clusters = len(set(list_labels))

    if model_name == "agglomerative":
        agg_defaults = {'metric': "euclidean", 'linkage': "ward"}
        for k, v in kwargs.items():
            agg_defaults[k] = v

        # create a dictionary of the count of occurrences of an item to a string label of the item
        ivl_dict = {}
        for k, v in Counter(list_labels).items():
            str_ = k.replace('<br>', '\n')
            ivl_dict[v] = str_
            if additional_dendro_info:
                additional_dendro_info[str_] = additional_dendro_info[k]

        cluster_result = AgglomerativeClustering(distance_threshold=0, n_clusters=None, **agg_defaults)
        cluster_result.fit(embeds)

        linkage_data = create_linkage_matrix(cluster_result)
        R = dendrogram(linkage_data, truncate_mode='lastp', p=n_clusters, no_plot=True)

        R_dict = {leaf: ivl_dict[int(re.sub("[()]", "", ivl))] for ivl, leaf in zip(R['ivl'], R['leaves'])}

        # create a dendrogram of the json data
        dendrogram_json = create_dendrogram_json(linkage_data, R, R_dict, additional_dendro_info, openai_api_key, search_str)

    class_to_ids = defaultdict(lambda: {c: [] for c in custom_data})
    for i in range(len(dataset)):
        for c in custom_data:
            class_to_ids[list_labels[i].replace('<br>', '\n')][c].append(dataset[i][c])

    def add_ids(dendro_json, class_to_ids):
        if "children" in dendro_json:
            for node in dendro_json["children"]:
                add_ids(node, class_to_ids)

        if dendro_json["name"] in class_to_ids:
            for c in custom_data:
                dendro_json[c] = class_to_ids[dendro_json["name"]][c]

    add_ids(dendrogram_json, class_to_ids)

    return dendrogram_json

def scatter_3d(
    vectors_3d: np.ndarray, 
    dataset: Dataset, 
    list_labels: list[str], 
    custom_data: list[str],
) -> dict:
    """
    Creates a 3d scatterplot of the embeddings in vectors_3d, as well as including a list of labels
    and a list of custom data to display

    Args:
        vectors_3d - a 3d matrix of vectors to plot
        dataset - huggingface dataset that contains the custom_data which will be included in a legend
                  for the datapoints
        list_labels - list of labels in list format for the embeddings
        custom_data - the list of columns in dataset, the original database, that will also be included
                      in the figure

    Returns:
        dict - a json output file of the plotly graph
    """
    fig = px.scatter_3d(
        x=vectors_3d[:,0], y=vectors_3d[:,1], z=vectors_3d[:,2],
        color=list_labels,
        custom_data=[dataset[c] for c in custom_data],
    )
    fig.update_traces(
        hovertemplate="<br>".join([
            *[custom_data[i]+": %{customdata["+str(i)+"]}" for i in range(len(custom_data))]
        ])
    )

    return json.loads(fig.to_json())

def topic_modeling_over_time(
    search_str: str,
    embeds: np.ndarray, 
    dataset: Dataset, 
    sentence_transformer_name: str,
    openai_api_key: str,
    date_column_name: str,
    content_column_name: str,
) -> dict:
    """
    Extracts the number of articles per date range from the given dataset.

    Args:
        search_str - the search string to use to search concepts and produce output on the graph
        embeds - np.ndarray of the embeddings as extracted from weaviate
        dataset - Huggingface dataset that contains the date column of the items to extract
        sentence_transformer_name - the sentence transformer name
        openai_api_key - the open ai api key
        date_column_name - column name of the date column
        content_column_name - column name of the content column

    Returns:
        dict - json that contains the starts of the date ranges and the number of articles
               for that date
    """
    dates = dataset[date_column_name]

    if len(dates) > 0 and type(dates[0]) is str:
        for i, d in enumerate(dates):
            dates[i] = dateutil.parser.parse(d).replace(tzinfo=pytz.UTC)

    articles_per_date = {"date": [], "num_articles": [], "embeds": [], "contents": []}

    indicies = sorted(range(len(dates)), key=lambda k: dates[k])
    if len(dates) > 0:
        startdate = datetime.datetime(dates[0].year, dates[0].month, 1).replace(tzinfo=pytz.UTC)
        articles_per_date["date"].append(startdate.isoformat())
        articles_per_date["num_articles"].append(0)
        articles_per_date["embeds"].append(np.empty((0, embeds.shape[1])))
        articles_per_date["contents"].append([])
    for i in indicies:
        d = dates[i]
        border_date = startdate + dateutil.relativedelta.relativedelta(months=1)
        if d >= border_date:
            startdate = border_date
            articles_per_date["date"].append(startdate.isoformat())
            articles_per_date["num_articles"].append(0)
            articles_per_date["embeds"].append(np.empty((0, embeds.shape[1])))
            articles_per_date["contents"].append([])
        articles_per_date["num_articles"][-1] += 1
        articles_per_date["embeds"][-1] = np.append(articles_per_date["embeds"][-1], embeds[i].reshape(1, -1), axis=0)
        articles_per_date["contents"][-1].append(dataset[content_column_name][i])

    indicies = sorted(range(len(articles_per_date["num_articles"])), key=lambda k: articles_per_date["num_articles"][k], reverse=True)
    list_prompts = []
    for i in indicies[:5]:
        prompt = f"Extract one related theme out of the following articles as a category that is subtly different from {search_str} using 2 words: \n\n"

        j = 0

        for j, content in enumerate(articles_per_date["contents"][i]):
            prompt+=(content+"\n")
            j += 1
            if j == 9:
                break

        list_prompts.append(prompt)

    openai.api_key = openai_api_key

    responses = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=list_prompts,
        temperature=0.0,
        max_tokens=20,
    ).choices

    responses = [elem.text.strip().strip('"').strip("'") for elem in responses]
    responses = set(responses)
    responses = list(responses)

    if "TRANSFORMERS_CACHE" in os.environ:
        model = SentenceTransformer(sentence_transformer_name, cache_folder=os.environ["TRANSFORMERS_CACHE"])
    else:
        model = SentenceTransformer(sentence_transformer_name)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    vectors = model.encode(responses, device=device)

    avg_similarity_per_date = {"date": [], "avg_similarity_per_date": {t: [] for t in responses}}

    for date, emb in zip(articles_per_date["date"], articles_per_date["embeds"]):
        avg_similarity_per_date["date"].append(date)
        for t, v in zip(responses, vectors):
            avg_similarity_per_date["avg_similarity_per_date"][t].append(get_avg_similarity(v, emb))

    # Compute the normalized average similarity per date
    for t in responses:
        data = np.array(avg_similarity_per_date["avg_similarity_per_date"][t])
        avg_similarity_per_date["avg_similarity_per_date"][t] = list((data-data.mean())/data.std())

    plot_d = {"date": avg_similarity_per_date["date"]}
    for t in avg_similarity_per_date["avg_similarity_per_date"].keys():
        plot_d[t] = avg_similarity_per_date["avg_similarity_per_date"][t]
    df = pd.DataFrame(plot_d)
    fig = px.line(df, x="date", y=[t for t in avg_similarity_per_date["avg_similarity_per_date"].keys()])
    fig.write_image("yourfile.png")

    return avg_similarity_per_date

def tf_idf_matrix(classes: dict, vocab: set) -> tuple[dict, np.ndarray, np.ndarray]:
    """
    Creates a term to index dictionary, and tf as well as an tf_idf matrix of the tokens per class label

    Args:
        classes - a dict with class labels as keys to the tokens in the class and a vocab set
        vocab - set of tokens that contains the vocab

    Returns:
        tuple[dict, np.ndarray, np.ndarray]
            dict - a term to index dictionary
            np.ndarray - the tf matrix
            np.ndarray - the tf_idf matrix
    """
    term_index_dict = {}

    # Get word frequency per class (TF)
    tf = np.zeros((len(classes.keys()), len(vocab)))

    for c, _class in enumerate(classes.keys()):
        for t, term in enumerate(tqdm(vocab)):
            term_index_dict[term] = t
            tf[c, t] = classes[_class][term]

    idf = np.zeros((1, len(vocab)))

    # calculate average number of words per class
    A = tf.sum() / tf.shape[0]

    for t, term in enumerate(tqdm(vocab)):
        # frequency of term t across all classes
        f_t = tf[:,t].sum()
        # calculate IDF
        idf_score = np.log(1 + (A / f_t))
        idf[0, t] = idf_score

    tf_idf = tf*idf

    return term_index_dict, tf, tf_idf
