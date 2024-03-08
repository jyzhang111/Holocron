from numba.core.errors import NumbaDeprecationWarning
import warnings
# Ignore Numba Deprecation Warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)

import json
import multiprocessing
import mysql.connector
import nltk
import numpy as np
import os
import pandas as pd
import plotly.express as px
import re
import torch
import time
import transformers
import umap
import weaviate

from collections import defaultdict
from datasets import Dataset
from datetime import date, datetime
from dotenv import load_dotenv
from scipy.cluster.hierarchy import dendrogram, linkage
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from database_topic_modeling.utils.language_dependant_utils import (
    get_top_n_titles_per_class, 
    language_representation,
    retrieve_top_n_words_per_class,
    tokenization_and_stopword_removal,
    word_cloud,
)
from database_topic_modeling.utils.utils import (
    clustering,
    create_tokens_per_class_label_and_vocab,
    dim_reduction_3d,
    extract_articles_per_date,
    get_nearest_oldids,
    get_sentence_summaries,
    get_small_representations_from_oldids,
    plot_dendrogram,
    scatter_3d,
    topic_modeling_over_time,
    tf_idf_matrix,
)
from database_topic_modeling.utils.opensearch_utils import (
    get_nearest_oldids_without_neighbors,
)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def search_concept_for_output(
    search_str: str,
    opensearch_url: str,
    opensearch_user: str,
    opensearch_pass: str,
    index_name: str,
    google_app_credentials_path: str,
    openai_api_key: str,
    sentence_transformer_name: str='all-mpnet-base-v2',
    language: str='zh-CN',
    num_nearest_neighbors: int=500,
) -> tuple[np.ndarray, np.ndarray, Dataset, list, dict]:
    """
    Creates a clustering output graph given access to a database with chinese abstract-size pieces
    of text to produce topic modeling of and a weaviate cluster to store the vectors produced by
    the database. 

    Args:
        search_str - the search string to use to search concepts and produce output on the graph
        opensearch_url - the opensearch url for use as the database
        opensearch_user - the opensearch username
        opensearch_pass - the opensearch password
        index_name - the name of the index to name the opensearch subcollection
        google_app_credentials_path - the path of the google application credentials
        openai_api_key - the open ai api key
        sentence_transformer_name - the name of the sentence transformer to use for encoding 
                                    texts. Must be the same sentence transformer that was
                                    used to produce the weaviate cluster.
        language - the ISO code of the language to use for the sentence transformer, follows
                   the google translate standard of using zh-CN for simplified mandarin and 
                   en for English
        num_nearest_neighbors - the number of nearest neighbors to query for

    Returns:
        tuple(np.ndarray, Dataset, list, dict) - a tuple of the output
            np.ndarray - full dimensional output of the weaviate vectors
            np.ndarray - dimension reduced three dimensional output of the weaviate vectors
            Dataset - a huggingface dataset corresponding to the output vectors
            list - a list of the tag or sentence labels corresponding to the 3D embeddings
            dict - a dictionary of additional dendro information for the dendrogram
    """

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=google_app_credentials_path

    start_time = time.perf_counter()

    search_str = language_representation(search_str, language)

    if "TRANSFORMERS_CACHE" in os.environ:
        model = SentenceTransformer(sentence_transformer_name, cache_folder=os.environ["TRANSFORMERS_CACHE"])
    else:
        model = SentenceTransformer(sentence_transformer_name)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Get all vectors from weaviate and create an embedding matrix and a full dataset from it
    small_embeds, small_dataset = get_nearest_oldids_without_neighbors(
        model, 
        search_str, 
        opensearch_url, 
        opensearch_user,
        opensearch_pass, 
        index_name, 
        num_nearest_neighbors=num_nearest_neighbors
    )

    time1 = time.perf_counter()
    print(f"Time for weaviate query: {time1-start_time}")

    # umap to decrease dimensionality of the datapoints
    small_u = dim_reduction_3d('TSNE', small_embeds, n_components=3, perplexity=30.0, metric="cosine", random_state=42)

    time2 = time.perf_counter()
    print(f"Time for 3d dimension reduction: {time2-time1}")

    n_clusters = 10

    # agglomerative heirarchical clustering of the datapoints
    clusterer = clustering('agglomerative', small_u, n_clusters=n_clusters, metric='euclidean', linkage='ward')

    time3 = time.perf_counter()
    print(f"Time for clustering: {time3-time2}")

    # add the cluster labels to our dataset
    small_dataset = small_dataset.add_column('class', clusterer.labels_)

    small_dataset = tokenization_and_stopword_removal(
        small_dataset, 
        language=language, 
        column_to_tokenize="content" # TODO: change back to sentencetext
    )

    time4 = time.perf_counter()
    print(f"Time for tokenization_and_stopword_removal: {time4-time3}")

    dict_titles = get_top_n_titles_per_class(small_u, small_dataset, language=language, title_column_name="title")

    time5 = time.perf_counter()
    print(f"Time for getting the top n titles per class: {time5-time4}")

    labels = get_sentence_summaries(dict_titles, openai_api_key)

    time6 = time.perf_counter()
    print(f"Time for getting sentence summaries: {time6-time5}")

    additional_dendro_info = {v1: v2 for v1, v2 in zip(labels.values(), dict_titles.values())}

    small_list_labels = [labels[x] for x in clusterer.labels_]

    return small_embeds, small_u, small_dataset, small_list_labels, additional_dendro_info


if __name__ == '__main__':
    # Example usage

    search_str = input('What concept would you like to search? \n')

    language='en'

    # Load environment variables for passwords
    PARENT_PATH = os.path.dirname(os.path.abspath(__file__))
    dotenv_path = os.path.join(PARENT_PATH, "credentials", ".env")
    load_dotenv(dotenv_path)

    # read through environment variable or secrets manager
    GOOGLE_APPLICATION_CREDENTIALS = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
    OPENSEARCH_URL = os.environ["OPENSEARCH_URL"]
    OPENSEARCH_USER = os.environ["OPENSEARCH_USER"]
    OPENSEARCH_PASS = os.environ["OPENSEARCH_PASS"]

    embeds, u, dataset, list_labels, additional_dendro_info = search_concept_for_output(
        search_str,
        opensearch_url=OPENSEARCH_URL,
        opensearch_user=OPENSEARCH_USER,
        opensearch_pass=OPENSEARCH_PASS,
        index_name="openalex_dehydrated",
        google_app_credentials_path=GOOGLE_APPLICATION_CREDENTIALS,
        openai_api_key=OPENAI_API_KEY,
        language='en',
    )

    time1 = time.perf_counter()

    # TODO: Change 'publication_date' to column name depending on the name of the column
    articles_per_date = extract_articles_per_date(dataset, 'date')

    time2 = time.perf_counter()
    print(f"Time for getting articles: {time2-time1}")

    word_cloud_json = word_cloud(dataset, GOOGLE_APPLICATION_CREDENTIALS, 'en')

    time3 = time.perf_counter()
    print(f"Time for getting word cloud: {time3-time2}")
    
    dendrogram_json = plot_dendrogram(u, dataset, list_labels, ["id", "title"], additional_dendro_info, OPENAI_API_KEY, search_str=search_str, metric='euclidean', linkage='ward')

    time4 = time.perf_counter()
    print(f"Time for getting dendrogram: {time4-time3}")

    topic_modeling_over_time(search_str, embeds, dataset, 'all-mpnet-base-v2', OPENAI_API_KEY, 'date', 'content')

    time5 = time.perf_counter()
    print(f"Time for getting topic modeling: {time5-time4}")

    scatter_json = scatter_3d(u, dataset, list_labels, ["title", "date", "id"])

    time6 = time.perf_counter()
    print(f"Time for getting scatter: {time6-time5}")