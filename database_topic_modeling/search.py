from numba.core.errors import NumbaDeprecationWarning
import warnings
# Ignore Numba Deprecation Warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)

import json
import matplotlib.pyplot as plt
import mysql.connector
import nltk
import numpy as np
import pandas as pd
import plotly.express as px
import re
import torch
import transformers
import umap
import weaviate

from collections import defaultdict
from datasets import Dataset
from datetime import date, datetime
from scipy.cluster.hierarchy import dendrogram, linkage
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from database_topic_modeling.utils.language_dependant_utils import (
    get_top_n_titles_per_class, 
    language_representation,
    retrieve_top_n_words_per_class,
    tokenization_and_stopword_removal,
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
    tf_idf_matrix,
)
from database_topic_modeling.utils.weaviate_utils import (
    get_nearest_oldids_without_neighbors,
)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def search_concept_for_output(
    search_str: str,
    weaviate_url: str,
    weaviate_auth_api_key: str,
    class_name: str,
    id_column_name: str,
    custom_data: list[str],
    sentence_transformer_name: str='all-mpnet-base-v2',
    language: str='zh-CN',
    num_nearest_neighbors: int=500,
    type_label: str='sentence',
    output_html_file: str="outputs/clustering-search-3d-without-neighbors.html",
) -> tuple[dict, dict]:
    """
    Creates a clustering output graph given access to a database with chinese abstract-size pieces
    of text to produce topic modeling of and a weaviate cluster to store the vectors produced by
    the database. 

    Args:
        search_str - the search string to use to search concepts and produce output on the graph
        weaviate_url - the weaviate url for use as the database
        weaviate_auth_api_key - the weaviate auth api key
        class_name - the name of the class of the weaviate cluster
        id_column_name - the name of the column which was originally used as the id of the dataset
        custom_data - list of column names, the custom data from the database columns to include as
                      well in the legend of the 3d output
        sentence_transformer_name - the name of the sentence transformer to use for encoding 
                                    texts. Must be the same sentence transformer that was
                                    used to produce the weaviate cluster.
        language - the ISO code of the language to use for the sentence transformer, follows
                   the google translate standard of using zh-CN for simplified mandarin and 
                   en for English
        num_nearest_neighbors - the number of nearest neighbors to query for
        type_label - the type of label to give the items in the cluster, one of 'tag' or 'sentence'
        output_html_file - the name of the output html file for the 3d projection with the search results

    Returns:
        tuple(dict, dict) - a tuple of two dictionaries. The first represents the dendrogram json and the
                            second represents the scatterplot json
    """

    search_str = language_representation(search_str, language)

    client = weaviate.Client(
        url=weaviate_url,
        auth_client_secret=weaviate.AuthApiKey(api_key=weaviate_auth_api_key),
    )

    # Get all vectors from weaviate and create an embedding matrix and a full dataset from it
    small_embeds, small_dataset = get_nearest_oldids_without_neighbors(
        sentence_transformer_name, 
        search_str, 
        client, 
        class_name, 
        id_column_name=id_column_name, 
        num_nearest_neighbors=num_nearest_neighbors
    )

    # umap to decrease dimensionality of the datapoints
    small_u = dim_reduction_3d('TSNE', small_embeds, n_components=3, perplexity=30.0, metric="cosine", random_state=42)

    n_clusters = 10

    # agglomerative heirarchical clustering of the datapoints
    clusterer = clustering('agglomerative', small_u, n_clusters=n_clusters, metric='euclidean', linkage='ward')

    # add the cluster labels to our dataset
    small_dataset = small_dataset.add_column('class', clusterer.labels_)

    if type_label == 'tag':
        small_dataset = tokenization_and_stopword_removal(
            small_dataset, 
            language=language, 
            column_to_tokenize="abstract" # TODO: change back to sentencetext
        )

        classes, vocab = create_tokens_per_class_label_and_vocab(small_dataset)

        tf_idf = tf_idf_matrix(classes, vocab)

        labels = retrieve_top_n_words_per_class(tf_idf, classes.keys(), list(vocab), 6, language=language)

        additional_dendro_info = None
    elif type_label == 'sentence':
        dict_titles = get_top_n_titles_per_class(small_u, small_dataset, language=language, title_column_name="title")

        labels = get_sentence_summaries(dict_titles)

        additional_dendro_info = {v1: v2 for v1, v2 in zip(labels.values(), dict_titles.values())}

    small_list_labels = [labels[x] for x in clusterer.labels_]

    # TODO: Change 'publication_date' to column name depending on the name of the column
    articles_per_date = extract_articles_per_date(small_dataset, 'publication_date')

    dendrogram_json = plot_dendrogram(small_u, small_list_labels, n_clusters, type_label, additional_dendro_info, search_str=search_str, output_png_file="clustering-search-dendrogram.png", metric='euclidean', linkage='ward')

    scatter_json = scatter_3d(small_u, small_dataset, small_list_labels, custom_data, output_html_file)

    return articles_per_date, dendrogram_json, scatter_json


if __name__ == '__main__':
    # Example usage

    inp = input('What concept would you like to search? \n')
    articles_per_date, dendrogram_json, scatter_json = search_concept_for_output(
        inp,
        weaviate_url="http://192.168.20.101:8080",
        weaviate_auth_api_key="2d1c4cc7-7175-4965-936f-5439695e9c65",
        class_name="Openalex_cn_new",
        id_column_name="id",
        custom_data=["title", "publication_date"],
        language='en',
        type_label='sentence',
    )
