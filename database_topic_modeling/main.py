from typing import Optional, Union, Any

from numba.core.errors import NumbaDeprecationWarning
import warnings
# Ignore Numba Deprecation Warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)

import clickhouse_connect
import json
import mysql.connector
import nltk
import numpy as np
import os
import plotly.express as px
import re
import torch
import transformers
import weaviate

from collections import defaultdict
from datasets import Dataset
from datetime import date, datetime
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from database_topic_modeling.create_huggingface_dataset import (
    create_huggingface_dataset,
    create_huggingface_dataset_and_embeds,
)
from database_topic_modeling.utils.language_dependant_utils import (
    retrieve_top_n_words_per_class,
    tokenization_and_stopword_removal,
)
from database_topic_modeling.utils.utils import (
    batch_encode_texts,
    clustering,
    create_tokens_per_class_label_and_vocab,
    dim_reduction_3d,
    get_cols_and_rows,
    plot_dendrogram,
    scatter_3d,
    tf_idf_matrix,
)
from database_topic_modeling.utils.weaviate_utils import (
    upload_to_weaviate,
    get_ids_from_prev_cluster_and_create_class,
)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def produce_cluster_graphs_and_3d_output(
    client_or_connection: Union[mysql.connector.connection_cext.CMySQLConnection, clickhouse_connect.driver.httpclient.HttpClient],
    tablename: str,
    id_column_name: str,
    text_column_name: str,
    weaviate_url: str,
    weaviate_auth_api_key: str,
    class_name: str,
    custom_data: list[str],
    sentence_transformer_name: str='DMetaSoul/sbert-chinese-general-v2',
    language: str='zh-CN',
    batch_size: int=32,
    type_label: str='tag',
    output_png_file: str="outputs/clustering-output.png",
    output_html_file: str="outputs/clustering-topics-3d.html",
) -> None:
    """
    Creates a clustering output graph given access to a database with English/Chinese abstract-size
    pieces of text to produce topic modeling of, and a weaviate cluster to store the vectors produced 
    by the database. 

    Args:
        client_or_connection - the mysql connection object or the Clickhouse client object from which
                               to extract rows and cols
        tablename - the tablename of the database to query
        id_column_name - the name of the column which will be used as the id of the dataset
        text_column_name - the name of the column which contains the text to encode in the dataset
        weaviate_url - the weaviate url for use as the database
        weaviate_auth_api_key - the weaviate auth api key
        class_name - the name of the class to name the weaviate cluster
        custom_data - list of column names, the custom data from the database columns to include as
                      well in the legend of the 3d output
        sentence_transformer_name - the name of the sentence transformer to use for encoding 
                                    chinese texts
        language - the ISO code of the language to use for the sentence transformer, follows
                   the google translate standard of using zh-CN for simplified mandarin and 
                   en for English
        batch_size - the batch size of the sentence_transformer
        type_label - the type of label to give the items in the cluster
        output_png_file - the name of the output png file for the clustering dendrogram
        output_html_file - the name of the output html file for the 3d projection
    """

    cols, rows = get_cols_and_rows(client_or_connection, tablename)

    client = weaviate.Client(
        url=weaviate_url,
        auth_client_secret=weaviate.AuthApiKey(api_key=weaviate_auth_api_key),
    )

    ids = get_ids_from_prev_cluster_and_create_class(client, class_name, sentence_transformer_name)

    dataset = create_huggingface_dataset(rows, cols, ids, id_column_name=id_column_name, text_column_name=text_column_name)

    model = SentenceTransformer(sentence_transformer_name)
    model.to(device)

    UPLOAD_BS = 10_000
    for i in range(0, len(dataset), UPLOAD_BS):
        embeds = batch_encode_texts(model, dataset[i:min(i+UPLOAD_BS, len(dataset))], batch_size, device)

        upload_to_weaviate(client, class_name, embeds, dataset[i:min(i+UPLOAD_BS, len(dataset))])

    # # Nearest Neighbor search

    # Get all vectors from weaviate and create an embedding matrix and a full dataset from it
    embeds, dataset = create_huggingface_dataset_and_embeds(client, class_name, id_column_name=id_column_name)

    # TSNE to decrease dimensionality of the datapoints
    u = dim_reduction_3d('TSNE', embeds, n_components=3, perplexity=30.0, metric="cosine", random_state=42)

    n_clusters=6

    # agglomerative heirarchical clustering of the datapoints
    clusterer = clustering('agglomerative', u, n_clusters=n_clusters, metric='euclidean', linkage='ward')

    # add the cluster labels to our dataset
    dataset = dataset.add_column('class', clusterer.labels_)

    if type_label == 'tag':
        dataset = tokenization_and_stopword_removal(dataset, language=language)

        classes, vocab = create_tokens_per_class_label_and_vocab(dataset)

        tf_idf = tf_idf_matrix(classes, vocab)

        labels = retrieve_top_n_words_per_class(tf_idf, classes.keys(), list(vocab), 6, language=language)

        additional_dendro_info = None
    elif type_label == 'sentence':
        dict_titles = get_top_n_titles_per_class(u, dataset, language=language, title_column_name="title")

        labels = get_sentence_summaries(dict_titles)

        additional_dendro_info = {v1: v2 for v1, v2 in zip(labels.values(), dict_titles.values())}

    list_labels = [labels[x] for x in clusterer.labels_]
    
    plot_dendrogram(list_labels, u, n_clusters, type_label, additional_dendro_info, output_png_file=output_png_file, metric='euclidean', linkage='ward')

    scatter_3d(u, list_labels, dataset, custom_data, output_html_file)


if __name__ == '__main__':
    # # Example usage
    # conn = mysql.connector.connect(
    #         host="35.245.30.192",
    #         user="root",
    #         password="xTG1ghg7ZlQvRzr66iYG",
    #         database="test"
    #     )

    # produce_cluster_graphs_and_3d_output(
    #     conn,
    #     tablename="article_cn",
    #     id_column_name="id",
    #     text_column_name="text",
    #     weaviate_url="https://database-cluster-e8lalwdl.weaviate.network",
    #     weaviate_auth_api_key="BoyZ0LFzJvR38BW8ZJLP5A40bHfs2e51j7bt",
    #     class_name="Database",
    #     custom_data=["title", "source", "source_date"]
    # )

    from dotenv import load_dotenv
    dotenv_path = os.path.join("credentials", ".env")
    load_dotenv(dotenv_path)

    CLICKHOUSE_USER = os.getenv('CLICKHOUSE_USER')
    CLICKHOUSE_PASS = os.getenv('CLICKHOUSE_PASS')

    client = clickhouse_connect.get_client(
        host='192.168.20.101', 
        port='8123', 
        username=CLICKHOUSE_USER, 
        password=CLICKHOUSE_PASS,
        database='openalex_cn',
    )

    produce_cluster_graphs_and_3d_output(
        client,
        tablename="works",
        id_column_name="id",
        text_column_name="abstract_inverted_index",
        weaviate_url="https://g5ps2xtsquochnvbump5a.c0.us-east4.gcp.weaviate.cloud",
        weaviate_auth_api_key="9rnxcpvZJV9sRFOzXQKvgtan8ze7uofDuBvi",
        class_name="Database",
        custom_data=["title", "doi", "publication_date"],
        sentence_transformer_name='all-mpnet-base-v2',
        language='en',
        batch_size=32,
        output_png_file="clustering-output-openalex.png",
        output_html_file="clustering-topics-3d-openalex.html",
    )