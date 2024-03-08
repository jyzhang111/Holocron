import ast
import jieba
import nltk
import numpy as np
import re
import string
from datasets import Dataset
from collections import defaultdict
from google.cloud import translate_v2 as translate
from textblob import Word

from database_topic_modeling.utils.utils import (
    create_tokens_per_class_label_and_vocab,
    tf_idf_matrix,
)

import os
# PARENT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=os.path.join(PARENT_PATH, "credentials", "application_default_credentials.json")

def get_top_n_titles_per_class(
    u: np.ndarray,
    dataset: Dataset,
    language: str='zh-CN',
    title_column_name: str="title",
    n: int=10,
) -> dict:
    """
    Returns the top n titles per class in the form of a dictionary

    Args:
        u - np.ndarray of the 3d vectors that represent the embeddings 
        dataset - huggingface dataset that contains the oldids and other information of the embeddings
                  to return
        language - the ISO code of the language to use for the sentence transformer, follows the google
                   translate standard
        title_column_name - the column name of the column corresponding to the titles
        n - an int for the number of titles to return

    Returns:
        dict - a dictionary with classes mapped to a list of the top n titles per class
    """
    embed_dict = defaultdict(lambda: [np.zeros((u.shape[1],)), 0])

    for i, d in enumerate(dataset):
        embed_dict[d["class"]][0] += u[i]
        embed_dict[d["class"]][1] += 1

    embed_dict = {k: v[0]/v[1] for k, v in embed_dict.items()}

    return_dict = {}

    for k, v in embed_dict.items():
        distances = np.linalg.norm(u-v, axis=1)

        nearest_values = np.argsort(distances)[:n]
        titles = [dataset[int(i)][title_column_name] for i in nearest_values]

        if language == 'zh-CN':
            translated_text = translate_client.translate(titles, source_language='zh-CN', target_language='en')
            translated_text = [t['translatedText'] for t in translated_text]
        elif language == 'en':
            translated_text = titles

        return_dict[k] = titles

    return return_dict

def language_representation(
    search_str: str, 
    language: str='zh-CN',
) -> str:
    """
    Returns the representation of the search string in the language specified

    Args:
        search_str - the search string to search
        language - the ISO code of the language to use for the sentence transformer, follows the google
                   translate standard

    Returns:
        str - a string of the translated text
    """

    translate_client = translate.Client()

    if language != "en":
        search_str = translate_client.translate(search_str, source_language='en', target_language=language)['translatedText']

    return search_str

def retrieve_top_n_words_per_class(
    tf_idf: np.ndarray, 
    list_classes: list,
    list_vocab: set,
    n: int,
    language: str='zh-CN', 
) -> dict:
    """
    Retrieves the top n terms per class

    Args:
        tf_idf - a tf_idf matrix of size len(classes), len(vocab)
        list_classes - a list of classes corresponding to the rows of the tf_idf matrix
        list_vocab - a list of vocabulary terms corresponding to the columns of the tf_idf matrix
        n - an integer representing the top n terms to return per class
        language - the ISO code of the language to perform tokenization in, follows the google
                   translate standard

    Returns:
        dict - a dictionary of the top n words per class
    """

    top_idx = np.argpartition(tf_idf, -n)[:, -n:]

    translate_client = translate.Client()

    # graph labels
    labels = {}

    for c, _class in enumerate(list_classes):
        topn_idx = top_idx[c, :]
        # map the index positions back to the original words in the vocab
        topn_terms = [list_vocab[idx] for idx in topn_idx]

        if language == 'zh-CN':
            translated_text = translate_client.translate(topn_terms, source_language='zh-CN', target_language='en')
            translated_text = [t['translatedText'] for t in translated_text]
        elif language == 'en':
            translated_text = topn_terms

        # print(f"Class {_class} top {n} terms: {translated_text}")
        labels[_class] = str(translated_text)

    return labels

def tokenization_and_stopword_removal(
    dataset: Dataset, 
    language: str='zh-CN', 
    column_to_tokenize: str='sentencetext',
    **kwargs
) -> Dataset:
    """
    Tokenize and remove stopwords from the dataset from column_to_tokenize, and place them in a
    column named 'tokens'

    Args:
        dataset - Huggingface Dataset on which to perform tokenization on 'column_to_tokenize'
        language - the ISO code of the language to perform tokenization in, follows the google
                   translate standard
        column_to_tokenize - column to tokenize

    Returns:
        Dataset - the tokenized dataset
    """
    if language == "zh-CN":
        # perform tokenization and stopword removal of the Chinese text
        punc = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏. \u3000"

        dataset = dataset.map(lambda x: {'tokens': re.sub(r"[%s]+" %punc, '', x[column_to_tokenize])})

        # tokenize using the jieba library
        dataset = dataset.map(lambda x: {'tokens': jieba.lcut(x['tokens'])})
        # remove stopwords
        extra_stopwords = ['中', '月', '年', '上', 'LICS', '\xa0']
        stopwords = set(nltk.corpus.stopwords.words('chinese') + extra_stopwords)
        dataset = dataset.map(lambda x: {'tokens': [word for word in x['tokens'] if (word not in stopwords) and len(word)<10]})
    elif language == "en":
        punc = string.punctuation

        dataset = dataset.map(lambda x: {'tokens': re.sub(r"[%s]+" %punc, '', x[column_to_tokenize])})

        # tokenize, separating by whitespace
        dataset = dataset.map(lambda x: {'tokens': nltk.tokenize.wordpunct_tokenize(x['tokens'])})
        # remove stopwords
        stopwords = set(nltk.corpus.stopwords.words('english'))
        # Uncapitalize elements if only first letter is capital
        dataset = dataset.map(lambda x: {'tokens': [word.lower() if word.istitle() else word for word in x['tokens']]})
        # stopwords from nltk are all lowercase (so are our tokens)
        dataset = dataset.map(lambda x: {'tokens': [word for word in x['tokens'] if word not in stopwords]})

    return dataset

def word_cloud(dataset: Dataset, google_app_credentials_path: str, language: str='zh-CN') -> dict:
    """
    Returns a dictionary of a word cloud of the data in abstracts

    Args:
        dataset - huggingface dataset that contains the abstracts for creating the word cloud 
        google_app_credentials_path - the path of the google application credentials
        language - the ISO code of the language to perform tokenization in, follows the google
                   translate standard

    Returns:
        dict - word cloud dictionary
    """
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=google_app_credentials_path
    
    # Get word frequency per class (TF)
    classes, vocab = create_tokens_per_class_label_and_vocab(dataset)

    term_index_dict, tf, tf_idf = tf_idf_matrix(classes, vocab)

    labels = retrieve_top_n_words_per_class(tf_idf, classes.keys(), list(vocab), 6, language=language)

    word_cloud_json_1 = {}

    for v in labels.values():
        list_words = ast.literal_eval(v)
        for word in list_words:
            word_cloud_json_1[word] = round(tf[:, term_index_dict[word]].sum())

    word_cloud_json_2 = {}

    for k, v in sorted(word_cloud_json_1.items(), key=lambda x:x[1], reverse=True):
        singular = Word(k).singularize()
        if singular not in '’”“':
            if singular in word_cloud_json_2:
                word_cloud_json_2[singular] += v
            else:
                word_cloud_json_2[singular] = v

    return word_cloud_json_2