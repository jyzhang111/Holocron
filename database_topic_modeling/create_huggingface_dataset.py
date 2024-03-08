from datasets import Dataset
import weaviate
import numpy as np

def create_huggingface_dataset(
	rows: list[tuple],
	cols: list[str],
    ids: set[str],
	id_column_name: str="id",
	text_column_name: str="sentencetext"
) -> Dataset:
    """
    Creates a huggingface dataset from a list of rows returned by the cursor.fetchall() function,
    as well as a list of the column names of the database. The retuned dataset has the
    id_column_name changed to "id" and the text_column_name changed to "sentencetext". Excludes
    the ids listed in ids.

    Args:
        rows - the rows of the database
        cols - the columns of the database as a list of strings
        ids - a set of the ids to exclude from the dataset
        id_column_name - the name of the column which will be used as the id of the dataset
        text_column_name - the name of the column which contains the text to encode in the dataset

    Returns:
        Dataset - the Huggingface dataset representing the original data in the database
    """
    dict_ = {}
    if "oldid" in cols and id_column_name != "oldid":
        raise ValueError("A column named id already exists in the database. For input into weaviate "
            "our function assumes the id column will be named oldid. Please consider renaming the column "
            "named oldid, or convert that column to the column referenced by id_column_name.")
    if "sentencetext" in cols and text_column_name != "sentencetext":
        raise ValueError("A column named sentencetext already exists in the database. For input into "
            "our transformer model our function assumes the text column will be named sentencetext. "
            "Please consider renaming the column named sentencetext, or convert that column to the column "
            "referenced by text_column_name.")

    oldid_index = None

    for j, name in enumerate(cols):
        if name == id_column_name:
            cols[j] = "oldid"
            oldid_index = j
        elif name == text_column_name:
            cols[j] = "sentencetext"
        dict_[cols[j]] = []

    for row in rows:
        if row[oldid_index] in ids:
            continue
        for i, col in enumerate(cols):
            dict_[col].append(row[i])

    ds = Dataset.from_dict(dict_)

    return ds


def create_huggingface_dataset_and_embeds(
    client: weaviate.client.Client, 
    class_name: str,
    id_column_name: str="id",
) -> tuple[np.ndarray, Dataset]:
    """
    Creates an embedding matrix and a huggingface dataset from a class in a weaviate client.
    The retuned embedding matrix and dataset contain all the data from the class in the weaviate
    client, and has the id_column_name changed changed back from "oldid" to the correct name.

    Args:
        client - the weaviate client to use to query the objects
        class_name - the class_name (database name) of the vectors in the weaviate client
        id_column_name - the original id column name of the column changed to "oldid" in weaviate

    Returns:
        np.ndarray - an embedding matrix containing all the vectors from the weaviate class
        Dataset - the Huggingface dataset representing the extra properties in the weaviate
                  vector database
    """
    dict_ = {}
    data = client.data_object.get(with_vector=True, limit=100000, class_name=class_name)['objects']

    vector_dimension = 50
    if len(data) > 0:
        vector_dimension = len(data[0]['vector'])
        for col_name in data[0]['properties'].keys():
            if col_name == 'oldid':
                col_name = id_column_name
            dict_[col_name] = []

    embeds = np.zeros((len(data), vector_dimension))

    for i, d in enumerate(data):
        for k, v in data[i]['properties'].items():
            if k == 'oldid':
                k = id_column_name
            dict_[k].append(v)
        embeds[i] = data[i]['vector']

    ds = Dataset.from_dict(dict_)

    return embeds, ds

