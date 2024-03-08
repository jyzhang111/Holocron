Has the capability of producing topic modeling as well as search functionality to obtain similar model embeddings. main.py produces embeddings given a database of abstracts, and produces topic modeling output. search.py implements the ability to search and return related topics for a search query. Due to main.py producing embeddings before search.py, it **must be** run before search.py. To use these functions, you must have a <ins>database with a table and the names of the corresponding id and abstract text columns</ins>, <ins>weaviate cluster credentials, as well as the class_name of the weaviate cluster</ins>, and <ins> a list of column names that you want to include in the 3d output</ins>. As well, if it does not work, you may need new google translate credentials in `application_default_credentials.json`.

To use, ```pip install -e database_topic_modeling```

Example usage surrounding `main.py`, see function comments for more details:

```
import mysql.connector
from database_topic_modeling.main import produce_cluster_graphs_and_3d_output

conn = mysql.connector.connect(
    host="35.245.30.192",
    user="root",
    password="xTG1ghg7ZlQvRzr66iYG",
    database="test"
)

produce_cluster_graphs_and_3d_output(
    conn,
    tablename="article_cn",
    id_column_name="id",
    text_column_name="text",
    weaviate_url="https://database-cluster-e8lalwdl.weaviate.network",
    weaviate_auth_api_key="BoyZ0LFzJvR38BW8ZJLP5A40bHfs2e51j7bt",
    class_name="Database",
    custom_data=["title", "source", "source_date"]
)
```

Example usage surrounding `search.py`, see function comments for more details:

```
from database_topic_modeling.search import search_concept_for_output

inp = input('What concept would you like to search? \n')
search_concept_for_output(
    inp,
    weaviate_url="https://database-cluster-e8lalwdl.weaviate.network",
    weaviate_auth_api_key="BoyZ0LFzJvR38BW8ZJLP5A40bHfs2e51j7bt",
    class_name="Database",
    id_column_name="id",
    custom_data=["title", "source", "source_date"],
)
```