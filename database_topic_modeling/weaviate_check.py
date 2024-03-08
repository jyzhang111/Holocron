import weaviate
import numpy as np
from sentence_transformers import SentenceTransformer
client = weaviate.Client(
    url="https://nlp-l1n47s1t.weaviate.network",
    auth_client_secret=weaviate.AuthApiKey(api_key="qR81GfOIgNnkfEEgjC08WigEFaQWUB8ZRaxK"),
)

class_obj = {
    "class": "test",
    "vectorizer": "text2vec-huggingface",  # If set to "none" you must always provide vectors yourself. Could be any other "text2vec-*" also.
    "moduleConfig": {
        "text2vec-huggingface": {
          "model": 'DMetaSoul/sbert-chinese-general-v2',
          "options": {
            "waitForModel": True,
            # "useGPU": True,
            "useCache": True,
          }
        }
    }
}

n = 50000

vectors = np.random.rand(n,768)

# client.schema.create_class(class_obj)
# client.batch.configure(batch_size=100)
# with client.batch() as batch:
#     for i in range(n):
#         batch.add_data_object(
#                 data_object={"object": "lol"},
#                 uuid=weaviate.util.generate_uuid5(str(i)),
#                 class_name="test",
#                 vector=vectors[i],
#             )
#         print(i)

import requests
resp = requests.get('https://raw.githubusercontent.com/weaviate-tutorials/quickstart/main/data/jeopardy_tiny.json')
data = json.loads(resp.text)  # Load data

client.batch.configure(batch_size=100)  # Configure batch
with client.batch as batch:  # Initialize a batch process
    for i, d in enumerate(data):  # Batch import data
        print(f"importing question: {i+1}")
        properties = {
            "answer": d["Answer"],
            "question": d["Question"],
            "category": d["Category"],
        }
        batch.add_data_object(
            data_object=properties,
            class_name="Question"
        )