import setuptools

setuptools.setup(
    name='database_topic_modeling', 
    version='1.0', 
    packages=['database_topic_modeling'], 
    install_requires=['sentence-transformers', 'mysql-connector-python', 'umap-learn', 
                      'hdbscan', 'matplotlib', 'plotly', 'jieba', 'sacremoses', 
                     'google-cloud-translate==2.0.1', 'python-dotenv', 'pandas', 'datasets',
                     'openai==0.28.0', 'weaviate-client', 'textblob'], 
    # license='', 
    author='John Zhang', 
    author_email='john@holocron.tech', 
    description='This package has the capability of producing topic modeling output as well as '
                'search functionality to obtain similar model embeddings.', 
)