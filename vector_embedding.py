# vector_embedding.py
import os
from langchain_pinecone import PineconeVectorStore
from langchain_cohere import CohereEmbeddings



def embedd(analyzed_results):
    os.environ['PINECONE_API_KEY'] = '63cc481a-df02-4a58-8dea-cc4905005f70'  
    os.environ["COHERE_API_KEY"] = "V5jqL0gNm1xo9gIzN0x32c9Nc814b4gh9gpOY6EY"
    
    index_name = "asapp-temp1"  # Choose a suitable name for your index

    # Initialize Cohere embeddings
    '''embeddings = CohereEmbeddings(
        model="embed-english-v3.0", 
        user_agent="Cohere-Python-Client"
        # input_type="text"  # Specify that we are embedding text
    )'''
    embeddings = CohereEmbeddings(
        model="embed-english-v3.0"
    )

    # Prepare texts and metadata for storage
    texts = analyzed_results

    print(type(texts))
    # Store analyzed results in the vector database
    vector_store = PineconeVectorStore.from_texts(
        texts=texts,
        embedding=embeddings,
        index_name=index_name
    )
