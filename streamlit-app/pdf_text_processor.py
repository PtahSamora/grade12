import numpy as np
import faiss
from openai import OpenAI

# client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
import os
client = OpenAI()


def get_openai_embeddings(sentences):
    embeddings = []
    for sentence in sentences:
        response = client.embeddings.create(input=sentence,
        model="text-embedding-ada-002")
        embeddings.append(response.data.embedding)
    return embeddings

def create_faiss_index(embeddings):
    d = embeddings.shape[1]  # Dimension of embeddings
    index = faiss.IndexFlatL2(d)  # Create a flat L2 index
    index.add(embeddings)  # Add vectors to the index
    return index