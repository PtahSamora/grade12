
import streamlit as st
import openai
from openai import OpenAI
import os 
from sentence_transformers import SentenceTransformer, util
import time

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

api_key = os.getenv('OPENAI_API_KEY')




@st.cache_data
def Initialize_sentence_transformer():
    model_name = "sentence-transformers/paraphrase-MiniLM-L3-v2"
    embeddings = SentenceTransformer(model_name)
    return embeddings

@st.cache_data
def encode_each_paragraph(paragraphs, embeddings):
    responses = []
    for paragraph in paragraphs:
        response = embeddings.encode([paragraph], convert_to_tensor=True)
        responses.append((paragraph, response))
    return responses


# st.write(query)
@st.cache_data
def choose_most_relevant_sentence(embeddings, responses, query):
    query_embedding = embeddings.encode([query], convert_to_tensor=True)
    best_response = None
    best_similarity = -1.0
    answers = []

    for paragraph, response in responses:
        
        similarity = util.pytorch_cos_sim(query_embedding, response).item()
        
        if similarity >= 0.6:
            
            # count += 1
            
            answers.append(paragraph)
    answer = "\n".join(answers)
    return answer

@st.cache_data
def query_the_llm(answer, query):
    try:
        prompt_message = f"{answer}\n{query}"
        response = client.completions.create(
            model="gpt-3.5-turbo-instruct",
            prompt=prompt_message
            )
        return response.choices[0].text.strip()
    except Exception as e:
        print(f"An error occurred: {e}")
        return None