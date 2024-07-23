# from sentence_transformers import SentenceTransformer
# import os
# from PyPDF2 import PdfReader
# from PyPDF2.errors import PdfReadError
# import re
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from chat_with_pdf import Initialize_sentence_transformer
# import faiss
# import numpy as np
# import pickle


# api_key = os.getenv('OPENAI_API_KEY')
# # Function to read table names from the file
# def list_pdf_files(directory):
#     """ Returns a list of filenames ending in '.pdf' in the specified directory """
#     return [file for file in os.listdir(directory) if file.endswith('.pdf')]


# folder_path = '/Users/samorasixaba/Documents/Matric Preparation Project/downloaded_papers'
# def create_and_store_embeddings(chunks, pdf_path, index, metadata_dict):
#         model_name = "all-MiniLM-L6-v2"
#         model = SentenceTransformer(model_name)
#         print("Model loaded successfully!")
#         # Generate embeddings and store in index
#         embeddings = model.encode(chunks, convert_to_tensor=False)
#         ids = np.arange(len(chunks)) + index.ntotal

#         # Add embeddings to the index
#         index.add_with_ids(embeddings, ids)
#         # Store metadata
#         for i, para_id in enumerate(ids):
#             metadata_dict[para_id] = {
#                 'text': chunks[i],
#                 'file': os.path.basename(pdf_path)
#             }
# # Get the list of PDF files
# pdf_files = list_pdf_files(folder_path)

# for file in pdf_files:  
#     # selected_file = st.selectbox('Select a PDF file to view:', pdf_files)

#     file_path = os.path.join(folder_path, file)
#     # with st.expander("## 🔎 Preview Paper"):
#             # display_pdf(file_path)
#     # if st.button("Load Exam Paper"):
#     # @st.cache_data
#     try:
#         def extract_text_from_pdf(file_path):
#             pdf_reader = PdfReader(file_path)
#             text = []
#             for page in pdf_reader.pages:
#                 text.append(page.extract_text())
#             return " ".join(text)
#         # LinkClick.aspx?fileticket=FZpiQZ3046g%3d&tabid=610&portalid=0&mid=1918.pdf
#         # @st.cache_data
#         def process_text(text):
#             # Clean and prepare text
#             text = text.replace('\n', ' ')
#             text = re.sub(r'\s+', ' ', text)
#             text = text.strip()
#             return text
#         pdf_reader = PdfReader(file_path)

#         text = extract_text_from_pdf(file_path)

#         text = process_text(text)
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000, 
#             chunk_overlap=200,
#             length_function=len
#         )
#         chunks = text_splitter.split_text(text)

#         # chunks
#         if chunks:
#             faiss_index = faiss.IndexIDMap(faiss.IndexFlatL2(384))  # Adjust size based on embedding dimension
#             metadata = {}
#             # create_and_store_embeddings(chunks, file, faiss_index, metadata)
#             create_and_store_embeddings(chunks, folder_path, faiss_index, metadata)
#             faiss.write_index(faiss_index, 'faiss_index.index')

#             with open('metadata.pkl', 'wb') as f:
#                 pickle.dump(metadata, f)
#     except PdfReadError as e:
#         print(f"Failed to read PDF due to a read error: {e}")
#     except Exception as e:
#         # Handle other exceptions that may occur
#         print(f"An unexpected error occurred: {e}")
   
# # Setup Faiss index and metadata storage


# # model = SentenceTransformer('all-MiniLM-L6-v2')
# # print("Model loaded successfully!")

import os
import re
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader, errors as pdf_errors
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter

def list_pdf_files(directory):
    return [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.pdf')]

def extract_text_from_pdf(file_path):
    try:
        pdf_reader = PdfReader(file_path)
        return " ".join([page.extract_text() or "" for page in pdf_reader.pages])
    except pdf_errors.PdfReadError as e:
        print(f"Failed to read PDF due to a read error: {e}")
        return None

def process_text(text):
    text = text.replace('\n', ' ').strip()
    return re.sub(r'\s+', ' ', text)

def create_and_store_embeddings(chunks, pdf_path, index, metadata_dict):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks, convert_to_tensor=False)
    ids = np.arange(len(chunks)) + index.ntotal
    index.add_with_ids(embeddings, ids.astype(np.int64))  # Ensure ids are the correct type
    for i, para_id in enumerate(ids):
        metadata_dict[para_id] = {'text': chunks[i], 'file': os.path.basename(pdf_path)}

# Main execution
folder_path = '/Users/samorasixaba/Documents/Matric Preparation Project/downloaded_papers'
pdf_files = list_pdf_files(folder_path)
faiss_index = faiss.IndexIDMap(faiss.IndexFlatL2(384))
metadata = {}

for file_path in pdf_files:
    text = extract_text_from_pdf(file_path)
    if text:
        text = process_text(text)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
        chunks = text_splitter.split_text(text)
        if chunks:
            create_and_store_embeddings(chunks, file_path, faiss_index, metadata)

# Save index and metadata after processing all files
faiss.write_index(faiss_index, 'faiss_index.index')
with open('metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)
