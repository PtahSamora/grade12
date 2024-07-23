import streamlit as st 
import pymupdf
import fitz
import os
import re
import numpy as np
import matplotlib
matplotlib.use("agg")
import base64
from PIL import Image
import io
import pickle
import faiss
# import dill as pickle
import pytesseract
from PIL import Image
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from pdfminer.high_level import extract_text
from openai import OpenAI
from openai import OpenAI
# from azure.storage.blob import BlobServiceClient, BlobClient
from io import BytesIO

from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain_community.vectorstores.faiss import FAISS
from langchain_community.vectorstores import FAISS

from pandasai import SmartDataframe
# from pandasai.callbacks import BaseCallback
# from pandasai.llm import OpenAI
from pandasai.responses.response_parser import ResponseParser


from image_processing import load_image
from image_processing import preprocess_image
from image_processing import extract_text_from_image
from image_processing import process_text_with_llm

from pdf_text_processor import create_faiss_index
from pdf_text_processor import get_openai_embeddings

from vision_analysis import vision_analysis
# from get_image_url import upload_image_to_blob
from sentence_transformers import SentenceTransformer, util
from matplotlib.backends.backend_agg import RendererAgg
import logging

from chat_with_pdf import Initialize_sentence_transformer
from chat_with_pdf import encode_each_paragraph
from chat_with_pdf import choose_most_relevant_sentence
from chat_with_pdf import query_the_llm

from style_css import create_header
from style_css import html_template
from style_css import sidebar_style


from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions, BlobClient
from datetime import datetime, timedelta
from azure.storage.blob import BlobServiceClient

from sklearn.metrics.pairwise import cosine_similarity
import contextlib
# from doctr.io import read_pdf
# from doctr.models import ocr_predictor



# from gpt4all import GPT4All
# from langchain.llms.gpt4all import GPT4All
# import inspect
# print(inspect.signature(GPT4All.__init__))
# logging.basicConfig(level=logging.DEBUG)
load_dotenv()
# client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
client = OpenAI()  
# _lock = RendererAgg.lock

st.set_page_config(layout="wide")
api_key = os.getenv('OPENAI_API_KEY')
# model = SentenceTransformer('all-MiniLM-L6-v2')

with st.sidebar:
    st.title('AI Matric Study Mate :book:')
    st.markdown('''
    ## Usage: 
    This app is an LLM-powered designed to allow you to study for your matric exams.
    - Select Any Past Paper to Practice
    - Ask AI to clarify any questions you are having difficulty understanding
    - Write your solution on piece of paper
    - Upload your solution and get it checked by AI.
    ''' )
    add_vertical_space(20)
    st.write('Made with :heart: by Vuyile Sixaba')

# Function to read table names from the file
# def list_pdf_files(directory):
#     """ Returns a list of filenames ending in '.pdf' in the specified directory """
#     return [file for file in os.listdir(directory) if file.endswith('.pdf')]

# def display_pdf(path):
#     """ Opens a PDF file and displays each page as an image in Streamlit """
#     try:
#         doc = fitz.open(path)
#         for page in doc:
#             pix = page.get_pixmap()
#             img = pix.tobytes("png")  # Convert to PNG image bytes
#             st.image(img, caption=f"Page {page.number + 1}")
#     except Exception as e:
#         st.error(f"An error occurred: {e}")
def initialize_blob_service():
    # Load connection string from environment or config
    connect_str = os.getenv("connect_str")
    if not connect_str:
        raise ValueError("Azure Storage connection string not found.")

    # Create the BlobServiceClient object
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    return blob_service_client

# Initialize it at the start of your main program or before you call any functions using it
blob_service_client = initialize_blob_service()

def load_embeddings_for_pdf(container_name, pdf_file_name, blob_service_client):
    embeddings_file_name = f"{pdf_file_name.split('.pdf')[0]}_embeddings.pkl"
    container_client = blob_service_client.get_container_client(container_name)
    blob_client = container_client.get_blob_client(embeddings_file_name)

    try:
        if blob_client.exists():
            embeddings_stream = blob_client.download_blob().readall()
            embeddings = pickle.loads(embeddings_stream)
            return embeddings
        else:
            return None
    except Exception as e:
        print(f"Failed to load embeddings: {e}")
        return None


def display_pdf(pdf_stream):
    """Displays each page of a PDF from a byte stream as an image in Streamlit."""
    try:
        # Ensure that pdf_stream is a BytesIO object
        if not isinstance(pdf_stream, io.BytesIO):
            pdf_stream = io.BytesIO(pdf_stream)

        # Open the PDF from the byte stream
        doc = fitz.open("pdf", pdf_stream)
        for page in doc:
            pix = page.get_pixmap()
            img = pix.tobytes("png")  # Convert to PNG image bytes
            # Use BytesIO to convert bytes to a format that Streamlit can display
            img_stream = io.BytesIO(img)
            st.image(img_stream, caption=f"Page {page.number + 1}", width=600)
    except Exception as e:
        st.error(f"An error occurred: {e}")

def list_pdf_files(container_name):
    """Retrieve a list of PDF files from the specified Azure Blob Storage container."""
    container_client = blob_service_client.get_container_client(container_name)
    # Optionally, filter the list to only PDF files
    return [blob.name for blob in container_client.list_blobs() if blob.name.endswith('.pdf')]

def download_pdf(container_name, blob_name):
    """Download a PDF file from Azure Blob Storage and return its content as bytes."""
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    download_stream = blob_client.download_blob()
    return download_stream.readall()

def extract_text_from_pdf(pdf_bytes):
    """
    Extracts text from a PDF byte stream.

    Args:
    pdf_bytes (bytes or io.BytesIO): Raw byte content or BytesIO object of the PDF file.

    Returns:
    str: Extracted text joined into a single string.
    """
    try:
        # Wrap bytes in BytesIO if they're not already a BytesIO object
        if not isinstance(pdf_bytes, io.BytesIO):
            pdf_stream = io.BytesIO(pdf_bytes)
        else:
            pdf_stream = pdf_bytes

        # Reset stream position to the start
        pdf_stream.seek(0)

        # Load the PDF from the byte stream
        pdf_reader = PdfReader(pdf_stream)
        text = []
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:  # Ensure there is text extracted before appending
                text.append(page_text)
        return " ".join(text)
    except Exception as e:
        print(f"Failed to extract text from PDF stream: {str(e)}")
        return ""
    

def create_and_store_embeddings(chunks, blob_name, index, metadata_dict):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks, convert_to_tensor=False)
    ids = np.arange(len(chunks)) + index.ntotal
    index.add_with_ids(embeddings, ids.astype(np.int64))
    for i, para_id in enumerate(ids):
        metadata_dict[para_id] = {'text': chunks[i], 'file': os.path.basename(blob_name)}
# model = SentenceTransformer("infgrad/stella_en_1.5B_v5", trust_remote_code=True).cuda()
# def get_query_embedding(query):
#     # model = SentenceTransformer("infgrad/stella_en_1.5B_v5", trust_remote_code=True).cuda()
#     embeddings = model.encode(query, convert_to_tensor=False)
#     # ids = range(len(chunks))  #
#     return model.encode([query], convert_to_tensor=False)
def get_query_embedding(text):
    response = client.embeddings.create(input=text,
    model="text-embedding-3-small"  # Update with the appropriate engine name
    )
    embeddings = np.array(response.data[0].embedding)
    # embeddings = response.data[0].embedding
    return embeddings
    

# @st.cache(allow_output_mutation=True)  # Caches the function to avoid reloading model
def load_embeddings(blob_name, container_name, blob_service_client):
    """Load pre-stored embeddings and their IDs for a given blob name."""
    embeddings_blob_name = f"{blob_name.split('.pdf')[0]}_embeddings.pkl"
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=embeddings_blob_name)
    
    try:
        download_stream = blob_client.download_blob()
        data = pickle.loads(download_stream.readall())

        # Check and convert embeddings
        embeddings = data['embeddings']
        if isinstance(embeddings, list):
            try:
                # Attempt to convert list of arrays to a single 2D array
                embeddings = np.vstack(embeddings)
            except ValueError as ve:
                # Handle the case where vstack fails due to incompatible shapes
                print("Error stacking embeddings:", ve)
                return None, None
        print("Loaded embeddings shape:", embeddings.shape)  # Confirm shape after conversion

        ids = data['ids']
        return embeddings, ids
    except Exception as e:
        print(f"Failed to load embeddings for {blob_name}: {e}")
        return None, None
    
def find_relevant_chunks(query, embeddings, ids):
    """Find the chunks most relevant to the query within the loaded embeddings."""
    query_embedding = get_query_embedding(query)
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:5]  # Get indices of top 5 similar embeddings
    return [ids[i] for i in top_indices], [similarities[i] for i in top_indices]

def load_metadata(blob_name, container_name, blob_service_client):
    metadata_blob_name = f"{blob_name.split('.pdf')[0]}_metadata.pkl"
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=metadata_blob_name)
    try:
        download_stream = blob_client.download_blob()
        metadata = pickle.loads(download_stream.readall())
        return metadata
    except Exception as e:
        print(f"Failed to load metadata for {blob_name}: {e}")
        return None


def main():
    # Display the header
    create_header()
    
    html_template = """
        <style>
        .right-corner-image {
            position: absolute;
            top: 1;
            right: 1;
            margin: 10px;
        }
        </style>
        <img src="data:image/jpeg;base64,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" class="right-corner-image" width="100">
     """

    sidebar_style = """
        <style>
        /* Targeting the sidebar more specifically */
        [data-testid="stSidebar"] > div:first-child {
            background-color: #3cb371 !important;
        }
        /* Attempting to target all text within the sidebar to change its color to white */
        [data-testid="stSidebar"] .css-1d391kg, [data-testid="stSidebar"] .st-cb, [data-testid="stSidebar"] .st-dd, [data-testid="stSidebar"] {
            color: #ffffff !important;
        }

        /* If the above doesn't cover all text, this broader rule might help */
        [data-testid="stSidebar"] * {
            color: #ffffff !important;
        }
        </style>
    """
    # Initializing session state variables if they don't exist
    if 'embeddings' not in st.session_state:
        st.session_state.embeddings = None
    if 'metadata' not in st.session_state:
        st.session_state.metadata = None

    st.markdown(html_template, unsafe_allow_html=True)

    st.markdown("##")  # Adjust the number of hashes to control the space size
     # Inject the CSS with st.markdown
    st.markdown(sidebar_style, unsafe_allow_html=True)
    st.header("Chat with Your Exam Papers")
    # Initialize FAISS index and metadata
    container_name = os.getenv("container_name")
    # Get the list of PDF files
    pdf_files = list_pdf_files(container_name)

    try:
        faiss_index = faiss.read_index('faiss_index.index')
        with open('metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
    except FileNotFoundError:
        faiss_index = faiss.IndexFlatL2(768)  # Assuming embedding size of 768
        metadata = {}
        st.warning("Metadata or index file not found. Starting with a new index and metadata.")
    except Exception as e:
        st.error(f"Error loading index or metadata: {e}")
        return
    # def load_embeddings(blob_name, container_name, blob_service_client):
    #     """Load pre-stored embeddings and their IDs for a given blob name."""
    #     embeddings_blob_name = f"{blob_name.split('.pdf')[0]}_embeddings.pkl"
    #     blob_client = blob_service_client.get_blob_client(container=container_name, blob=embeddings_blob_name)
        
    #     try:
    #         download_stream = blob_client.download_blob()
    #         data = pickle.loads(download_stream.readall())
    #         embeddings = data['embeddings']
    #         ids = data['ids']
    #         return embeddings, ids
    #     except Exception as e:
    #         print(f"Failed to load embeddings for {blob_name}: {e}")
    #         return None, None
    def process_query(query, embeddings, ids, metadata):
        # Process the query and find relevant information
        query_embedding = get_query_embedding(query)
        # print("Query embedding shape:", query_embedding.shape)
        # print("Stored embeddings shape:", embeddings.shape)
        similarities = cosine_similarity([query_embedding], embeddings)[0]
        top_idx = np.argmax(similarities)  # Index of the highest similarity score
        relevant_id = ids[top_idx]  # Get the relevant ID

        context = metadata.get(relevant_id, {}).get('text', "Relevant text not found.")
        answer = ask_the_llm(query, context)
        return answer

    def ask_the_llm(question, context):
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {'role': 'system', 'content': f"Answer the following question based on the context provided:\n\nContext: {context}\n\nQuestion: {question}. Provide guidance on how to think about the problem and list 5 similar problems for practice."},
                {'role': 'user', 'content': question}
            ],
            temperature=0.1
        )
        return response.choices[0].message.content
    if pdf_files:  
        selected_file = st.selectbox('Choose a PDF file:', options=pdf_files)
        pdf_stream = download_pdf(container_name, selected_file)
        with st.expander("## 🔎 Preview Paper"):
            display_pdf(pdf_stream)

        # st.write(metadata)
        # pdf_file = st.selectbox('Choose a PDF file:', options=list(set(data['file'] for data in metadata.values())))
        if st.button("Load Exam Paper"):
            # Load embeddings and metadata only if not already loaded or if a new file is selected
            if selected_file != st.session_state.get('last_loaded_file', None):
                st.session_state.embeddings, st.session_state.ids = load_embeddings(selected_file, container_name, blob_service_client)
                # st.session_state.embeddings, _ = load_embeddings(selected_file, container_name, blob_service_client)
                st.session_state.metadata = load_metadata(selected_file, container_name, blob_service_client)
                st.session_state.last_loaded_file = selected_file

        query = st.text_input("Ask questions about Your Exam Paper:")
        if st.session_state.get('embeddings') is not None and query:
            if 'ids' in st.session_state and 'metadata' in st.session_state:
                response = process_query(query, st.session_state.embeddings, st.session_state.ids, st.session_state.metadata)
                st.write(response)
            else:
                st.error("Please load the exam paper properly before querying.")
    else:
        st.write("No PDF files found in the directory.")

    
#################################################################################################################
    solution_from_user = st.file_uploader("Please Upload Your Solution", type= ["png", "jpg", "jpeg"])
    if solution_from_user:
        # image = load_image(solution_from_user)
        image = Image.open(solution_from_user)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        # st.write("")
        connect_str = os.getenv('connection_str')
        container_name = "mathapp-container"
        # Usage example
        img_url = os.getenv('img_url')
        # image_url = upload_image_to_blob(image_url_stream, 'mathapp-container', 'test_image.jpg')
        # print("Image URL:", image_url)
        
        # def generate_blob_sas_url(blob_name):
        #     account_name = 'mathappanalysis'
        #     account_key = 'your_account_key'
        #     container_name = 'mathapp-container'

        #     sas_blob = generate_blob_sas(
        #         account_name=account_name,
        #         container_name=container_name,
        #         blob_name=blob_name,
        #         account_key=account_key,
        #         permission=BlobSasPermissions(read=True),
        #         expiry=datetime.utcnow() + timedelta(hours=1)  # Link is valid for 1 hour
        #     )
        #     # img_url = 'https://mathappanalysis.blob.core.windows.net/mathapp-container/test_image.jpg?sp=r&st=2024-05-12T14:38:57Z&se=2024-05-16T22:38:57Z&spr=https&sv=2022-11-02&sr=b&sig=r%2Fcv0KG291rk6b2tml5kUB1PhG9jvRa1JNi%2Brn7IrXY%3D'
        #     url = f"https://{account_name}.blob.core.windows.net/{container_name}/{blob_name}?{sas_blob}"
        #     return url

        # blob_rul = generate_blob_sas_url('test_image.jpg')
        # vision_analyze_image = vision_analysis(blob_rul)
        # st.write(vision_analyze_image)
        processed_image = preprocess_image(image)
        # generate_blob_sas_url("test_image.jpg")
        # st.write(processed_image)
        # texts_ds = ds.map_batches(ExtractText, compute=ActorPoolStrategy(size=2))
        extracted_text = extract_text_from_image(processed_image)
        further_processed_text = process_text_with_llm(extracted_text)
        # extracted_text = process_text_with_llm(processed_image)
        # st.text(extracted_text)
        # st.write(further_processed_text)
        # st.text_area("Edit the text:", value=extracted_text, height=200)
        user_input = st.text_area("Please edit and ensure text reflects your solution:", value=extracted_text, height=150)
        if st.button('Save Changes'):
            # This block runs when the user clicks 'Save Changes'
            st.success("Text edited successfully!")
            # Display the edited text (or you might choose to save it to a file or database)
            st.text(user_input)
            # client = OpenAI()   
            def ask_the_model(question):
                response = client.chat.completions.create(model="gpt-4o-2024-05-13",
                messages = [{'role': 'system', 'content':"Act as a final grade mathematics teacher, grade the correctness of the solution and if solution is correct respond in a congratulatory manner also provide step by step corrections to incorrect solutions"},
                            {'role': 'user', 'content':f"{question}"}],
                temperature=0.1   
                )
                return response.choices[0].message.content
             
            teacher_response = ask_the_model(f"{user_input}")

            st.write(teacher_response)
metadata ={}
if __name__ == '__main__': 
    load_dotenv()
    main()