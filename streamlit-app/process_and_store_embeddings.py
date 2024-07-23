from azure.storage.blob import BlobClient, BlobServiceClient
import pickle
import streamlit as st 
import os
import io 
from PyPDF2 import PdfReader
import pymupdf
import fitz
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
import openai
from openai import OpenAI
from transformers import pipeline

load_dotenv()

client = OpenAI()

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
def process_text(text):
    # Clean and prepare text
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

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

container_name = os.getenv("container_name")
    # Get the list of PDF files
# pdf_files = list_pdf_files(container_name)

successful_embeddings = set()

def process_and_store_embeddings(container_name, blob_service_client, successful_embeddings_set):
    container_client = blob_service_client.get_container_client(container_name)
    blobs = container_client.list_blobs()

    for blob in blobs:
        if blob.name.endswith('.pdf'):
            print(f"Processing PDF: {blob.name}")
            try:
                pdf_stream = download_pdf(container_name, blob.name)
                text = extract_text_from_pdf(pdf_stream)
                text = process_text(text)
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
                chunks = text_splitter.split_text(text)
                embeddings, ids = create_embeddings(chunks)  # Get embeddings and their IDs
                
                # Generate metadata for each chunk
                metadata = {id: {'text': chunk} for id, chunk in zip(ids, chunks)}

                # Prepare embeddings and metadata for storage
                data_to_store = {'embeddings': embeddings, 'ids': ids}
                metadata_data = pickle.dumps(metadata)

                # Serialize the embeddings along with their IDs
                embeddings_data = pickle.dumps(data_to_store)

                # Store embeddings
                embeddings_blob_name = f"{blob.name.split('.pdf')[0]}_embeddings.pkl"
                embeddings_blob_client = container_client.get_blob_client(embeddings_blob_name)
                embeddings_blob_client.upload_blob(embeddings_data, overwrite=True)

                # Store metadata
                metadata_blob_name = f"{blob.name.split('.pdf')[0]}_metadata.pkl"
                metadata_blob_client = container_client.get_blob_client(metadata_blob_name)
                metadata_blob_client.upload_blob(metadata_data, overwrite=True)

                # Mark successful processing
                successful_embeddings_set.add(blob.name)
                print(f"Successfully uploaded embeddings and metadata for {blob.name}")
            except Exception as e:
                print(f"Error processing {blob.name}: {str(e)}")

# def create_embeddings(chunks):
#     """Create embeddings and generate unique identifiers for each chunk."""
#     model = SentenceTransformer("infgrad/stella_en_1.5B_v5", trust_remote_code=True).cuda()
#     embeddings = model.encode(chunks, convert_to_tensor=False)
#     ids = range(len(chunks))  # Simple sequential IDs; customize as needed
#     return embeddings, list(ids)

def create_embeddings(chunks):
    """Create embeddings using OpenAI's model and generate unique identifiers for each chunk."""
    # Assuming 'client' is your authenticated OpenAI client
    embeddings = []
    for chunk in chunks:
        response = client.embeddings.create(input=chunk,
    model="text-embedding-3-small"  # Update with the appropriate engine name
    )
        # response = openai.Embedding.create(
        #     input=chunk,
        #     model="text-embedding-ada-002"  # Choose the appropriate model
        # )
        embedding = response.data[0].embedding
        embeddings.append(embedding)
    ids = range(len(chunks))  # Simple sequential IDs; customize as needed
    return embeddings, list(ids)

process_and_store_embeddings(container_name, blob_service_client, successful_embeddings)
