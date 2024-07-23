import streamlit as st
import fitz  # PyMuPDF
import os
import re
import pickle
import faiss
from PyPDF2 import PdfReader
from pdfminer.high_level import extract_text
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure the OpenAI client with your API key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Set up Streamlit page configuration
st.set_page_config(layout="wide")

# Sidebar configuration and content
def display_sidebar():
    with st.sidebar:
        st.title('AI Matric Study Mate :book:')
        st.markdown('''
        ## Usage: 
        - Select Any Past Paper to Practice
        - Ask AI to clarify any questions you are having difficulty understanding
        - Write your solution on piece of paper
        - Upload your solution and get it checked by AI.
        ''')
        st.write('Made with :heart: by Vuyile Sixaba')

def list_pdf_files(directory):
    """ Returns a list of filenames ending in '.pdf' in the specified directory """
    return [file for file in os.listdir(directory) if file.endswith('.pdf')]

def display_pdf(path):
    """ Opens a PDF file and displays each page as an image in Streamlit """
    try:
        doc = fitz.open(path)
        for page in doc:
            pix = page.get_pixmap()
            img = pix.tobytes("png")
            st.image(img, caption=f"Page {page.number + 1}")
    except Exception as e:
        st.error(f"An error occurred: {e}")

def get_paragraph_embeddings(file_name, faiss_index, metadata):
    """ Retrieve embeddings from Faiss index for paragraphs from the selected file """
    ids = [id for id, data in metadata.items() if data['file'] == file_name]
    if not ids:
        st.write("No IDs found for the selected file.")
        return [], []
    valid_ids = [int(id) for id in ids if 0 <= id < faiss_index.ntotal]
    embeddings = [faiss_index.reconstruct(id) for id in valid_ids]
    return embeddings, ids

def main():
    display_sidebar()
    folder_path = '/Users/samorasixaba/Documents/Matric Preparation Project/downloaded_papers'
    pdf_files = list_pdf_files(folder_path)

    if pdf_files:
        selected_file = st.selectbox('Choose a PDF file:', pdf_files)
        file_path = os.path.join(folder_path, selected_file)
        with st.expander("## 🔎 Preview Paper"):
            display_pdf(file_path)

        with open('metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)

        faiss_index = faiss.read_index('faiss_index.index')

        if st.button("Load Exam Paper"):
            text = extract_text(file_path)
            text = re.sub(r'\s+', ' ', text.replace('\n', ' ')).strip()
            embeddings, ids = get_paragraph_embeddings(selected_file, faiss_index, metadata)

            # Handle user questions about the content
            query = st.text_input("Ask questions about the exam paper:")
            if query and embeddings:
                response = openai.Completion.create(
                    model="text-davinci-003",
                    prompt=f"Question: {query}\n\nContext: {text}",
                    max_tokens=150
                )
                st.write(response.choices[0].text.strip())
    else:
        st.write("No PDF files found in the directory.")

if __name__ == '__main__':
    main()
