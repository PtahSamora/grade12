import os
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient

# Load environment variables
load_dotenv()

# Retrieve the connection string and container name from environment variables
connect_str = os.getenv("connect_str")
container_name = os.getenv("container_name")

# Initialize a BlobServiceClient
blob_service_client = BlobServiceClient.from_connection_string(connect_str)

# Get a container client
container_client = blob_service_client.get_container_client(container_name)
# Uncomment the next line if you need to create the container during the first run
# container_client.create_container()

def upload_pdf(file_path):
    # Extract the filename from the full file path
    blob_name = os.path.basename(file_path)
    # Create a blob client for this specific file
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

    # Open the file in binary mode and upload it to the blob storage
    with open(file_path, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)  # Overwrite if the blob already exists

# Specify the folder path where the PDF files are located
folder_path = '/Users/samorasixaba/Documents/Matric Preparation Project/downloaded_papers'
# Iterate over all files in the folder and upload those ending with '.pdf'
for file in os.listdir(folder_path):
    if file.endswith('.pdf'):
        file_path = os.path.join(folder_path, file)
        upload_pdf(file_path)