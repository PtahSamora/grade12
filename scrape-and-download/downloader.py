import os
import requests
# from scraper import get_links_to_math_pdfs
import subprocess
import json

# Read the links from the file
with open('math_links.json', 'r') as file:
    math_papers_links = json.load(file)

# Directory where we want to save the PDFs
save_dir = '/Users/samorasixaba/Documents/Matric Preparation Project/downloaded_papers'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for link in math_papers_links:
    # Make a GET request to download the PDF
    response = requests.get(link, stream=True)

    if response.status_code == 200:
        # Extract the file name from the link
        file_name = link.split('/')[-1]
        file_path = os.path.join(save_dir, file_name)

        # Write the content to a file
        with open(file_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded: {file_name}")
    else:
        print(f"Failed to download the file: {link}")
