import os

# Directory where the downloaded files are stored
download_dir = '/Users/samorasixaba/Documents/Matric Preparation Project/downloaded_papers'

for filename in os.listdir(download_dir):
    old_file = os.path.join(download_dir, filename)
    
    # Check if the file already has a .pdf extension to avoid double-appending
    if not filename.lower().endswith('.pdf'):
        new_file = old_file + '.pdf'
        os.rename(old_file, new_file)
        print(f"Renamed '{old_file}' to '{new_file}'")
