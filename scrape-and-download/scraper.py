import requests
from bs4 import BeautifulSoup
import json
from urllib.parse import urljoin, unquote, urlparse, parse_qs

# Get the full URL
def get_full_url(relative_url):
    return urljoin("https://www.education.gov.za", relative_url)

# The initial URL containing links to yearly exam papers
start_url = "https://www.education.gov.za/Curriculum/NationalSeniorCertificate(NSC)Examinations/NSCPastExaminationpapers.aspx"

# Start session
session = requests.Session()

# Send a GET request to the start URL
response = session.get(start_url)

# Check for successful response
if response.status_code == 200:
    # Parse the HTML content of the page
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'id': 'dnn_ctr1741_Links_lstLinks'})

    if table:
        anchors = table.find_all('a', href=True)
        # Store the links to each year's examination papers
        year_links = [get_full_url(anchor['href']) for anchor in anchors]
        # year_links = [get_full_url(a['href']) for a in table.find_all('a', href=True) if a.text]
        print("Collected Year Links:", year_links)
    else:
        print("No links found in the table with class 'LinksDesignTable'.")
else:
    print(f"Failed to retrieve the main page, status code: {response.status_code}")

# Placeholder list for Mathematics paper links
math_papers_links = []
is_math_section = False  # Flag to indicate if we are in the 'Mathematics' section

for year_link in year_links:
    response = session.get(year_link)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')

        # Locate the section header for Mathematics
        math_header = soup.find('span', string='Mathematics')
        
        if math_header:
            # Collect all the sibling elements after the header until we reach the next section
            for sibling in math_header.parent.find_next_siblings():
                # Check if the sibling is a header, indicating a new section
                if sibling.name in ['h2', 'h3', 'h4', 'h5', 'h6', 'h7']:
                    # If it's a header, we've reached the end of the Mathematics section
                    break
                else:
                    # Otherwise, collect all the <a> tags within this sibling element
                    links = sibling.find_all('a')
                    for link in links:
                        # Verify if it's a valid link and then append to our list
                        if 'href' in link.attrs and 'paper' in link.text.lower() and 'english' in link.text.lower():
                            full_url = get_full_url(link['href'])
                            math_papers_links.append(full_url)                  
        else:
            print(f"No 'Mathematics' section found on page: {year_link}")
file_path = '/Users/samorasixaba/Documents/Matric Preparation Project/scrape-and-download/math_links.json'
with open(file_path, 'w', encoding='utf-8') as file:
    json.dump(math_papers_links, file, ensure_ascii=False, indent=4)


