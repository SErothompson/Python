import requests
from bs4 import BeautifulSoup
import os
import urllib.parse

# URL of the page to scrape
url = "https://arxiv.org/list/cs.AI/recent"

# Create a directory to save the PDFs
if not os.path.exists('pdfs'):
    os.makedirs('pdfs')

# Function to download PDF files
def download_pdf(pdf_url, save_path):
    response = requests.get(pdf_url)
    with open(save_path, 'wb') as f:
        f.write(response.content)

# Function to get all PDF links recursively
def get_all_pdfs(url, base_url=None):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    pdf_links = []
    for link in soup.find_all('a'):
        href = link.get('href')
        if href and href.startswith('/pdf'):
            full_url = urllib.parse.urljoin(base_url if base_url else url, href)
            pdf_links.append(full_url)
    
    return pdf_links

# Get all PDF links from the page
pdf_links = get_all_pdfs(url, base_url=url)

# Download each PDF
for pdf_link in pdf_links:
    pdf_name = pdf_link.split('/')[-1]
    save_path = os.path.join('pdfs', pdf_name + ".pdf")
    download_pdf(pdf_link, save_path)
    print(f"Downloaded: {pdf_name}")

print("All PDFs have been downloaded.")