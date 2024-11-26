import requests
from bs4 import BeautifulSoup

url = "https://webscraper.io/test-sites"  # Replace with the URL you want to scrape
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    page_content = response.text
else:
    print(f"Failed to retrieve the page. Status code: {response.status_code}")

soup = BeautifulSoup(page_content, 'html.parser')

links = soup.find_all('a')
for link in links:
    href = link.get('href')
    print(href)
