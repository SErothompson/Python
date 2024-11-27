import requests
from bs4 import BeautifulSoup

url = "https://www.uoguide.com/Bane"  # Replace with the URL you want to scrape
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}

response = requests.get(url, headers=headers)

if response.status_code == 200:
    page_content = response.text
    soup = BeautifulSoup(page_content, 'html.parser')
    
    links = soup.find_all('a')
    for link in links:
        href = link.get('href')
        print(href)
else:
    print(f"Failed to retrieve the page. Status code: {response.status_code}")
