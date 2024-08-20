import requests
from bs4 import BeautifulSoup
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import urllib3
import torch

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

DATA_URL = 'https://studentaffairs.unt.edu/events/index.html'
DB_FAISS_PATH = 'vectorstore/db_faiss'
SCRAPED_DATA_FILE = 'scraped_data.txt'
SCRAPED_URLS_FILE = 'scraped_urls.txt'
TIME_LOG_FILE = 'time_log.txt'
MAX_WEBSITES = 500

scraped_urls = set()

def is_valid_response(response):
    """
    Check if the response status code is between 200 and 205.
    """
    return 200 <= response.status_code <= 205

def fetch_page_content(url):
    """
    Fetch the content of a webpage with SSL verification disabled.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, verify=False) 
        if is_valid_response(response):
            response.encoding = response.apparent_encoding 
            html_content = response.text
            if "<html" not in html_content.lower():
                print(f"Warning: The content fetched from {url} does not appear to be valid HTML.")
                return None
            return html_content
        else:
            print(f"Invalid response for {url}: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Failed to scrape {url}: {e}")
        return None

def parse_content(html_content):
    """
    Parse the HTML content to extract a single block of text and image URLs.
    """
    soup = BeautifulSoup(html_content, 'lxml')  # Use 'lxml' parser
    paragraphs = soup.find_all('p')
    text_data = " ".join(para.get_text() for para in paragraphs).strip()
    images = soup.find_all('img', src=True)
    image_data = {f"image_{i}": img['src'] for i, img in enumerate(images)}
    return text_data, image_data

def extract_links(soup):
    """
    Extract all links from the soup object.
    """
    links = soup.find_all('a', href=True)
    return [link['href'] for link in links if link['href'].startswith('http')]

def find_next_page(soup):
    """
    Find the link to the next page.
    """
    next_page = soup.find('a', text='Next')  # Adjust this selector based on actual HTML
    if next_page and 'href' in next_page.attrs:
        return next_page['href']
    return None

def scrape_single_url(url, depth, max_depth):
    """
    Scrape a single URL for text and images, and recursively scrape linked pages.
    """
    global scraped_urls
    html_content = fetch_page_content(url)
    if html_content is None:
        return {}

    soup = BeautifulSoup(html_content, 'lxml')  # Use 'lxml' parser
    text_data, image_data = parse_content(html_content)
    urls = extract_links(soup)
    
    scraped_urls.add(url)
    
    data = {url: {'text': text_data, 'images': image_data}}
    
    if depth < max_depth:
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(scrape_single_url, link, depth + 1, max_depth): link for link in urls if len(scraped_urls) < MAX_WEBSITES}
            for future in as_completed(futures):
                sub_data = future.result()
                data.update(sub_data)
    
    # Check for next page
    next_page_url = find_next_page(soup)
    if next_page_url and len(scraped_urls) < MAX_WEBSITES:
        data.update(scrape_single_url(next_page_url, depth, max_depth))
    
    return data

def scrape_website(url, depth=1, max_depth=5):
    """
    Initiate the scraping process for the given URL.
    """
    return scrape_single_url(url, depth, max_depth)

def save_scraped_data(scraped_data):
    """
    Save the scraped text and image data to a file.
    """
    with open(SCRAPED_DATA_FILE, 'w', encoding='utf-8') as file:
        for url, contents in scraped_data.items():
            file.write(f"URL: {url}\n")
            file.write(f"text_0: {contents['text']}\n")
            for image_id, image_url in contents['images'].items():
                file.write(f"{image_id}: {image_url}\n")
            file.write('\n')

def save_scraped_urls():
    """
    Save the scraped URLs to a file.
    """
    with open(SCRAPED_URLS_FILE, 'w', encoding='utf-8') as file:
        for url in scraped_urls:
            file.write(url + '\n')

def create_documents(scraped_data):
    """
    Create Document objects from the scraped text data.
    """
    return [Document(page_content=contents['text'], metadata={"source": url}) 
            for url, contents in scraped_data.items()]

def build_vector_store(documents):
    """
    Build a FAISS vector store from the documents.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': device}
    )
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)

def log_time(start_time, end_time):
    """
    Log the start and end times of the scraping process.
    """
    response_time = end_time - start_time
    with open(TIME_LOG_FILE, 'w', encoding='utf-8') as file:
        file.write(f"Start time: {time.ctime(start_time)}\n")
        file.write(f"End time: {time.ctime(end_time)}\n")
        file.write(f"Response time: {response_time:.6f} seconds\n")
    print(f"Response time: {response_time:.6f} seconds")

def create_vector_db():
    """
    Main function to create the vector database.
    """
    start_time = time.time()
    
    scraped_data = scrape_website(DATA_URL)
    save_scraped_data(scraped_data)
    save_scraped_urls()
    
    documents = create_documents(scraped_data)
    build_vector_store(documents)
    
    end_time = time.time()
    log_time(start_time, end_time)

if __name__ == "__main__":
    create_vector_db()