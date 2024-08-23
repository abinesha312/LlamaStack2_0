import urllib.parse
import urllib3
import aiohttp
import asyncio
import torch
import time
import re

from bs4 import BeautifulSoup
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

DATA_URL = 'https://studentaffairs.unt.edu/events/index.html'
DB_FAISS_PATH = 'vectorstore/db_faiss'
SCRAPED_DATA_FILE = 'scraped_data.txt'
SCRAPED_URLS_FILE = 'scraped_urls.txt'
TIME_LOG_FILE = 'time_log.txt'
MAX_WEBSITES = 20000
TIMEOUT = 10  
RETRIES = 3 

scraped_urls = set()

def is_valid_response(status_code):
    """
    Check if the response status code is between 200 and 205.
    """
    return 200 <= status_code <= 205

async def fetch_page_content(session, url, timeout=TIMEOUT, retries=RETRIES):
    """
    Fetch the content of a webpage with SSL verification disabled, with retry logic.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
    }
    attempt = 0
    while attempt < retries:
        try:
            encoded_url = urllib.parse.quote(url, safe=':/')
            print(f"Fetching URL: {encoded_url}") 

            async with session.get(encoded_url, headers=headers, timeout=timeout) as response:
                if is_valid_response(response.status):
                    html_content = await response.text(errors='replace')
                    if "<html" not in html_content.lower():
                        print(f"Warning: The content fetched from {url} does not appear to be valid HTML.")
                        return None
                    return html_content
                else:
                    print(f"Invalid response for {url}: {response.status}")
                    return None
        except asyncio.TimeoutError:
            attempt += 1
            print(f"Timeout while fetching {url}, attempt {attempt} of {retries}")
            if attempt >= retries:
                print(f"Skipping {url} after {retries} attempts due to timeout.")
                return None
            await asyncio.sleep(2)
        except aiohttp.ClientError as e:
            print(f"Failed to scrape {url}: {e}")
            return None

def parse_content(html_content):
    """
    Parse the HTML content to extract a single block of text and image URLs.
    """
    soup = BeautifulSoup(html_content, 'lxml') 
    paragraphs = soup.find_all('p')
    text_data = " ".join(para.get_text() for para in paragraphs).strip()
    phone_pattern = re.compile(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}')
    email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    phones = phone_pattern.findall(text_data)
    emails = email_pattern.findall(text_data)
    for phone in phones:
        text_data = text_data.replace(phone, f" {phone} ")
    for email in emails:
        text_data = text_data.replace(email, f" {email} ")

    
    text_data = " ".join(text_data.split())

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
    next_page = soup.find('a', text='Next')
    if next_page and 'href' in next_page.attrs:
        return next_page['href']
    return None

async def scrape_single_url(session, url, depth, max_depth):
    """
    Scrape a single URL for text and images, and recursively scrape linked pages.
    """
    global scraped_urls
    html_content = await fetch_page_content(session, url)
    if html_content is None:
        return {}
    
    '''
    lxml - parser will parse the data even it is XML or HTML 
    '''

    soup = BeautifulSoup(html_content, 'lxml')  
    text_data, image_data = parse_content(html_content)
    urls = extract_links(soup)
    
    scraped_urls.add(url)
    
    data = {url: {'text': text_data, 'images': image_data}}
    
    if depth < max_depth:
        tasks = [scrape_single_url(session, link, depth + 1, max_depth) for link in urls if len(scraped_urls) < MAX_WEBSITES]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, dict):
                data.update(result)
    
    '''
    Check for the Next page in the scrapped from the current page
    '''
    next_page_url = find_next_page(soup)
    if next_page_url and len(scraped_urls) < MAX_WEBSITES:
        next_page_data = await scrape_single_url(session, next_page_url, depth, max_depth)
        data.update(next_page_data)
    
    return data

async def scrape_website(url, depth=1, max_depth=100):
    """
    Initiate the scraping process for the given URL.
    """
    async with aiohttp.ClientSession() as session:
        return await scrape_single_url(session, url, depth, max_depth)

def save_scraped_data(scraped_data):
    """
    Save the scraped text and image data to a file.
    """
    with open(SCRAPED_DATA_FILE, 'w', encoding='utf-8', errors='replace') as file:
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
    with open(SCRAPED_URLS_FILE, 'w', encoding='utf-8', errors='replace') as file:
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
    with open(TIME_LOG_FILE, 'w', encoding='utf-8', errors='replace') as file:
        file.write(f"Start time: {time.ctime(start_time)}\n")
        file.write(f"End time: {time.ctime(end_time)}\n")
        file.write(f"Response time: {response_time:.6f} seconds\n")
    print(f"Response time: {response_time:.6f} seconds")

async def create_vector_db():
    """
    Main function to create the vector database.
    """
    start_time = time.time()
    
    scraped_data = await scrape_website(DATA_URL)
    save_scraped_data(scraped_data)
    save_scraped_urls()
    
    documents = create_documents(scraped_data)
    build_vector_store(documents)
    
    end_time = time.time()
    log_time(start_time, end_time)

if __name__ == "__main__":
    asyncio.run(create_vector_db())