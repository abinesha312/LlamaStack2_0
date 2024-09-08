import urllib.parse
import urllib3
import json
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
SCRAPED_DATA_FILE = 'scraped_data.json'
SCRAPED_URLS_FILE = 'scraped_urls.json'
TIME_LOG_FILE = 'time_log.json'
MAX_WEBSITES = 20
TIMEOUT = 10
RETRIES = 3
scraped_urls = set()

def is_valid_response(status_code):
    return 200 <= status_code <= 205

async def fetch_page_content(session, url, timeout=TIMEOUT, retries=RETRIES):
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

def parse_content(html_content, url):
    soup = BeautifulSoup(html_content, 'lxml')
    body = soup.find('body')
    body_elements = []
    if body:
        for element in body.stripped_strings:
            if element.strip():  # Only add non-empty strings
                body_elements.append(element.strip())
    
    paragraphs = soup.find_all('p')
    text_data = " ".join(para.get_text(strip=True) for para in paragraphs)
    images = soup.find_all('img', src=True)
    image_data = [img['src'] for img in images]
    headings = {f"h{i}": [h.get_text(strip=True) for h in soup.find_all(f'h{i}')] for i in range(1, 7)}
    lists = [ul.get_text(separator=", ", strip=True) for ul in soup.find_all('ul')]
    urls = extract_links(soup, url)
    structured_data = {
        "URL": url,
        "Body": body_elements,
        "text": text_data,
        "images": image_data,
        "headings": headings,
        "lists": lists,
        "URLs": urls
    }
    return structured_data


def extract_links(soup, base_url):
    links = soup.find_all('a', href=True)
    unt_pattern = re.compile(r'^https?://(?:[\w-]+\.)*unt\.edu')
    
    extracted_links = []
    for link in links:
        href = link['href']
        full_url = urllib.parse.urljoin(base_url, href)
        if unt_pattern.match(full_url):
            extracted_links.append(full_url)
    
    return extracted_links

def find_next_page(soup):
    next_page = soup.find('a', text='Next')
    if next_page and 'href' in next_page.attrs:
        return next_page['href']
    return None

async def scrape_single_url(session, url, depth, max_depth):
    global scraped_urls
    if url in scraped_urls:
        print(f"Already visited {url}, skipping.")
        return {}
    html_content = await fetch_page_content(session, url)
    if html_content is None:
        return {}
    soup = BeautifulSoup(html_content, 'lxml')
    structured_data = parse_content(html_content, url)
    scraped_urls.add(url)
    data = {url: structured_data}
    if depth < max_depth:
        unt_links = extract_links(soup, url)
        tasks = [scrape_single_url(session, link, depth + 1, max_depth) for link in unt_links if link not in scraped_urls and len(scraped_urls) < MAX_WEBSITES]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, dict):
                data.update(result)
    return data

async def scrape_website(url, depth=1, max_depth=5000):
    async with aiohttp.ClientSession() as session:
        return await scrape_single_url(session, url, depth, max_depth)

def save_scraped_data(scraped_data):
    numbered_data = {f"URL_{str(i+1).zfill(2)}": data for i, data in enumerate(scraped_data.values())}
    with open(SCRAPED_DATA_FILE, 'w', encoding='utf-8') as file:
        json.dump(numbered_data, file, indent=4, ensure_ascii=False)

def save_body_content(scraped_data):
    body_content = {}
    for i, data in enumerate(scraped_data.values(), 1):
        url_key = f"URL_{str(i).zfill(2)}"
        body_content[url_key] = data['Body']
    with open('body_content.json', 'w', encoding='utf-8') as file:
        json.dump(body_content, file, indent=4, ensure_ascii=False)
    return body_content

def save_scraped_urls():
    urls_data = {f"URL_{str(i+1).zfill(2)}": url for i, url in enumerate(scraped_urls)}
    with open(SCRAPED_URLS_FILE, 'w', encoding='utf-8') as file:
        json.dump(urls_data, file, indent=4, ensure_ascii=False)

def create_documents(body_content):
    documents = []
    for url, content in body_content.items():
        if content:
            try:
                doc = Document(page_content=' '.join(content) if isinstance(content, list) else str(content), metadata={"source": url})
                documents.append(doc)
            except Exception as e:
                print(f"Error creating document for {url}: {e}")
                print(f"Content type: {type(content)}")
                print(f"Content: {content[:100]}...")  # Print first 100 characters
    return documents

def build_vector_store(documents):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': device}
    )
    db = FAISS.from_documents(documents, embeddings)
    db.save_local(DB_FAISS_PATH)

def log_time(start_time, end_time):
    response_time = end_time - start_time
    time_log = {
        "start_time": time.ctime(start_time),
        "end_time": time.ctime(end_time),
        "response_time": f"{response_time:.6f} seconds"
    }
    with open(TIME_LOG_FILE, 'w', encoding='utf-8') as file:
        json.dump(time_log, file, indent=4, ensure_ascii=False)
    print(f"Response time: {response_time:.6f} seconds")

async def create_vector_db():
    start_time = time.time()
    try:
        scraped_data = await asyncio.wait_for(scrape_website(DATA_URL), timeout=3600)  # 1 hour timeout
        save_scraped_data(scraped_data)
        body_content = save_body_content(scraped_data)
        save_scraped_urls()
        documents = create_documents(body_content)
        if documents:
            build_vector_store(documents)
        else:
            print("No valid documents to build vector store.")
    except asyncio.TimeoutError:
        print("Scraping process timed out after 1 hour.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        end_time = time.time()
        log_time(start_time, end_time)

if __name__ == "__main__":
    asyncio.run(create_vector_db())