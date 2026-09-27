import os
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

def sanitize_filename(url):
    """Create a safe filename from a URL."""
    parsed = urlparse(url)
    name = f"{parsed.netloc}_{parsed.path}".replace("/", "_").replace("-", "_")
    name = re.sub(r'[^a-zA-Z0-9_]', '', name)
    return name[:50] + ".txt"

def fetch_and_save_urls(source_file, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(source_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Regex to find markdown links: [text](url)
    urls = re.findall(r'\[.*?\]\((https?://[^\s\)]+)\)', content)
    
    print(f"Found {len(urls)} URLs in {source_file}")

    for idx, url in enumerate(urls, 1):
        print(f"[{idx}/{len(urls)}] Fetching: {url}")
        try:
            # Add a timeout and a common user agent to avoid being blocked
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove scripts, styles, and navigation elements
            for el in soup(['script', 'style', 'nav', 'header', 'footer']):
                el.decompose()
                
            text = soup.get_text(separator='\n', strip=True)

            if not text:
                print("  -> No text content found, skipping.")
                continue

            # Save to text file
            filename = sanitize_filename(url)
            filepath = os.path.join(output_dir, filename)
            
            # Prepend the original URL as metadata in the text file
            with open(filepath, 'w', encoding='utf-8') as out_f:
                out_f.write(f"Source URL: {url}\n\n")
                out_f.write(text)
                
            print(f"  -> Saved to {filename}")

        except Exception as e:
            print(f"  -> Failed: {str(e)}")

if __name__ == "__main__":
    SOURCE_FILE = "data/rag/awesome_ai_ml_resources.txt"
    OUTPUT_DIR = "data/rag/scraped_content"
    
    fetch_and_save_urls(SOURCE_FILE, OUTPUT_DIR)
    print("\nDone! You can now ingest the contents of data/rag/scraped_content/ using your pipeline.")
