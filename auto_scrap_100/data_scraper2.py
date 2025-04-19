import os
import time
import requests
import urllib3
from duckduckgo_search import DDGS
from urllib.parse import urlparse
import mimetypes
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Suppress unverified HTTPS warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def create_session():
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    })
    return session

def get_extension_from_url_or_content(url, response):
    path = urlparse(url).path
    ext = os.path.splitext(path)[1]
    if ext and len(ext) <= 5:
        return ext.lstrip(".")
    content_type = response.headers.get("content-type", "").split(";")[0]
    guessed = mimetypes.guess_extension(content_type)
    if guessed:
        return guessed.lstrip(".")
    return "jpg"

def download_image(url, folder, person_name, seen_hashes, session, bad_domains, lock, max_retries=3):
    parsed = urlparse(url)
    if parsed.netloc in bad_domains:
        return None

    for attempt in range(max_retries):
        try:
            response = session.get(url, stream=True, timeout=(5, 15), verify=False)
            if response.status_code == 200 and response.headers.get("content-type", "").startswith("image"):
                img_bytes = response.content
                img_hash = hashlib.md5(img_bytes).hexdigest()

                with lock:
                    if img_hash in seen_hashes:
                        return None
                    seen_hashes.add(img_hash)

                ext = get_extension_from_url_or_content(url, response)
                file_path = os.path.join(folder, f"{person_name.replace(' ', '_')}_{img_hash}.{ext}")
                with open(file_path, "wb") as f:
                    f.write(img_bytes)
                return file_path
        except Exception as e:
            time.sleep(2)  # brief wait before retry

    bad_domains.add(parsed.netloc)
    return None

def fetch_image_urls(keyword, total_needed, seen_urls, retries=3):
    for attempt in range(retries):
        try:
            with DDGS() as ddgs:
                results = ddgs.images(keywords=keyword, max_results=total_needed*5, safesearch="Off", size="Large")
                urls = []
                for r in results:
                    url = r.get("image")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        urls.append(url)
                    if len(urls) >= total_needed:
                        break
                return urls
        except Exception as e:
            print(f"[!] DuckDuckGo failed on attempt {attempt+1}/{retries}: {e}")
            time.sleep(5)
    return []

def download_images(search_terms, save_name, num_images=1000, save_folder="images", threads=10, batch_size=300, pause_time=60):
    folder = os.path.join(save_folder, save_name.replace(" ", "_"))
    os.makedirs(folder, exist_ok=True)

    session = create_session()
    seen_hashes = set()
    seen_urls = set()
    bad_domains = set()
    lock = Lock()
    downloaded_count = 0
    batch_number = 0
    term_index = 0

    print(f"\n🔍 Starting download of {num_images} images for: {save_name}")

    while downloaded_count < num_images:
        keyword = search_terms[term_index % len(search_terms)]
        remaining = num_images - downloaded_count
        batch = min(batch_size, remaining)
        print(f"\n🚀 Batch {batch_number + 1}: Searching for '{keyword}' to fetch {batch} images...")

        image_urls = fetch_image_urls(keyword, batch, seen_urls)
        if not image_urls:
            print(f"❌ No more results for '{keyword}'. Trying next keyword...")
            term_index += 1
            if term_index >= len(search_terms):
                print("🛑 All search terms exhausted.")
                break
            continue

        with ThreadPoolExecutor(max_workers=threads) as executor:
            future_to_url = {
                executor.submit(download_image, url, folder, save_name, seen_hashes, session, bad_domains, lock): url
                for url in image_urls
            }

            for future in as_completed(future_to_url):
                try:
                    result = future.result(timeout=30)
                    if result:
                        downloaded_count += 1
                        print(f"✅ Downloaded [{downloaded_count}/{num_images}] → {os.path.basename(result)}")
                    if downloaded_count >= num_images:
                        break
                except Exception as e:
                    print(f"[!] Error in thread: {e}")

        batch_number += 1
        term_index += 1

        # Memory control: Limit set size
        if len(seen_urls) > 50000:
            seen_urls = set(list(seen_urls)[-25000:])
        if len(seen_hashes) > 50000:
            seen_hashes = set(list(seen_hashes)[-25000:])

        if downloaded_count < num_images:
            print("🕐 Taking a 60-second break before next batch...\n")
            time.sleep(pause_time)

    print(f"\n🎉 Done! Downloaded {downloaded_count} images of '{save_name}' to → {folder}")

if __name__ == "__main__":
    download_images(
        search_terms=[
            "Benedict Cumberbatch",
            "Benedict Cumberbatch awards",
            "Benedict Cumberbatch actor",
            "Benedict Cumberbatch potraits",
            "Benedict Cumberbatch hd",
            "Benedict Cumberbatch movie scenes",
            "Benedict Cumberbatch photoshoot",
        ],
        save_name="Benedict Cumberbatch",
        num_images=1000
    )
