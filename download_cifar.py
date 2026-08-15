import os
import tarfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

URLS = [
    "https://cs231n.stanford.edu/cifar-10-python.tar.gz",
    "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
]
DATA_DIR = "./data"
ARCHIVE_PATH = os.path.join(DATA_DIR, "cifar-10-python.tar.gz")
EXTRACT_DIR = os.path.join(DATA_DIR, "cifar-10-batches-py")
NUM_THREADS = 16


def download_range(url: str, start: int, end: int, chunk_idx: int) -> tuple[int, bytes]:
    headers = {"Range": f"bytes={start}-{end}", "User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return chunk_idx, resp.read()


def download_from_url(url: str) -> bool:
    try:
        print(f"[Dataset] Fetching content length from {url}...")
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            total_length = int(resp.headers.get("Content-Length"))

        print(f"[Dataset] Size: {total_length / (1024*1024):.2f} MB. Downloading using {NUM_THREADS} threads...")
        chunk_size = total_length // NUM_THREADS
        ranges = []
        for i in range(NUM_THREADS):
            start = i * chunk_size
            end = total_length - 1 if i == NUM_THREADS - 1 else (i + 1) * chunk_size - 1
            ranges.append((start, end, i))

        chunks = [b""] * NUM_THREADS
        completed = 0
        with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
            futures = {executor.submit(download_range, url, r[0], r[1], r[2]): r[2] for r in ranges}
            for future in as_completed(futures):
                idx, data = future.result()
                chunks[idx] = data
                completed += 1
                print(f"[Dataset] Chunk {completed}/{NUM_THREADS} ready ({len(data)/(1024*1024):.2f} MB)", flush=True)

        print("[Dataset] Writing archive file...", flush=True)
        with open(ARCHIVE_PATH, "wb") as f:
            for chunk in chunks:
                f.write(chunk)

        print("[Dataset] Extracting tar archive...", flush=True)
        with tarfile.open(ARCHIVE_PATH, "r:gz") as tar:
            tar.extractall(path=DATA_DIR)

        print("[Dataset] CIFAR-10 successfully downloaded and extracted!", flush=True)
        return True
    except Exception as e:
        print(f"[Dataset] Failed downloading from {url}: {e}", flush=True)
        if os.path.exists(ARCHIVE_PATH):
            os.remove(ARCHIVE_PATH)
        return False


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(EXTRACT_DIR) and len(os.listdir(EXTRACT_DIR)) >= 5:
        print("[Dataset] CIFAR-10 already extracted and ready!", flush=True)
        return

    for url in URLS:
        if download_from_url(url):
            return

    raise RuntimeError("All dataset download mirrors failed!")


if __name__ == "__main__":
    main()
