import json
import os
from urllib.parse import urlparse

from Backend_crawl.crawl import fetch_and_process
from Backend_crawl.reformat import process_markdown_entry, split_long_entries
from crawl4ai import AsyncWebCrawler, BrowserConfig
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from RAG.rechunk_and_faiss import process_rechunk_and_faiss

SAVE_DIR = "crawl_results_json"
os.makedirs(SAVE_DIR, exist_ok=True)

app = FastAPI()


class CrawlRequest(BaseModel):
    url: HttpUrl
    max_depth: int = 3


def remove_global_duplicates(docs):
    """
    Loại bỏ các entry trùng lặp toàn cục (giữa các URL).
    """
    seen = set()
    for doc in docs:
        filtered = []
        for entry in doc["content"]:
            if isinstance(entry, str) and entry not in seen:
                seen.add(entry)
                filtered.append(entry)
        doc["content"] = filtered
    return docs


@app.post("/crawl")
async def crawl_website(req: CrawlRequest):
    parsed_url = urlparse(str(req.url))
    domain = parsed_url.netloc.replace(".", "_")
    file_path = os.path.join(SAVE_DIR, f"{domain}.json")

    browser_config = BrowserConfig(browser_type="chromium", headless=True)
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    try:
        results = await fetch_and_process(
            crawler=crawler,
            url=str(req.url),
            max_depth=req.max_depth,
            bypass=False,
        )

        final_docs = []
        for page in results:
            url = page.get("url", "unknown")
            raw_markdown = page.get("markdown", [])

            cleaned = process_markdown_entry(raw_markdown)
            segmented = split_long_entries(cleaned)

            final_docs.append({"url": url, "content": segmented})

        final_docs = remove_global_duplicates(final_docs)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(final_docs, f, indent=2, ensure_ascii=False)

        await process_rechunk_and_faiss(domain, file_path)

        return {
            "message": "Crawl + Rechunk + FAISS completed",
            "pages_crawled": len(results),
            "json_file": file_path,
            "domain": domain,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
