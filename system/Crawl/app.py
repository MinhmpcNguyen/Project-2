import json
import os
from typing import Optional

from Backend_crawl.crawl import fetch_and_process
from Backend_crawl.reformat import (
    process_json,
    remove_global_duplicates,
    split_long_entries,
)
from crawl4ai import AsyncWebCrawler, BrowserConfig
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from RAG.chunking import chunk_by_length
from RAG.len_db import upload_to_pinecone
from RAG.metadata import create_metadata

app = FastAPI()

SAVE_DIR = "crawl_results_json"
os.makedirs(SAVE_DIR, exist_ok=True)


class CrawlRequest(BaseModel):
    url: str
    max_depth: int = 3
    bypass: Optional[bool] = False
    filename: str


@app.post("/crawl")
async def crawl_website(req: CrawlRequest):
    name = req.filename.strip()
    if not name:
        raise HTTPException(status_code=400, detail="filename is required.")

    if not name.endswith(".json"):
        filename = name + ".json"
    else:
        filename = name
        name = name[:-5]

    file_path = os.path.join(SAVE_DIR, filename)
    index_name = name.replace("_", "-") + "-index"

    browser_config = BrowserConfig(browser_type="chromium", headless=True)
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    try:
        #  Crawl markdown
        markdown_pages = await fetch_and_process(
            crawler=crawler,
            url=req.url,
            max_depth=req.max_depth,
            bypass=req.bypass,
        )

        #  Pipeline xử lý
        step1 = process_json(markdown_pages)
        step2 = split_long_entries(step1)
        step3 = remove_global_duplicates(step2)

        #  Chunk nội dung theo độ dài
        final_docs = []
        for doc in step3:
            url = doc["url"]
            chunks = chunk_by_length(doc["content"], max_words=350)
            final_docs.append({"url": url, "content": chunks})

        #  Lưu JSON
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(final_docs, f, indent=2, ensure_ascii=False)

        #  Tạo metadata + lưu + upload lên Pinecone
        output_dir = os.path.join("no_api", "results", name.replace("-", "_"))
        metadata_list = create_metadata(final_docs, output_dir=output_dir)
        upload_to_pinecone(metadata_list, index_name=index_name)

        return {
            "message": "Crawl completed and uploaded to Pinecone",
            "pages_crawled": len(final_docs),
            "json_file": file_path,
            "pinecone_index": index_name,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
