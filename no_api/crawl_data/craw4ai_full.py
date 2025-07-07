import asyncio
import json
import os
import sys
import time
from urllib.parse import urljoin, urlparse

from crawl4ai import AsyncWebCrawler, BrowserConfig
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from process_markdown import preprocess_markdown  # bạn tự viết file này hoặc để trống


async def fetch_and_process(
    crawler,
    url,
    max_depth,
    depth=0,
    visited=None,
    bypass=False,
    results=None,
    results_file="results.json",
):
    """
    Fetch and process a single URL and recursively visit internal links, saving results to a JSON file.
    """

    #  Ensure `visited` and `results` persist across recursive calls
    if visited is None:
        visited = set()
    if results is None:
        results = []

    base_url = url.rstrip("/")
    parsed_url = urlparse(url)
    normalized_url = parsed_url._replace(fragment="").geturl().rstrip("/")

    if normalized_url in visited:
        return results  #  Return existing results instead of resetting

    visited.add(normalized_url)

    try:
        print(f"Processing URL: {normalized_url} at depth {depth}")

        #  Crawl the URL
        result = await crawler.arun(
            url=normalized_url,
            bypass_cache=bypass,  #  Set `bypass_cache` dynamically
            magic=True,  #  Enables smart Crawl4AI strategies
            exclude_external_links=True,
            exclude_social_media_links=True,
            markdown_generator=DefaultMarkdownGenerator(
                content_filter=PruningContentFilter(
                    threshold=0.48,  #  Removes content with a low relevance score
                    threshold_type="fixed",
                    min_word_threshold=0,
                ),
                options={"ignore_links": True},
            ),
        )

        #  Extract markdown content safely
        markdown = (
            result.markdown_v2.fit_markdown
            if result.markdown_v2 and result.markdown_v2.fit_markdown
            else ""
        )
        processed_markdown = preprocess_markdown(markdown)

        page_data = {
            "depth": depth,
            "url": normalized_url,
            "markdown": processed_markdown,
        }
        results.append(page_data)  #  Append to results **without resetting**

        #  Save continuously to prevent data loss
        with open(results_file, "w", encoding="utf-8") as file:
            json.dump(results, file, indent=4, ensure_ascii=False)

        #  Recursively process internal links
        if depth < max_depth:
            for link in result.links["internal"]:
                absolute_url = urljoin(base_url, link["href"])
                parsed_url = urlparse(absolute_url)
                normalized_url = parsed_url._replace(fragment="").geturl().rstrip("/")

                if normalized_url not in visited:
                    await fetch_and_process(
                        crawler,
                        normalized_url,
                        max_depth,
                        depth + 1,
                        visited=visited,
                        results=results,
                        results_file=results_file,
                    )

    except Exception as e:
        print(f"Error processing URL {normalized_url}: {e}")

    return results  #  Return accumulated results


# --- Main ---
async def main():
    start_url = "https://docs.trava.finance/portal"
    max_depth = 10
    results_dir = "crawl_results"
    results_file = os.path.join(results_dir, "results.json")

    os.makedirs(results_dir, exist_ok=True)

    browser_config = BrowserConfig(browser_type="chromium", headless=True)
    crawler = AsyncWebCrawler(config=browser_config)

    await crawler.start()

    print(f"Bắt đầu crawling từ: {start_url} (max_depth={max_depth})")
    start_time = time.time()

    try:
        await fetch_and_process(
            crawler=crawler,
            url=start_url,
            max_depth=max_depth,
            results_file=results_file,
        )

        elapsed_time = time.time() - start_time
        print("\n Crawl hoàn tất.")
        print(f"Kết quả lưu ở: {results_file}")
        print(f"Thời gian thực hiện: {elapsed_time:.2f} giây")

    except Exception as e:
        print(f"Lỗi trong quá trình crawling: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n Dừng chương trình bởi người dùng.")
