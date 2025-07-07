import re
from urllib.parse import urljoin, urlparse

import markdown
from bs4 import BeautifulSoup
from crawl4ai import DefaultMarkdownGenerator, PruningContentFilter


# Process paragraph text
def process_text(txt: str) -> list:
    # Split paragraph by highlight part `_**`
    h_p = re.split(r"_\*\*", txt.strip())
    t = []

    for text in h_p:
        try:
            # Split each section into key-value pairs at `**_`
            matches = re.split(r"\*\*_\n", text, maxsplit=1)
            key = matches[0].strip()

            # If there's no value, store raw text instead
            if len(matches) <= 1 or not matches[1].strip():
                section = text.strip()  # Store full raw text
            else:
                value = matches[1].strip()
                section = {key: value}
        except Exception as e:
            section = {"Error": str(e), "RawText": text}  # Error handling

        t.append(section)

    return t


# Process full markdown
def preprocess_markdown(md_text: str) -> list:
    # Check if markdown contains headers (##)
    if "##" not in md_text:
        return process_text(md_text)

    # Split by "##" but keep meaningful content
    raw_sections = re.split(r"##\s*", md_text)

    md = []

    for section in raw_sections:
        d = {}
        section = section.strip()
        if not section:  # Skip empty sections
            continue

        # Extract the first line as the header, remaining as content
        parts = section.split("\n", 1)
        if len(parts) == 2:
            header, content = parts[0].strip(), parts[1].strip()
        else:
            header, content = parts[0].strip(), ""

        # Ignore completely empty headers
        if not header:
            continue

        # Remove asterisks (**) from headers
        clean_header = re.sub(r"\*+", "", header).strip()

        # Convert markdown content to plain text
        clean_content = (
            BeautifulSoup(markdown.markdown(content), "html.parser")
            .get_text(separator=" ")
            .strip()
        )
        d[clean_header] = process_text(clean_content)
        md.append(d)

    return md


async def fetch_and_process(
    crawler,
    url,
    max_depth,
    depth=0,
    visited=None,
    bypass=False,
    results=None,
):
    if visited is None:
        visited = set()
    if results is None:
        results = []

    base_url = url.rstrip("/")
    parsed_url = urlparse(url)
    normalized_url = parsed_url._replace(fragment="").geturl().rstrip("/")

    if normalized_url in visited:
        return results

    visited.add(normalized_url)

    try:
        result = await crawler.arun(
            url=normalized_url,
            bypass_cache=bypass,
            magic=True,
            exclude_external_links=True,
            exclude_social_media_links=True,
            markdown_generator=DefaultMarkdownGenerator(
                content_filter=PruningContentFilter(
                    threshold=0.48,
                    threshold_type="fixed",
                    min_word_threshold=0,
                ),
                options={"ignore_links": True},
            ),
        )

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
        results.append(page_data)

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
                        bypass=bypass,
                    )

    except Exception as e:
        print(f"Error processing URL {normalized_url}: {e}")

    return results
