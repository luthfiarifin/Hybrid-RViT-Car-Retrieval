import asyncio
import logging
from playwright.async_api import Browser
import json
import random
from urllib.parse import quote_plus

from ..utils.retry_async import retry_async


class OlxScraper:
    """
    Scrapes OLX by using a real Playwright browser to visit the API endpoint directly.
    This is the most robust method for handling advanced anti-bot measures.
    """

    def __init__(
        self,
        browser: Browser,
        semaphore: asyncio.Semaphore,
        images_per_term: int,
        max_pages: int = 10,
    ):
        self.browser = browser
        self.semaphore = semaphore
        self.images_per_term = images_per_term
        self.max_pages = max_pages

    @retry_async(retries=3, delay=10, backoff=2)
    async def scrape(self, search_term: str) -> list[str]:
        """
        Uses a browser to navigate directly to the API URL and parse the JSON content.
        """
        async with self.semaphore:
            context = None
            urls = set()

            try:
                # Use a shared context but a new page for each task
                context = await self.browser.new_context(
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
                )
                page = await context.new_page()

                query = quote_plus(search_term)
                base_url = f"https://www.olx.co.id/api/relevance/v4/search?facet_limit=100&platform=web-desktop&query={query}&relaxedFilters=true"

                for page_num in range(1, self.max_pages + 1):
                    api_url = f"{base_url}&page={page_num}"

                    try:
                        # Navigate the browser directly to the API endpoint
                        await page.goto(
                            api_url, wait_until="domcontentloaded", timeout=30000
                        )

                        # The content of the page will be the raw JSON text
                        json_text = await page.locator("pre").inner_text()
                        data = json.loads(json_text)

                        if not data.get("data"):
                            break  # No more data, we're done with this term

                        for item in data["data"]:
                            if "images" in item and isinstance(item["images"], list):
                                for image_data in item["images"]:
                                    if (
                                        "big" in image_data
                                        and "url" in image_data["big"]
                                    ):
                                        urls.add(image_data["big"]["url"])
                            if len(urls) >= self.images_per_term:
                                break
                    except Exception as e:
                        logging.error(
                            f"[OLX-API-BROWSER] Failed for '{search_term}' on page {page_num}: {e}"
                        )
                        # Let the retry decorator handle the failure
                        raise e

                    if len(urls) >= self.images_per_term:
                        break

                    # Add a small delay between page requests
                    await page.wait_for_timeout(random.randint(500, 1500))

            except Exception as e:
                logging.error(f"[OLX] Top-level failure for '{search_term}': {e}")
            finally:
                if context:
                    await context.close()

            return list(urls)
