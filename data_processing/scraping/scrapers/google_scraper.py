import asyncio
import logging
from playwright.async_api import Browser


class GoogleImageScraper:
    """
    Scraper for Google Images that fetches image URLs based on search terms.
    """

    def __init__(
        self,
        browser: Browser,
        semaphore: asyncio.Semaphore,
        images_per_term: int,
        image_selector: str = "img.YQ4gaf",
    ):
        self.browser = browser
        self.semaphore = semaphore
        self.images_per_term = images_per_term
        self.image_selector = image_selector

    async def scrape(self, search_term: str) -> list[str]:
        async with self.semaphore:
            page = None
            urls = set()

            try:
                page = await self.browser.new_page()
                search_url = f"https://www.google.com/search?q={search_term.replace(' ', '+')}&tbm=isch"

                await page.goto(search_url, wait_until="networkidle", timeout=60000)

                for _ in range(5):
                    await page.evaluate(
                        "window.scrollTo(0, document.body.scrollHeight)"
                    )
                    await asyncio.sleep(1.5)

                    img_elements = await page.query_selector_all(self.image_selector)
                    for img in img_elements:
                        src = await img.get_attribute("src")
                        if src and src.startswith("http"):
                            urls.add(src)
                    if len(urls) >= self.images_per_term:
                        break

            except Exception as e:
                logging.error(f"[Google] Failed to scrape '{search_term}': {e}")
            finally:
                if page:
                    await page.close()

            return list(urls)
