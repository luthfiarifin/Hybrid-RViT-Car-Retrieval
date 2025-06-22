import asyncio
import logging
import aiohttp
from bs4 import BeautifulSoup
import random

from ..utils.retry_async import retry_async


class CarmudiScraper:
    """
    Scraper for Carmudi that fetches car images based on search terms.
    """

    def __init__(
        self,
        session: aiohttp.ClientSession,
        semaphore: asyncio.Semaphore,
        images_per_term: int,
        max_pages: int = 5,
    ):
        self.session = session
        self.semaphore = semaphore
        self.images_per_term = images_per_term
        self.max_pages = max_pages

    @retry_async(retries=3, delay=5)
    async def scrape(self, search_term: str) -> list[str]:
        """
        Scrapes car images from Carmudi based on the search term.
        """
        parts = search_term.lower().split()
        brand, model = parts[0], "-".join(parts[1:])
        base_url = f"https://www.carmudi.co.id/mobil-dijual/{brand}/{model}/indonesia"

        urls = set()
        for page_num in range(1, self.max_pages + 1):
            async with self.semaphore:  # Wait for a free slot
                search_url = f"{base_url}?page_size=50&page={page_num}"

                try:
                    headers = {
                        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36",
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
                        "Accept-Encoding": "gzip, deflate, br",
                        "Accept-Language": "en-US,en;q=0.9,id;q=0.8",
                        "Cache-Control": "max-age=0",
                        "Sec-Ch-Ua": '"Google Chrome";v="137", "Chromium";v="137", "Not/A)Brand";v="24"',
                        "Sec-Ch-Ua-Mobile": "?0",
                        "Sec-Ch-Ua-Platform": '"macOS"',
                        "Sec-Fetch-Dest": "empty",
                        "Sec-Fetch-Mode": "navigate",
                        "Sec-Fetch-Site": "same-origin",
                    }

                    async with self.session.get(
                        search_url, headers=headers, timeout=20
                    ) as response:
                        if response.status != 200:
                            break

                        html = await response.text()
                        soup = BeautifulSoup(html, "html.parser")
                        car_listings = soup.find_all("article", class_="listing--card")

                        if not car_listings:
                            break

                        for listing in car_listings:
                            image_tag = listing.find("img", class_="listing__img")
                            if image_tag and "data-src" in image_tag.attrs:
                                urls.add(image_tag["data-src"])
                            elif image_tag and "src" in image_tag.attrs:
                                urls.add(image_tag["src"])
                            if len(urls) >= self.images_per_term:
                                break

                        if len(urls) >= self.images_per_term:
                            break
                except Exception as e:
                    logging.error(
                        f"[Carmudi] Failed to scrape '{search_term}' on page {page_num}, url: {search_url}: {e}"
                    )
                    raise

            await asyncio.sleep(random.uniform(0.5, 1.5))
        return list(urls)
