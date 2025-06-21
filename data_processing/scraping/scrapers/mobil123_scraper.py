import asyncio
import aiohttp
from bs4 import BeautifulSoup
import random

from ..utils.retry_async import retry_async


class Mobil123Scraper:
    """
    Scraper for Mobil123 that fetches car images based on search terms.
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
        parts = search_term.lower().split()
        brand, model = parts[0], "-".join(parts[1:])
        base_url = f"https://www.mobil123.com/mobil-dijual/{brand}/{model}/indonesia"

        urls = set()
        for page_num in range(1, self.max_pages + 1):
            async with self.semaphore:  # Wait for a free slot before making a request
                search_url = f"{base_url}?page_size=50&page={page_num}"
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
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

            await asyncio.sleep(random.uniform(0.5, 1.5))
        return list(urls)
