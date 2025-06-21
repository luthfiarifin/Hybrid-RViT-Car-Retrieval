import os
import asyncio
import aiohttp
from playwright.async_api import async_playwright
import logging
import pandas as pd
import json
from tqdm.asyncio import tqdm as aio_tqdm
from collections import defaultdict

from .scrapers.google_scraper import GoogleImageScraper
from .scrapers.mobil123_scraper import Mobil123Scraper
from .scrapers.carmudi_scraper import CarmudiScraper

logging.basicConfig(
    level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s"
)


class MasterScraper:
    """
    Master scraper that orchestrates scraping from multiple sources (Google, Mobil123, Carmudi)
    and manages image downloads and logging.
    """

    def __init__(
        self,
        images_dir,
        config_path="config.json",
        csv_path="master_scrape_log.csv",
        summary_csv_path="scrape_summary_report.csv",
    ):
        with open(config_path, "r") as f:
            config = json.load(f)

        self.csv_path = csv_path
        self.images_dir = images_dir
        self.summary_csv_path = summary_csv_path
        self.car_classes = config["car_classes"]
        self.images_per_term = config["images_per_term"]
        self.max_pages_per_term = config.get("max_pages_per_term", 5)

        self.master_data_list = []

    async def download_image(self, session, semaphore, url, path, filename):
        """
        Downloads an image robustly, respecting a semaphore to limit concurrency.
        """
        async with semaphore:
            filepath = os.path.join(path, filename)

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Referer": "https://www.google.com/",
                "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
            }

            try:
                async with session.get(url, timeout=60, headers=headers) as response:
                    if response.status == 200:
                        content_type = response.headers.get("Content-Type", "")
                        if "image" not in content_type:
                            return {
                                "filepath": url,
                                "status": "failed",
                                "reason": f"Not an image (Content-Type: {content_type})",
                            }

                        content = await response.read()
                        with open(filepath, "wb") as f:
                            f.write(content)
                        return {
                            "filepath": filepath,
                            "status": "success",
                            "reason": None,
                        }

                    return {
                        "filepath": url,
                        "status": "failed",
                        "reason": f"HTTP {response.status}",
                    }
            except Exception as e:
                return {"filepath": url, "status": "failed", "reason": str(e)}

    async def run(self):
        seen_image_urls = set()
        scrape_stats = defaultdict(lambda: {"unique_found": 0, "duplicates_skipped": 0})

        async with async_playwright() as p, aiohttp.ClientSession() as session:
            browser = await p.chromium.launch(headless=True)

            browser_semaphore = asyncio.Semaphore(2)
            http_semaphore = asyncio.Semaphore(5)

            mobil123_scraper = Mobil123Scraper(
                session, http_semaphore, self.images_per_term, self.max_pages_per_term
            )
            carmudi_scraper = CarmudiScraper(
                session, http_semaphore, self.images_per_term, self.max_pages_per_term
            )

            tasks = []
            for class_name, search_terms in self.car_classes.items():
                for term in search_terms:
                    google_scraper = GoogleImageScraper(
                        browser, browser_semaphore, self.images_per_term
                    )

                    tasks.append(
                        {
                            "source": "Google",
                            "task": google_scraper.scrape(term),
                            "class": class_name,
                            "term": term,
                        }
                    )
                    tasks.append(
                        {
                            "source": "Mobil123",
                            "task": mobil123_scraper.scrape(term),
                            "class": class_name,
                            "term": term,
                        }
                    )
                    tasks.append(
                        {
                            "source": "Carmudi",
                            "task": carmudi_scraper.scrape(term),
                            "class": class_name,
                            "term": term,
                        }
                    )

            scrape_coroutines = [t["task"] for t in tasks]
            results = await aio_tqdm.gather(
                *scrape_coroutines, desc="Scraping All Sources"
            )

            await browser.close()

            for i, url_list in enumerate(results):
                source = tasks[i]["source"]
                for url in url_list:
                    if url not in seen_image_urls:
                        seen_image_urls.add(url)
                        scrape_stats[source]["unique_found"] += 1
                        task_info = tasks[i]
                        class_path = os.path.join(self.images_dir, task_info["class"])
                        os.makedirs(class_path, exist_ok=True)
                        filename = f"{task_info['class']}_{hash(url)}.jpg"
                        self.master_data_list.append(
                            {
                                "class": task_info["class"],
                                "search_term": task_info["term"],
                                "source": source,
                                "image_url": url,
                                "image_path": os.path.join(class_path, filename),
                            }
                        )
                    else:
                        scrape_stats[source]["duplicates_skipped"] += 1

            summary_df = pd.DataFrame.from_dict(scrape_stats, orient="index")
            summary_df.index.name = "Source"
            summary_df.to_csv(self.summary_csv_path)

            download_semaphore = asyncio.Semaphore(20)
            download_tasks = [
                self.download_image(
                    session,
                    download_semaphore,
                    item["image_url"],
                    os.path.dirname(item["image_path"]),
                    os.path.basename(item["image_path"]),
                )
                for item in self.master_data_list
            ]
            download_results = await aio_tqdm.gather(
                *download_tasks, desc="Downloading Images"
            )

            master_df = pd.DataFrame(self.master_data_list)
            master_df["download_status"] = [res["status"] for res in download_results]
            master_df["reason"] = [res["reason"] for res in download_results]
            master_df.to_csv(self.csv_path, index=False)
