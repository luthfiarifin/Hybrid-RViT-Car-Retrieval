import logging
import os
import asyncio
import aiohttp
from playwright.async_api import async_playwright
import pandas as pd
import json
from tqdm.asyncio import tqdm as aio_tqdm
from collections import defaultdict

from .scrapers.olx_scraper import OlxScraper
from .scrapers.mobil123_scraper import Mobil123Scraper
from .scrapers.carmudi_scraper import CarmudiScraper


class MasterScraper:
    """
    Master scraper that orchestrates scraping from multiple sources,
    manages image downloads, and handles robust cleanup.
    """

    def __init__(
        self,
        config_path="config.json",
        images_dir="data/images",
        csv_path="reports/master_scrape_log.csv",
        summary_csv_path="reports/scrape_summary_report.csv",
    ):
        with open(config_path, "r") as f:
            config = json.load(f)

        self.car_classes = config["car_classes"]
        self.images_per_term = config["images_per_term"]
        self.max_pages_per_term = config.get("max_pages_per_term", 10)

        self.image_download_path = images_dir
        self.master_log_path = csv_path
        self.summary_report_path = summary_csv_path
        self.master_data_list = []

    async def run(self):
        seen_image_urls = set()
        scrape_stats = defaultdict(lambda: {"unique_found": 0, "duplicates_skipped": 0})

        async with async_playwright() as p, aiohttp.ClientSession() as session:
            browser = None
            try:
                browser = await p.chromium.launch(headless=False)

                browser_semaphore = asyncio.Semaphore(2)
                http_semaphore = asyncio.Semaphore(5)

                mobil123_scraper = Mobil123Scraper(
                    session,
                    http_semaphore,
                    self.images_per_term,
                    self.max_pages_per_term,
                )
                carmudi_scraper = CarmudiScraper(
                    session,
                    http_semaphore,
                    self.images_per_term,
                    self.max_pages_per_term,
                )
                olx_scraper = OlxScraper(
                    browser,
                    browser_semaphore,
                    self.images_per_term,
                    self.max_pages_per_term,
                )

                tasks = []
                for class_name, search_terms in self.car_classes.items():
                    for term in search_terms:
                        tasks.append(
                            {
                                "source": "OLX",
                                "task": olx_scraper.scrape(term),
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

            except asyncio.CancelledError:
                logging.warning("Scraping process was cancelled by the user.")
            finally:
                if browser and browser.is_connected():
                    await browser.close()

            # Process and deduplicate results
            for i, url_list in enumerate(results):
                source = tasks[i]["source"]
                for url in url_list:
                    if url not in seen_image_urls:
                        seen_image_urls.add(url)
                        scrape_stats[source]["unique_found"] += 1
                        task_info = tasks[i]
                        class_path = os.path.join(
                            self.image_download_path, task_info["class"]
                        )
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

            # Save scraping performance summary
            summary_df = (
                pd.DataFrame.from_dict(scrape_stats, orient="index")
                .reset_index()
                .rename(columns={"index": "Source"})
            )
            summary_df.to_csv(self.summary_report_path, index=False)

            # Save the list of images to be downloaded
            master_df = pd.DataFrame(self.master_data_list)
            master_df.to_csv(self.master_log_path, index=False)

            print(
                f"Scraping finished. Found {len(self.master_data_list)} unique images to download."
            )
