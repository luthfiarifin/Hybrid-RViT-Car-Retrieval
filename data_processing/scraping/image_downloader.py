import os
import asyncio
import aiohttp
from playwright.async_api import async_playwright
import pandas as pd
from tqdm.asyncio import tqdm as aio_tqdm


class ImageDownloader:
    """
    Reads a log file of URLs and downloads them concurrently using the
    best method for each source.
    """

    def __init__(
        self,
        csv_path="reports/master_scrape_log.csv",
    ):
        self.csv_path = csv_path

    async def _download_http(self, session, semaphore, item):
        """Downloads an image using fast, direct HTTP requests."""
        async with semaphore:
            url, path, source = item["image_url"], item["image_path"], item["source"]
            referer_map = {
                "Mobil123": "https://www.mobil123.com/",
                "Carmudi": "https://www.carmudi.co.id/",
                "Google": "https://www.google.com/",
            }
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)...",
                "Referer": referer_map.get(source, "https://www.google.com/"),
            }

            try:
                async with session.get(url, timeout=20, headers=headers) as response:
                    if response.status == 200 and "image" in response.headers.get(
                        "Content-Type", ""
                    ):
                        with open(path, "wb") as f:
                            f.write(await response.read())
                        return {"image_path": path, "status": "success", "reason": None}
                    return {
                        "image_path": path,
                        "status": "failed",
                        "reason": f"HTTP {response.status}",
                    }
            except Exception as e:
                return {"image_path": path, "status": "failed", "reason": str(e)}

    async def _download_browser(self, browser, semaphore, item):
        """Downloads an image using a full browser page to bypass protection."""
        async with semaphore:
            url, path = item["image_url"], item["image_path"]
            context = None
            try:
                context = await browser.new_context(java_script_enabled=False)
                page = await context.new_page()
                await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                await page.locator("img").first.screenshot(path=path)
                await context.close()
                return {"image_path": path, "status": "success", "reason": None}
            except Exception as e:
                if context:
                    await context.close()
                return {"image_path": path, "status": "failed", "reason": str(e)}

    async def download_all_images(self):
        """Main method to read the log and run all download tasks."""
        if not os.path.exists(self.csv_path):
            print(
                f"Log file not found at {self.csv_path}. Please run the scraper first."
            )
            return

        df = pd.read_csv(self.csv_path)
        print(f"Found {len(df)} images to download.")

        olx_items = df[df["source"] == "OLX"].to_dict("records")
        other_items = df[df["source"] != "OLX"].to_dict("records")

        async with async_playwright() as p, aiohttp.ClientSession() as session:
            browser = await p.chromium.launch(headless=False)

            browser_dl_semaphore = asyncio.Semaphore(4)
            http_dl_semaphore = asyncio.Semaphore(20)

            olx_tasks = [
                self._download_browser(browser, browser_dl_semaphore, item)
                for item in olx_items
            ]
            other_tasks = [
                self._download_http(session, http_dl_semaphore, item)
                for item in other_items
            ]

            all_tasks = olx_tasks + other_tasks
            results = await aio_tqdm.gather(*all_tasks, desc="Downloading All Images")

            await browser.close()

        # Update the DataFrame with download results
        results_map = {res["image_path"]: res for res in results if res}
        df["download_status"] = df["image_path"].map(
            lambda p: results_map.get(p, {}).get("status", "failed")
        )
        df["reason"] = df["image_path"].map(
            lambda p: results_map.get(p, {}).get("reason", "download_not_attempted")
        )

        # Overwrite the log file with the new status
        df.to_csv(self.csv_path, index=False)
        print("Download process complete. Master log has been updated with statuses.")
