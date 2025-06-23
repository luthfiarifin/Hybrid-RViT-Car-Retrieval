import os
import shutil
import pandas as pd
from PIL import Image
from ultralytics import YOLO
from tqdm import tqdm
import logging
import concurrent.futures

logging.basicConfig(
    level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s"
)


class DatasetValidator:
    """
    A class to validate and crop images in a dataset using YOLOv12 for object detection.
    It checks for minimum resolution, detects car-related content, and crops images accordingly.
    """

    def __init__(
        self,
        yolo_model_path="models/yolo/yolo12m.pt",
        min_resolution=(150, 150),
        max_workers=8,
        master_csv_path="master_scrape_log.csv",
        rejected_dir="rejected_images",
    ):
        self.master_log_path = master_csv_path
        self.rejected_path = rejected_dir

        self.min_width, self.min_height = min_resolution
        self.max_workers = max_workers

        self.car_related_labels = {"car", "truck", "bus"}

        self.model = YOLO(yolo_model_path)
        print("YOLOv12 object detection model loaded.")

    def _process_single_image(self, row_tuple):
        index, row = row_tuple
        image_path = row["image_path"]
        status = "rejected_other"

        if not os.path.exists(image_path):
            return "file_not_found"

        try:
            # Resolution check
            with Image.open(image_path) as img:
                width, height = img.size

            if width < self.min_width or height < self.min_height:
                rejection_path = os.path.join(
                    self.rejected_path, "low_resolution", os.path.basename(image_path)
                )
                shutil.move(image_path, rejection_path)
                return "rejected_low_resolution"

            # Content Check & Cropping (YOLO)
            results = self.model(image_path, verbose=False)

            car_boxes = []
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls)
                    class_name = self.model.names[class_id]
                    if class_name in self.car_related_labels:
                        car_boxes.append(box.xyxy[0].tolist())

            if car_boxes:
                largest_box = max(
                    car_boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1])
                )

                with Image.open(image_path) as img:
                    cropped_image = img.crop(largest_box).convert("RGB")
                    cropped_image.save(image_path, "JPEG", quality=80, optimize=True)

                status = "valid_and_cropped"
            else:
                rejection_path = os.path.join(
                    self.rejected_path, "not_a_car", os.path.basename(image_path)
                )
                shutil.move(image_path, rejection_path)
                status = "rejected_not_a_car"

        except Exception as e:
            logging.error(f"Could not process {image_path}: {e}")
            status = "rejected_processing_error"
            if os.path.exists(image_path):
                rejection_path = os.path.join(
                    self.rejected_path, "processing_error", os.path.basename(image_path)
                )
                shutil.move(image_path, rejection_path)

        return status

    def validate_and_crop_images(self):
        if not os.path.exists(self.master_log_path):
            logging.error(
                f"Master log file not found at {self.master_log_path}. Please run the scraper first."
            )
            return

        # Only process images where download_status == 'success' and (validation_status is null or valid_and_cropped)
        df = pd.read_csv(self.master_log_path)
        df_downloaded = df[
            (df["download_status"] == "success")
            & (
                df["validation_status"].isnull()
                | (df["validation_status"] == "valid_and_cropped")
            )
        ].copy()

        print(f"Found {len(df_downloaded)} images to validate and crop.")

        # Setup rejection folders
        for folder in ["low_resolution", "not_a_car", "processing_error"]:
            os.makedirs(os.path.join(self.rejected_path, folder), exist_ok=True)

        # Use ThreadPoolExecutor to process images concurrently
        statuses = [None] * len(df_downloaded)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            # Create a map of futures
            future_to_index = {
                executor.submit(self._process_single_image, row_tuple): row_tuple[0]
                for row_tuple in df_downloaded.iterrows()
            }

            # Process as they complete with a tqdm progress bar
            for future in tqdm(
                concurrent.futures.as_completed(future_to_index),
                total=len(df_downloaded),
                desc="Validating & Cropping Images",
            ):
                index = future_to_index[future]
                try:
                    status = future.result()
                    df_index_pos = df_downloaded.index.get_loc(index)
                    statuses[df_index_pos] = status
                except Exception as e:
                    logging.error(f"An error occurred for item at index {index}: {e}")
                    df_index_pos = df_downloaded.index.get_loc(index)
                    statuses[df_index_pos] = "executor_failed"

        df_downloaded["validation_status"] = statuses

        df.loc[df_downloaded.index, "validation_status"] = statuses
        df.to_csv(self.master_log_path, index=False)
        print(f"Validation complete. Report saved to {self.master_log_path}")
