import os
import shutil
import glob
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class DatasetSplitter:
    """
    A class to split an image dataset into training, validation, and test sets
    while maintaining class distribution (stratified split).
    """

    def __init__(self, source_dir, output_dir, train_ratio=0.8, val_ratio=0.1):
        self.source_dir = source_dir
        self.output_dir = output_dir

        if train_ratio + val_ratio >= 1.0:
            raise ValueError(
                "The sum of train_ratio and val_ratio must be less than 1."
            )

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1.0 - train_ratio - val_ratio

        self.image_paths = []
        self.labels = []

    def _discover_data(self):
        logging.info(f"Scanning source directory: {self.source_dir}")
        image_extensions = ("*.jpg", "*.jpeg", "*.png")
        for ext in image_extensions:
            self.image_paths.extend(
                glob.glob(os.path.join(self.source_dir, "**", ext), recursive=True)
            )

        if not self.image_paths:
            logging.error(
                f"No images found in the source directory. Please check the path."
            )
            return False

        self.labels = [os.path.basename(os.path.dirname(p)) for p in self.image_paths]
        logging.info(
            f"Found {len(self.image_paths)} images belonging to {len(set(self.labels))} classes."
        )
        return True

    def _copy_files(self, files, labels, split_name):
        logging.info(f"Copying files for the '{split_name}' set...")
        for src_path, label in tqdm(
            zip(files, labels), total=len(files), desc=f"Copying {split_name} files"
        ):
            if not os.path.exists(src_path):
                logging.warning(f"Source file not found, skipping: {src_path}")
                continue

            destination_dir = os.path.join(self.output_dir, split_name, label)
            os.makedirs(destination_dir, exist_ok=True)
            shutil.copy(src_path, destination_dir)

    def split_and_copy(self):
        if not self._discover_data():
            return

        if os.path.exists(self.output_dir):
            logging.warning(
                f"Output directory '{self.output_dir}' already exists. It will be deleted."
            )
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        logging.info(
            f"Performing stratified split: {self.train_ratio*100:.0f}% train, {self.val_ratio*100:.0f}% val, {self.test_ratio*100:.0f}% test."
        )

        X_train, X_temp, y_train, y_temp = train_test_split(
            self.image_paths,
            self.labels,
            test_size=(self.val_ratio + self.test_ratio),
            stratify=self.labels,
            random_state=42,
        )

        relative_test_size = self.test_ratio / (self.val_ratio + self.test_ratio)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=relative_test_size,
            stratify=y_temp,
            random_state=42,
        )

        logging.info("Splitting complete. Starting file copy operation.")

        self._copy_files(X_train, y_train, "train")
        self._copy_files(X_val, y_val, "val")
        self._copy_files(X_test, y_test, "test")

        print("\n" + "=" * 50)
        logging.info("Dataset splitting and copying finished successfully!")
        print("=" * 50)
        print(f"Total images in source: {len(self.image_paths)}")
        print("-" * 20)
        print(f"Training set: {len(X_train)} images")
        print(f"Validation set: {len(X_val)} images")
        print(f"Test set: {len(X_test)} images")
        print(f"Total split: {len(X_train) + len(X_val) + len(X_test)} images")
        print("=" * 50)
