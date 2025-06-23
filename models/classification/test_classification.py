import os
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import numpy as np
import random

from models.classification.hybrid_resnet_vit import HybridRestnetVit


class CarClassificationTester:
    """
    Test the trained car classification model on a grid of 8x10 images from the test set.
    """

    def __init__(
        self,
        test_dir,
        class_names,
        model_path=None,
        num_classes=8,
        grid_rows=8,
        grid_cols=10,
        img_size=224,
        embed_dim=768,
        num_heads=12,
        num_layers=6,
        dropout=0.1,
    ):
        self.test_dir = test_dir
        self.class_names = class_names
        self.model_path = model_path
        self.num_classes = num_classes
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.model = self._load_model()
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def _load_model(self):
        if self.model_path is None:
            results_dir = "models/classification/results"
            model_files = [f for f in os.listdir(results_dir) if f.endswith(".pth")]
            if not model_files:
                raise FileNotFoundError("No model checkpoint found in models/results.")
            model_files.sort()
            self.model_path = os.path.join(results_dir, model_files[-1])

        model = HybridRestnetVit(
            image_size=self.img_size,
            num_classes=self.num_classes,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )
        model.load_state_dict(torch.load(self.model_path, map_location="cpu"))
        model.eval()
        return model

    def get_test_images(self, max_images=None):
        if max_images is None:
            max_images = self.grid_rows * self.grid_cols

        image_paths = []
        labels = []

        for class_name in sorted(os.listdir(self.test_dir)):
            class_folder = os.path.join(self.test_dir, class_name)
            if not os.path.isdir(class_folder):
                continue
            for img_name in os.listdir(class_folder):
                if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    image_paths.append(os.path.join(class_folder, img_name))
                    labels.append(class_name)

        combined = list(zip(image_paths, labels))
        random.shuffle(combined)
        image_paths, labels = zip(*combined) if combined else ([], [])
        image_paths = list(image_paths)[:max_images]
        labels = list(labels)[:max_images]
        return image_paths, labels

    def predict(self, image_path):
        img = Image.open(image_path).convert("RGB")
        input_tensor = self.transform(img).unsqueeze(0)

        with torch.no_grad():
            output = self.model(input_tensor)
            pred_idx = output.argmax(dim=1).item()
            pred_label = self.class_names[pred_idx]

        return img, pred_label

    def plot_predictions(self):
        image_paths, true_labels = self.get_test_images()
        fig, axes = plt.subplots(
            self.grid_rows,
            self.grid_cols,
            figsize=(2 * self.grid_cols, 2 * self.grid_rows),
        )

        correct = 0
        total = len(image_paths)

        for idx, (img_path, true_label) in enumerate(zip(image_paths, true_labels)):
            img, pred_label = self.predict(img_path)
            if pred_label == true_label:
                correct += 1
            row, col = divmod(idx, self.grid_cols)
            ax = axes[row, col]
            ax.imshow(np.array(img))
            ax.set_title(f"T: {true_label}\nP: {pred_label}", fontsize=8)
            ax.axis("off")

        for idx in range(len(image_paths), self.grid_rows * self.grid_cols):
            row, col = divmod(idx, self.grid_cols)
            axes[row, col].axis("off")
        plt.tight_layout()
        plt.show()

        accuracy = correct / total if total > 0 else 0
        print(f"Test Accuracy: {accuracy:.2%} ({correct}/{total})")
