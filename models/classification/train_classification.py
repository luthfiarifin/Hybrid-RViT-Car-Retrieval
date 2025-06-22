import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from tqdm import tqdm
import time

from models.classification.hybrid_resnet_vit import HybridRestnetVit

import albumentations as A
from albumentations.pytorch import ToTensorV2


class CarAugmentation:
    """
    CarAugmentation: Albumentations-based aggressive augmentation pipeline for car images.
    Includes perspective, rotation, blur, coarse dropout, and color jitter.
    """

    def __init__(self):
        self.aug = A.Compose(
            [
                A.RandomResizedCrop(
                    224, 224, scale=(0.8, 1.0), ratio=(0.75, 1.33), p=1.0
                ),
                A.HorizontalFlip(p=0.5),
                A.RandomPerspective(distortion_scale=0.3, p=0.5),
                A.Rotate(limit=15, p=0.7),
                A.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0, p=0.8
                ),
                A.GaussianBlur(blur_limit=(5, 9), sigma_limit=(0.1, 5.0), p=0.5),
                A.CoarseDropout(
                    max_holes=8,
                    max_height=22,
                    max_width=22,
                    min_holes=1,
                    min_height=10,
                    min_width=10,
                    fill_value=0,
                    p=0.2,
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

    def __call__(self, img):
        import numpy as np

        img = np.array(img)
        return self.aug(image=img)["image"]


class CarClassifierTrainer:
    """
    CarClassifierTrainer: A class to train a hybrid ResNet-ViT model for car classification.
    """

    def __init__(
        self,
        train_dir="dataset/train",
        val_dir="dataset/val",
        num_classes=8,
        embed_dim=768,
        num_heads=12,
        num_layers=6,
        dropout=0.1,
        learning_rate=1e-4,
        batch_size=32,
        num_epochs=25,
        device=None,
        result_path="carvit_model.pth",
        use_weighted_loss=True,
        use_class_balancing=False,
        num_workers=0,
    ):
        self.DEVICE = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.DEVICE}")
        self.NUM_CLASSES = num_classes
        self.EMBED_DIM = embed_dim
        self.NUM_HEADS = num_heads
        self.NUM_LAYERS = num_layers
        self.DROPOUT = dropout
        self.LEARNING_RATE = learning_rate
        self.BATCH_SIZE = batch_size
        self.NUM_EPOCHS = num_epochs
        self.TRAIN_DIR = train_dir
        self.VAL_DIR = val_dir
        self.RESULT_PATH = result_path
        self.USE_WEIGHTED_LOSS = use_weighted_loss
        self.USE_CLASS_BALANCING = use_class_balancing
        self.NUM_WORKERS = num_workers

        # Initialize tracking variables
        self.train_losses = []
        self.val_accuracies = []
        self.val_losses = []
        self.epoch_times = []
        self.learning_rates = []

        self._prepare_data()
        self._init_model()

    def _prepare_data(self):
        # Data transformations
        self.train_transforms = CarAugmentation()
        self.val_transforms = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.train_dataset = ImageFolder(
            root=self.TRAIN_DIR, transform=self.train_transforms
        )
        self.val_dataset = ImageFolder(root=self.VAL_DIR, transform=self.val_transforms)

        # Create samplers for class balancing if enabled
        if self.USE_CLASS_BALANCING:
            from torch.utils.data import WeightedRandomSampler

            # Calculate class weights for sampling
            class_counts = torch.zeros(self.NUM_CLASSES)
            for _, target in self.train_dataset:
                class_counts[target] += 1

            # Inverse frequency weighting
            class_weights = 1.0 / class_counts
            sample_weights = [class_weights[target] for _, target in self.train_dataset]

            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )

            self.train_loader = DataLoader(
                dataset=self.train_dataset,
                batch_size=self.BATCH_SIZE,
                sampler=sampler,
                num_workers=self.NUM_WORKERS,
            )
            print("Using WeightedRandomSampler for class balancing")
        else:
            self.train_loader = DataLoader(
                dataset=self.train_dataset, batch_size=self.BATCH_SIZE, shuffle=True
            )

        self.val_loader = DataLoader(
            dataset=self.val_dataset, batch_size=self.BATCH_SIZE, shuffle=False
        )

    def _init_model(self):
        self.model = HybridRestnetVit(
            num_classes=self.NUM_CLASSES,
            embed_dim=self.EMBED_DIM,
            num_heads=self.NUM_HEADS,
            num_layers=self.NUM_LAYERS,
            dropout=self.DROPOUT,
        ).to(self.DEVICE)

        # Initialize loss function with optional class weighting
        if self.USE_WEIGHTED_LOSS:
            # Calculate class weights based on inverse frequency
            class_counts = torch.zeros(self.NUM_CLASSES)
            for _, target in self.train_dataset:
                class_counts[target] += 1

            # Inverse frequency weighting
            total_samples = len(self.train_dataset)
            class_weights = total_samples / (self.NUM_CLASSES * class_counts)
            class_weights = class_weights.to(self.DEVICE)

            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
            print(f"Using weighted CrossEntropyLoss with weights: {class_weights}")
        else:
            self.loss_fn = nn.CrossEntropyLoss()
            print("Using standard CrossEntropyLoss")

        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.LEARNING_RATE)

        # Print class distribution info
        self._print_class_distribution()

    def _print_class_distribution(self):
        """Print class distribution and imbalance statistics"""
        class_counts = torch.zeros(self.NUM_CLASSES)
        for _, target in self.train_dataset:
            class_counts[target] += 1

        print(f"\n📊 Class Distribution Analysis:")
        print(f"{'Class':<12} {'Count':<8} {'Percentage':<10} {'Imbalance Ratio':<15}")
        print("-" * 50)

        max_count = class_counts.max().item()
        for i, (class_name, count) in enumerate(
            zip(self.train_dataset.classes, class_counts)
        ):
            percentage = (count / len(self.train_dataset)) * 100
            imbalance_ratio = max_count / count.item()
            print(
                f"{class_name:<12} {int(count):<8} {percentage:<10.1f}% {imbalance_ratio:<15.2f}x"
            )

        # Calculate overall imbalance metrics
        min_count = class_counts.min().item()
        imbalance_factor = max_count / min_count
        print(
            f"\n📈 Imbalance Factor: {imbalance_factor:.2f}x (Most frequent / Least frequent)"
        )

        if imbalance_factor > 5:
            print(
                "⚠️  High imbalance detected! Consider using weighted loss or resampling."
            )
        elif imbalance_factor > 2:
            print("⚠️  Moderate imbalance detected. Weighted loss recommended.")
        else:
            print("✅ Relatively balanced dataset.")

    def train_one_epoch(self):
        self.model.train()
        loop = tqdm(self.train_loader, leave=True)
        running_loss = 0.0
        for batch_idx, (data, targets) in enumerate(loop):
            data, targets = data.to(self.DEVICE), targets.to(self.DEVICE)
            scores = self.model(data)
            loss = self.loss_fn(scores, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        return running_loss / len(self.train_loader)

    def check_accuracy(self):
        self.model.eval()
        num_correct = 0
        num_samples = 0
        running_val_loss = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.DEVICE), y.to(self.DEVICE)
                scores = self.model(x)
                loss = self.loss_fn(scores, y)
                running_val_loss += loss.item()

                _, predictions = scores.max(1)
                num_correct += (predictions == y).sum()
                num_samples += predictions.size(0)

                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(y.cpu().numpy())

        accuracy = (num_correct / num_samples) * 100
        avg_val_loss = running_val_loss / len(self.val_loader)

        print(f"Validation accuracy: {accuracy:.2f}%")
        print(f"Validation loss: {avg_val_loss:.4f}")

        return accuracy, avg_val_loss, all_predictions, all_targets

    def train(self):
        best_accuracy = 0.0
        best_model_state = None

        print("Starting training with detailed tracking...")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(
            f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}"
        )

        for epoch in range(self.NUM_EPOCHS):
            epoch_start_time = time.time()
            print(f"\n{'='*50}")
            print(f"Epoch {epoch+1}/{self.NUM_EPOCHS}")
            print(f"{'='*50}")

            # Training phase
            train_loss = self.train_one_epoch()

            # Validation phase
            val_accuracy, val_loss, predictions, targets = self.check_accuracy()

            # Track metrics
            epoch_time = time.time() - epoch_start_time
            current_lr = self.optimizer.param_groups[0]["lr"]

            self.train_losses.append(train_loss)
            self.val_accuracies.append(val_accuracy.item())
            self.val_losses.append(val_loss)
            self.epoch_times.append(epoch_time)
            self.learning_rates.append(current_lr)

            print(f"Time: {epoch_time:.2f}s | LR: {current_lr:.2e}")

            # Save best model
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_model_state = self.model.state_dict().copy()
                print(f"🎉 New best accuracy: {best_accuracy:.2f}%")

        print(f"\n{'='*50}")
        print("Training completed!")
        print(f"Best validation accuracy: {best_accuracy:.2f}%")

        # Save best model
        if best_model_state:
            torch.save(best_model_state, self.RESULT_PATH)
            print(f"Best model saved to {self.RESULT_PATH}")

        return {
            "train_losses": self.train_losses,
            "val_accuracies": self.val_accuracies,
            "val_losses": self.val_losses,
            "epoch_times": self.epoch_times,
            "learning_rates": self.learning_rates,
            "best_accuracy": best_accuracy,
            "final_predictions": predictions,
            "final_targets": targets,
        }
