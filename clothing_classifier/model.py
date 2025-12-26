import logging

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from torchvision import models

logger = logging.getLogger(__name__)


class ClothingClassifier(pl.LightningModule):
    def __init__(
        self,
        num_classes: int,
        model_name: str = "resnet50",
        pretrained: bool = True,
        freeze_layers: int = 40,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_classes = num_classes

        self.model = self._build_model(
            model_name, num_classes, pretrained, freeze_layers
        )

        self.criterion = nn.CrossEntropyLoss()

        self.train_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.val_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.val_top5_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes, top_k=5
        )
        self.val_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )

    def _build_model(
        self,
        model_name: str,
        num_classes: int,
        pretrained: bool,
        freeze_layers: int,
    ) -> nn.Module:
        weights = "IMAGENET1K_V1" if pretrained else None

        if model_name == "resnet50":
            model = models.resnet50(weights=weights)
            in_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes),
            )
        elif model_name == "resnet34":
            model = models.resnet34(weights=weights)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        if freeze_layers > 0:
            layers = list(model.children())
            for layer in layers[:freeze_layers]:
                for param in layer.parameters():
                    param.requires_grad = False
            logger.info(f"Froze first {freeze_layers} layers")

        return model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)

        preds = torch.argmax(logits, dim=1)
        acc = self.train_accuracy(preds, labels)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)

        preds = torch.argmax(logits, dim=1)

        self.val_accuracy(preds, labels)
        self.val_top5_accuracy(logits, labels)
        self.val_f1(preds, labels)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "val_acc",
            self.val_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val_top5_acc",
            self.val_top5_accuracy,
            on_step=False,
            on_epoch=True,
        )
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)

        preds = torch.argmax(logits, dim=1)

        self.val_accuracy(preds, labels)
        self.val_top5_accuracy(logits, labels)
        self.val_f1(preds, labels)

        self.log("test_loss", loss)
        self.log("test_acc", self.val_accuracy)
        self.log("test_top5_acc", self.val_top5_accuracy)
        self.log("test_f1", self.val_f1)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
            },
        }
