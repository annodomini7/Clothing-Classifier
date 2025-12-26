import json
import os

import hydra
import mlflow
import pytorch_lightning as pl
import torch
from dataset import ClothingDataModule
from hydra.utils import get_original_cwd
from model import ClothingClassifier
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, MLFlowLogger, TensorBoardLogger
from utils import download_data, export_to_onnx, get_git_commit_id


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(42)

    original_cwd = get_original_cwd()

    data_dir = os.path.join(original_cwd, cfg.data.data_dir)
    download_data(data_dir)
    mlflow_logger = MLFlowLogger(
        experiment_name="clothing-classification",
        tracking_uri=cfg.mlflow.tracking_uri,
        tags={"git_commit": get_git_commit_id()},
    )

    mlflow_logger.log_hyperparams(
        {
            "batch_size": cfg.data.batch_size,
            "num_workers": cfg.data.num_workers,
            "image_size": cfg.data.image_size,
            "model": cfg.model.model_name,
            "optimizer": cfg.optimizer._target_,
            "learning_rate": cfg.optimizer.lr,
            "max_epochs": cfg.training.max_epochs,
        }
    )

    checkpoint_dir = os.path.join(original_cwd, cfg.checkpoint.dirpath)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best-model-{epoch:02d}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    data_module = ClothingDataModule(
        data_dir=data_dir,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        image_size=cfg.data.image_size,
        transform_mean=cfg.data.mean,
        transform_std=cfg.data.std,
    )
    data_module.setup()

    model = ClothingClassifier(
        num_classes=data_module.get_num_classes(),
        model_name=cfg.model.model_name,
        pretrained=cfg.model.pretrained,
        freeze_layers=cfg.model.get("freeze_layers", 0),
        learning_rate=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.get("weight_decay", 1e-4),
    )

    plots_dir = os.path.join(original_cwd, "plots")
    tb_logger = TensorBoardLogger(save_dir=plots_dir, name="tb_logs")
    csv_logger = CSVLogger(save_dir=plots_dir, name="csv_logs")

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        logger=[tb_logger, csv_logger, mlflow_logger],
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)

    best_model_path = checkpoint_callback.best_model_path
    if best_model_path and os.path.isfile(best_model_path):
        print(f"Loading best model from: {best_model_path}")
        best_model = ClothingClassifier.load_from_checkpoint(best_model_path)
    else:
        print("No checkpoint found, using current model")
        best_model = model

    onnx_path = os.path.join(original_cwd, cfg.export.onnx_path)
    export_to_onnx(
        model=best_model,
        output_path=onnx_path,
    )

    weights_path = os.path.join(checkpoint_dir, "best_model_weights.pth")
    torch.save(best_model.state_dict(), weights_path)

    class_to_name_path = os.path.join(checkpoint_dir, "class_to_name.json")
    with open(class_to_name_path, "w") as f:
        json.dump(data_module.get_class_to_name(), f)

    class_to_cat_path = os.path.join(checkpoint_dir, "class_to_category.json")
    with open(class_to_cat_path, "w") as f:
        json.dump(data_module.get_class_to_category(), f)

    mlflow.pytorch.log_model(best_model, "model")


if __name__ == "__main__":
    main()
