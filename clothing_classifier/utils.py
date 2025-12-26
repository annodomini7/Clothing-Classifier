import os
import subprocess

import onnxruntime as ort
import torch
from dvc.repo import Repo
from kaggle.api.kaggle_api_extended import KaggleApi


def get_git_commit_id():
    try:
        cmd = ["git", "rev-parse", "HEAD"]
        return subprocess.check_output(cmd).decode().strip()
    except Exception:
        return "unknown"


def download_data(data_dir: str) -> None:
    csv_path = os.path.join(data_dir, "styles.csv")
    images_dir = os.path.join(data_dir, "images")

    if os.path.exists(csv_path) and os.path.exists(images_dir):
        print(f"Data already exists in {data_dir}")
        return

    try:
        repo = Repo(".")
        repo.pull()
        print(f"Data successfully pulled from DVC to {data_dir}")
        return
    except Exception as e:
        print(f"DVC pull failed: {e}. Downloading from Kaggle...")

    api = KaggleApi()
    dataset = "paramaggarwal/fashion-product-images-small"

    os.makedirs(data_dir, exist_ok=True)

    try:
        api.authenticate()
        print("Kaggle authentication successful!")
    except Exception as e:
        print(f"Kaggle authentication failed: {str(e)}")
        raise

    api.dataset_download_files(dataset, path=data_dir, unzip=True, quiet=False)
    print(f"Dataset downloaded to {data_dir}")


def export_to_onnx(
    model,
    output_path: str,
    input_size: int = 224,
    device: str = "cpu",
):
    model.eval().to(device)

    dummy_input = torch.randn(1, 3, input_size, input_size).to(device)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    torch.onnx.export(
        model.model,
        dummy_input,
        output_path,
        export_params=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"ONNX model saved to {output_path}")

    ort_session = ort.InferenceSession(output_path)
    onnx_input = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    _ = ort_session.run(None, onnx_input)[0]
    print("ONNX model validated successfully")
