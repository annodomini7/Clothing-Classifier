import json
from pathlib import Path

import fire
import numpy as np
import onnxruntime as ort
from PIL import Image


def preprocess_image(image_path: str, image_size: int = 224) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = Image.open(image_path).convert("RGB")
    image = image.resize((image_size, image_size))
    image = np.array(image, dtype=np.float32) / 255.0
    image = (image - mean) / std
    image = image.transpose(2, 0, 1)
    return image.astype(np.float32)


def load_labels(labels_path: str) -> dict:
    with open(labels_path, "r") as f:
        return json.load(f)


def softmax(x: np.ndarray) -> np.ndarray:
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()


def infer_single(
    model_path: str,
    image: str,
    labels_path: str = "checkpoints/class_to_name.json",
    top_k: int = 5,
) -> None:
    ort_session = ort.InferenceSession(model_path)

    labels = {}
    if Path(labels_path).exists():
        labels = load_labels(labels_path)

    input_data = preprocess_image(image)
    input_data = np.expand_dims(input_data, axis=0)

    input_name = ort_session.get_inputs()[0].name
    outputs = ort_session.run(None, {input_name: input_data})

    logits = outputs[0][0]
    probs = softmax(logits)
    top_indices = np.argsort(probs)[::-1][:top_k]

    print(f"\nImage: {image}")
    print(f"Top-{top_k} predictions:")
    for idx in top_indices:
        label = labels.get(str(idx), f"class_{idx}")
        print(f"  {label}: {probs[idx]:.2%}")


def main(
    model_path: str = "../triton/export/onnx/model.onnx",
    image: str = None,
    labels_path: str = "../checkpoints/class_to_name.json",
    top_k: int = 5,
) -> None:
    if image:
        infer_single(model_path, image, labels_path, top_k)
    else:
        print("Error in arguments")


if __name__ == "__main__":
    fire.Fire(main)
