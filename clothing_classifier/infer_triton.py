import json
from pathlib import Path

import fire
import numpy as np
import tritonclient.http as httpclient
from PIL import Image


def load_labels(labels_path: str) -> dict:
    with open(labels_path, "r") as f:
        return json.load(f)


def softmax(x: np.ndarray) -> np.ndarray:
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()


def preprocess_image(image_path: str, image_size: int = 224) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = Image.open(image_path).convert("RGB")
    image = image.resize((image_size, image_size))
    image = np.array(image, dtype=np.float32) / 255.0
    image = (image - mean) / std
    image = image.transpose(2, 0, 1)
    return image.astype(np.float32)


def main(
    image: str,
    url: str = "localhost:8900",
    model_name: str = "onnx",
    labels_path: str = "checkpoints/class_to_name.json",
    top_k: int = 5,
) -> None:
    client = httpclient.InferenceServerClient(url=url)

    if not client.is_server_live():
        print(f"Error: Triton server at {url} is not available")
        return

    labels = {}
    if Path(labels_path).exists():
        labels = load_labels(labels_path)

    input_data = preprocess_image(image)
    input_data = np.expand_dims(input_data, axis=0)

    inputs = [httpclient.InferInput("input", input_data.shape, "FP32")]
    inputs[0].set_data_from_numpy(input_data)

    outputs = [httpclient.InferRequestedOutput("output")]

    response = client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
    logits = response.as_numpy("output")[0]

    probs = softmax(logits)
    top_indices = np.argsort(probs)[::-1][:top_k]

    print(f"\nImage: {image}")
    print(f"Top-{top_k} predictions:")
    for idx in top_indices:
        label = labels.get(str(idx), f"class_{idx}")
        print(f"  {label}: {probs[idx]:.2%}")


if __name__ == "__main__":
    fire.Fire(main)
