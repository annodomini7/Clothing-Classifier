import io
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


def main(
    image: str,
    url: str = "localhost:8900",
    model_name: str = "ensemble_onnx",
    labels_path: str = "../checkpoints/class_to_name.json",
    top_k: int = 5,
) -> None:
    client = httpclient.InferenceServerClient(url=url)

    if not client.is_server_live():
        print(f"Error: Triton server at {url} is not available")
        return

    labels = {}
    if Path(labels_path).exists():
        labels = load_labels(labels_path)

    pil_image = Image.open(image).convert("RGB")
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG")
    image_bytes = buffer.getvalue()
    image_input = np.frombuffer(image_bytes, dtype=np.uint8)
    batched_input = image_input.reshape(1, -1)

    inputs = [httpclient.InferInput("raw_image", batched_input.shape, "UINT8")]
    inputs[0].set_data_from_numpy(batched_input)

    outputs = [httpclient.InferRequestedOutput("predictions")]

    response = client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
    logits = response.as_numpy("predictions")[0]

    probs = softmax(logits)
    top_indices = np.argsort(probs)[::-1][:top_k]

    print(f"\nImage: {image}")
    print(f"Top-{top_k} predictions:")
    for idx in top_indices:
        label = labels.get(str(idx), f"class_{idx}")
        print(f"  {label}: {probs[idx]:.2%}")


if __name__ == "__main__":
    fire.Fire(main)
