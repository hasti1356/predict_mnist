"""Evaluate a saved MNIST model on the test split."""

import sys
from pathlib import Path

from tensorflow import keras


def main() -> None:
    model_path = Path("mnist_model.keras")
    if not model_path.exists():
        print("Run train.py first to create mnist_model.keras", file=sys.stderr)
        sys.exit(1)

    model = keras.models.load_model(model_path)
    (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_test = x_test.reshape(-1, 784).astype("float32") / 255.0

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
