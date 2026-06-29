# MNIST Digit Classifier

Practice project for handwritten digit recognition with Keras on the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Train

```bash
python train.py
```

The script downloads MNIST automatically, trains a small fully connected network for five epochs, prints test accuracy, and saves `mnist_model.keras`.

## Model

- Input: flattened 28×28 grayscale images (784 features)
- Hidden: 128 ReLU units with 20% dropout
- Output: 10-class softmax
