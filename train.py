"""Train a dense classifier on the MNIST digit dataset."""

from tensorflow import keras


def build_model() -> keras.Model:
    return keras.Sequential(
        [
            keras.layers.Input(shape=(784,)),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(10, activation="softmax"),
        ]
    )


def main() -> None:
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = (x_train.reshape(-1, 784).astype("float32") / 255.0)
    x_test = (x_test.reshape(-1, 784).astype("float32") / 255.0)

    model = build_model()
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        x_train,
        y_train,
        epochs=5,
        batch_size=128,
        validation_split=0.1,
        verbose=1,
    )

    _, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {accuracy:.4f}")

    model.save("mnist_model.keras")


if __name__ == "__main__":
    main()
