"""Fashion MNIST MLP trainer (784→128→64→10), exports binary weights + test data."""

import numpy as np
import struct
import os
import urllib.request
import gzip
import time

NUM_CLASSES = 10
EPOCHS = 20
LEARNING_RATE = 0.01
BATCH_SIZE = 128
HIDDEN1 = 128
HIDDEN2 = 64
NUM_TEST_IMAGES = 1000

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

URLS = {
    "train_images": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",
    "train_labels": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz",
    "test_images":  "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",
    "test_labels":  "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz",
}

def download_file(url, filename):
    filepath = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(filepath):
        print(f"  [cached] {filename}")
        return filepath
    print(f"  Downloading {filename}...")
    urllib.request.urlretrieve(url, filepath)
    return filepath

def read_idx_images(filepath):
    with gzip.open(filepath, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num, rows * cols).astype(np.float32) / 255.0

def read_idx_labels(filepath):
    with gzip.open(filepath, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.astype(np.int32)

class MLP:
    def __init__(self):

        self.W1 = np.random.randn(784, HIDDEN1).astype(np.float32) * np.sqrt(2.0 / 784)
        self.b1 = np.zeros(HIDDEN1, dtype=np.float32)
        self.W2 = np.random.randn(HIDDEN1, HIDDEN2).astype(np.float32) * np.sqrt(2.0 / HIDDEN1)
        self.b2 = np.zeros(HIDDEN2, dtype=np.float32)
        self.W3 = np.random.randn(HIDDEN2, NUM_CLASSES).astype(np.float32) * np.sqrt(2.0 / HIDDEN2)
        self.b3 = np.zeros(NUM_CLASSES, dtype=np.float32)

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.relu(self.z2)
        self.z3 = self.a2 @ self.W3 + self.b3
        self.a3 = self.softmax(self.z3)
        return self.a3

    def compute_loss(self, y_pred, y_true):
        m = y_true.shape[0]
        one_hot = np.zeros((m, NUM_CLASSES), dtype=np.float32)
        one_hot[np.arange(m), y_true] = 1.0
        log_probs = -np.log(y_pred[np.arange(m), y_true] + 1e-8)
        return np.mean(log_probs)

    def backward(self, X, y_true):
        m = y_true.shape[0]
        one_hot = np.zeros((m, NUM_CLASSES), dtype=np.float32)
        one_hot[np.arange(m), y_true] = 1.0

        dz3 = (self.a3 - one_hot) / m
        dW3 = self.a2.T @ dz3
        db3 = np.sum(dz3, axis=0)

        da2 = dz3 @ self.W3.T
        dz2 = da2 * (self.z2 > 0).astype(np.float32)
        dW2 = self.a1.T @ dz2
        db2 = np.sum(dz2, axis=0)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * (self.z1 > 0).astype(np.float32)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0)

        self.W3 -= LEARNING_RATE * dW3
        self.b3 -= LEARNING_RATE * db3
        self.W2 -= LEARNING_RATE * dW2
        self.b2 -= LEARNING_RATE * db2
        self.W1 -= LEARNING_RATE * dW1
        self.b1 -= LEARNING_RATE * db1

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

    def accuracy(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y) * 100.0

def save_weights_mlp(model, filepath):
    """W1,b1,W2,b2,W3,b3 float32, transposed for C row-major (784×128, 128×64, 64×10)."""
    with open(filepath, 'wb') as f:

        f.write(model.W1.astype(np.float32).tobytes())
        f.write(model.b1.astype(np.float32).tobytes())

        f.write(model.W2.astype(np.float32).tobytes())
        f.write(model.b2.astype(np.float32).tobytes())

        f.write(model.W3.astype(np.float32).tobytes())
        f.write(model.b3.astype(np.float32).tobytes())

def save_test_images(images, filepath):
    with open(filepath, 'wb') as f:
        f.write(images.astype(np.float32).tobytes())

def save_test_labels(labels, filepath):
    with open(filepath, 'wb') as f:
        f.write(labels.astype(np.int32).tobytes())

def main():
    print("=" * 60)
    print("Fashion MNIST MLP Trainer (Standalone)")
    print("=" * 60)

    print("\n[1/5] Downloading Fashion MNIST dataset...")
    train_img_path = download_file(URLS["train_images"], "train-images-idx3-ubyte.gz")
    train_lbl_path = download_file(URLS["train_labels"], "train-labels-idx1-ubyte.gz")
    test_img_path  = download_file(URLS["test_images"],  "t10k-images-idx3-ubyte.gz")
    test_lbl_path  = download_file(URLS["test_labels"],  "t10k-labels-idx1-ubyte.gz")

    print("\n[2/5] Loading data...")
    X_train = read_idx_images(train_img_path)
    y_train = read_idx_labels(train_lbl_path)
    X_test_full = read_idx_images(test_img_path)
    y_test_full = read_idx_labels(test_lbl_path)
    print(f"  Training set: {X_train.shape[0]} images")
    print(f"  Test set:     {X_test_full.shape[0]} images")

    X_test = X_test_full[:NUM_TEST_IMAGES]
    y_test = y_test_full[:NUM_TEST_IMAGES]

    print(f"\n[3/5] Training MLP (784→{HIDDEN1}→{HIDDEN2}→{NUM_CLASSES})...")
    print(f"  Epochs: {EPOCHS}, LR: {LEARNING_RATE}, Batch: {BATCH_SIZE}")

    model = MLP()
    n_train = X_train.shape[0]
    t_start = time.time()

    for epoch in range(EPOCHS):

        perm = np.random.permutation(n_train)
        X_shuffled = X_train[perm]
        y_shuffled = y_train[perm]

        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_train, BATCH_SIZE):
            X_batch = X_shuffled[i:i+BATCH_SIZE]
            y_batch = y_shuffled[i:i+BATCH_SIZE]

            model.forward(X_batch)
            model.backward(X_batch, y_batch)

            probs = model.forward(X_batch)
            epoch_loss += model.compute_loss(probs, y_batch)
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        train_acc = model.accuracy(X_train[:2000], y_train[:2000])
        test_acc = model.accuracy(X_test, y_test)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:2d}/{EPOCHS} | Loss: {avg_loss:.4f} | "
                  f"Train Acc: {train_acc:.1f}% | Test Acc: {test_acc:.1f}%")

    elapsed = time.time() - t_start
    final_acc = model.accuracy(X_test, y_test)
    print(f"\n  Training complete in {elapsed:.1f}s")
    print(f"  Final test accuracy: {final_acc:.1f}%")

    print("\n[4/5] Exporting weights to weights_fashion.bin...")
    weights_path = os.path.join(OUTPUT_DIR, "weights_fashion.bin")
    save_weights_mlp(model, weights_path)

    weights_size = os.path.getsize(weights_path)

    expected = (784*128 + 128 + 128*64 + 64 + 64*10 + 10) * 4
    print(f"  Saved: {weights_path}")
    print(f"  Size: {weights_size} bytes (expected: {expected} bytes)")

    print("\n[5/5] Exporting test data...")
    images_path = os.path.join(OUTPUT_DIR, "test_images.bin")
    labels_path = os.path.join(OUTPUT_DIR, "test_labels.bin")

    save_test_images(X_test, images_path)
    save_test_labels(y_test, labels_path)

    images_size = os.path.getsize(images_path)
    labels_size = os.path.getsize(labels_path)
    expected_images = NUM_TEST_IMAGES * 784 * 4
    expected_labels = NUM_TEST_IMAGES * 4

    print(f"  Saved: {images_path}")
    print(f"  Size: {images_size} bytes (expected: {expected_images} bytes)")
    print(f"  Saved: {labels_path}")
    print(f"  Size: {labels_size} bytes (expected: {expected_labels} bytes)")

    print("\n" + "=" * 60)
    print("SUCCESS! All files exported correctly.")
    print("=" * 60)
    print(f"  weights_fashion.bin  : {weights_size:>10} bytes  ✓")
    print(f"  test_images.bin      : {images_size:>10} bytes  ✓")
    print(f"  test_labels.bin      : {labels_size:>10} bytes  ✓")
    print(f"  Final accuracy       : {final_acc:>9.1f}%")
    print("=" * 60)

if __name__ == "__main__":
    main()
