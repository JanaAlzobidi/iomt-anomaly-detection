"""
inference.py - inference pipeline for the best model of the project (Sparse AE).

Loads the saved Sparse Autoencoder and its metadata (threshold + feature names),
and exposes predict(X), which tells for each row of traffic whether it looks like an attack.

Usage:

    import sys
    sys.path.insert(0, "iomt-anomaly-detection/inference")
    from inference import predict
    results = predict(X_test[:5])

X must be prepared rows: 40 features, same order and scaling as data/test.npz.
"""

import json
import os

import numpy as np
import keras

# paths are relative to this file, so it works from any working directory
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(REPO_DIR, "models", "sparse_ae", "sparse_ae.keras")
META_PATH = os.path.join(REPO_DIR, "models", "sparse_ae", "sparse_ae_metadata.json")

_model = None
_meta = None


def load_model():
    """Loads the model and its metadata once, then reuses them on every call."""
    global _model, _meta
    if _model is None:
        _model = keras.models.load_model(MODEL_PATH)
        with open(META_PATH, encoding="utf-8") as f:
            _meta = json.load(f)
    return _model, _meta


def predict(X):
    """
    X: one row (shape (40,)) or many rows (shape (n, 40)), prepared like X_test.

    Returns a list with one dict per row:
        score        reconstruction error (higher = more abnormal)
        is_anomaly   True if the score is above the saved threshold (= attack)
        top_features the 3 features with the largest reconstruction error
    """
    model, meta = load_model()
    threshold = meta["threshold"]
    feature_names = np.array(meta["features"])

    X = np.asarray(X, dtype="float32")
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.shape[1] != len(feature_names):
        raise ValueError(f"expected {len(feature_names)} features, got {X.shape[1]}")

    reconstruction = model.predict(X, batch_size=4096, verbose=0)
    errors = np.square(X - reconstruction)          # error per feature
    scores = errors.mean(axis=1)                    # error per row = anomaly score
    top = np.argsort(-errors, axis=1)[:, :3]        # 3 features with the biggest error

    return [{"score": float(s),
             "is_anomaly": bool(s > threshold),
             "top_features": feature_names[t].tolist()}
            for s, t in zip(scores, top)]


if __name__ == "__main__":
    # quick check: python inference/inference.py
    test = np.load(os.path.join(REPO_DIR, "data", "test.npz"), allow_pickle=True)
    X_test, attack_test = test["X"], test["attack_type"]

    _, meta = load_model()
    threshold = meta["threshold"]
    print(f"Sparse AE loaded | threshold {threshold:.6g}\n")

    rng = np.random.default_rng(42)
    for i in rng.choice(len(X_test), 5, replace=False):
        r = predict(X_test[i])[0]
        true_type = str(attack_test[i])
        decision = "ATTACK" if r["is_anomaly"] else "normal"
        score = r["score"]
        features = ", ".join(r["top_features"])
        print(f"row {i:>5} | true: {true_type:<24} | predicted: {decision:<6} | "
              f"score {score:.3g} | top features: {features}")
