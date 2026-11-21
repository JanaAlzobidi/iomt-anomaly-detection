# Vanilla Autoencoder

A fully connected autoencoder trained on benign traffic only. Each row is compressed into a bottleneck and reconstructed; the reconstruction error (MSE) is the anomaly score. This is the base architecture of the project: the Denoising and Sparse autoencoders reuse it and add one change each.

## Files

| File | Content |
|---|---|
| `vanilla_ae.ipynb` | Full pipeline: model design, tuning, training, evaluation, comparison with PCA, save/load, inference |
| `vanilla_ae.keras` | Trained model |
| `vanilla_ae_metadata.json` | Threshold, feature order and hyperparameters, needed at inference time |

## Architecture

`40 → Dense(48, ReLU) → Dense(32, linear) "latent" → Dense(48, ReLU) → Dense(40, linear)`

| Hyperparameter | Value |
|---|---|
| Latent size | 32 |
| Hidden width | 48 (one hidden layer per side) |
| Latent activation | linear |
| Dropout | 0.0 |
| L2 | 1e-6 |
| Learning rate | 0.0005 (Adam) |

## Tuning

Keras Tuner, Bayesian optimization, 25 trials of up to 25 epochs. The objective is `val_mse`, the reconstruction error on benign validation data, so no attack data is used to choose the architecture.

The six best trials all use no dropout, and every trial with dropout 0.1 ranks in the bottom half: stronger regularization only increased the reconstruction error. The model is small (about 7,000 weights) compared with 150,000 training rows, so it underfits rather than overfits. A wider search with 100 trials found a model with lower validation error but the same F1 (0.886), so 25 trials were kept.

## Training

Up to 200 epochs, batch size 512, Early Stopping on `val_mse` (patience 10, best weights restored) and the learning rate halved when validation is stuck for 4 epochs. The validation error was still improving at epoch 200.

## Results

Threshold: 99th percentile of benign validation scores.

| Metric | PCA baseline | Vanilla AE |
|---|---|---|
| ROC-AUC | 0.938 | **0.955** |
| PR-AUC | 0.835 | **0.903** |
| F1 | 0.761 | **0.888** |
| Precision | **0.959** | 0.934 |
| Recall | 0.631 | **0.847** |
| False alarm rate | **0.37%** | 0.84% |

The largest gain is on the four MQTT flood attacks: PCA detects 6% to 29% of them, the autoencoder 100%. These attacks are not separable from normal traffic by a linear model.

The autoencoder still misses most ARP Spoofing (2%) and MQTT Malformed Data (24%) rows, and about half of Recon VulScan and PingSweep. In the 40 available features these attacks look like normal traffic, and none of 10 alternative training settings or 5 alternative scoring methods improved them.

## Inference

```python
import json, numpy as np, keras
from evaluate import per_feature_errors

model = keras.models.load_model("models/dense_ae/vanilla_ae.keras")
meta = json.load(open("models/dense_ae/vanilla_ae_metadata.json"))
THRESHOLD = meta["threshold"]
feature_names = np.array(meta["features"])

def predict(X):
    err = per_feature_errors(model, np.asarray(X, dtype="float32"))
    scores = err.mean(axis=1)
    top = np.argsort(-err, axis=1)[:, :3]
    return [{"score": float(s), "is_anomaly": bool(s > THRESHOLD),
             "top_features": feature_names[t].tolist()} for s, t in zip(scores, top)]
```

`X` must be prepared rows (40 features, same order and scaling as `data/test.npz`).
