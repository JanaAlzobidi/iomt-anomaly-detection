# Sparse Autoencoder
 
The Vanilla AE architecture with one change: an **L1 activity penalty** on the bottleneck layer. The penalty pushes most latent values towards zero, so the model has to describe benign traffic with only a few active units. The idea is that attacks need patterns this sparse code never learned, so they reconstruct worse. Like every model in the project, it is trained on benign traffic only, and the reconstruction error (MSE) is the anomaly score.
 
**Question:** does an L1 penalty on the bottleneck improve detection compared with the Vanilla AE?
 
## Files
 
| File | Content |
|---|---|
| `sparse_ae.ipynb` | Full pipeline: model design, tuning, training, sparsity check, evaluation, comparison with the Vanilla AE, save/load, inference |
| `sparse_ae.keras` | Trained model |
| `sparse_ae_metadata.json` | Threshold, feature order and hyperparameters, needed at inference time |
 
## Architecture
 
Same as the Vanilla AE, plus the L1 penalty on the latent layer:
 
`40 → Dense(48, ReLU) → Dense(32, linear) "latent" + L1 activity penalty → Dense(48, ReLU) → Dense(40, linear)`
 
| Hyperparameter | Value |
|---|---|
| Latent size | 32 |
| Hidden width | 48 (one hidden layer per side) |
| Latent activation | linear |
| Dropout | 0.0 |
| L2 (weights, every layer) | 1e-6 |
| **L1 (latent activations)** | **1e-6** |
| Learning rate | 0.0005 (Adam) |
 
## Tuning
 
Only the L1 strength is tuned; everything else is fixed so the comparison with the Vanilla AE isolates the effect of the penalty. Keras adds the activity penalty for every latent value in the batch (512 × 32), while the benign reconstruction MSE is around 1e-5, so the useful range is much lower than the 1e-5 to 1e-3 in the project plan. The search covered `0` (no penalty, i.e. the Vanilla AE), 1e-8, 1e-7, 1e-6, 1e-5, 1e-4 and 1e-3, up to 40 epochs each.
 
The selected value has the highest recall on synthetic anomalies (validation rows with 3 random features replaced by random values, the same rule as the Denoising AE). Only training and validation data are used; no test data and no attack labels.
 
| L1 | val MSE | Near-zero latent values | Synthetic anomalies caught |
|---|---|---|---|
| 0 (Vanilla) | 2.25e-5 | 2.0% | 0.952 |
| 1e-8 | 1.82e-5 | 8.3% | 0.973 |
| 1e-7 | 2.01e-5 | 8.0% | 0.970 |
| **1e-6** | 2.57e-5 | 14.9% | **0.973** |
| 1e-5 | 1.64e-4 | 13.2% | 0.954 |
| 1e-4 | 1.24e-3 | 12.6% | 0.892 |
| 1e-3 | 3.01e-2 | 2.1% | 0.027 |
 
1e-8, 1e-7 and 1e-6 are practically tied (1e-6 wins by one synthetic row out of 5,000). From 1e-5 the penalty starts to hurt reconstruction, and at 1e-3 the model collapses.
 
## Training
 
Same as the Vanilla AE: up to 200 epochs, batch size 512, Early Stopping on `val_mse` (patience 10, best weights restored) and the learning rate halved when validation is stuck for 4 epochs. The best epoch was the last one (200).
 
## Sparsity check
 
Compared with the saved Vanilla AE on benign validation data, the penalty clearly made the bottleneck sparser: 17.1% of latent values are near zero (vs 2.0%), and the average absolute latent value is about 60 times smaller (0.011 vs 0.667). Since the latent layer is linear, part of that shrinkage is a rescaling that the decoder compensates for, but the near-zero share is measured relative to each model's most active unit, so more units really are switched off.
 
## Results
 
Threshold: 99th percentile of benign validation scores (0.000166).
 
| Metric | PCA baseline | Vanilla AE | Sparse AE |
|---|---|---|---|
| ROC-AUC | 0.938 | 0.955 | **0.955** |
| PR-AUC | 0.835 | 0.903 | **0.905** |
| F1 | 0.761 | 0.888 | **0.891** |
| Precision | **0.959** | 0.934 | 0.940 |
| Recall | 0.631 | 0.847 | **0.847** |
| False alarm rate | **0.37%** | 0.84% | 0.76% |
 
Recall on the hardest attacks:
 
| Attack | Vanilla AE | Sparse AE |
|---|---|---|
| ARP Spoofing | 2.3% | 2.3% |
| MQTT Malformed Data | 24.0% | **25.7%** |
| Recon VulScan | **47.0%** | 45.7% |
| Recon PingSweep | 52.2% | **53.8%** |
 
All DoS, DDoS and MQTT flood attacks are detected at 100%, and Recon OSScan and PortScan at 89% and 96%.
 
**Answer to the question:** slightly. The Sparse AE is a little better than the Vanilla AE on every main metric, mostly through fewer false alarms (287 vs 316 benign rows flagged), but the gap is small. The hard attacks change by only a few rows each: in the 40 available features they look like normal traffic, and a different compression of the data does not change that.
 
**Sensitivity to the run.** Training on Colab is not fully reproducible even with `SEED = 42`. In an earlier run of the same notebook, `l1 = 1e-8` won the tuning by a similarly tiny margin, and that model reached only F1 = 0.810 because of more false alarms (3.4%) at the same P99 rule. The difference between the Sparse and Vanilla AE is smaller than the difference between two runs of this notebook.
 
## Inference
 
```python
import json, numpy as np, keras
from evaluate import per_feature_errors
 
model = keras.models.load_model("models/sparse_ae/sparse_ae.keras")
meta = json.load(open("models/sparse_ae/sparse_ae_metadata.json"))
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
 
