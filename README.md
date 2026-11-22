# IoMT Anomaly Detection with Autoencoders
 
Detecting cyberattacks on Internet of Medical Things (IoMT) devices without attack labels. Every model is trained on **benign network traffic only** and flags anything it cannot model well as an anomaly, so it can, in principle, catch attack types it has never seen.
 
We built one shared dataset and evaluation, then compared four autoencoder-based approaches against a PCA baseline.
 
## Team
 
| Member | Responsibilities | Model question |
|---|---|---|
| Jana | Data preparation, **Vanilla AE** | How much does a plain non-linear autoencoder improve on linear PCA? |
| Omar | Shared evaluation, **Denoising AE** | Does training on noisy inputs improve detection? |
| Raghad | Deployment, **Sparse AE** | Does an L1 penalty on the bottleneck improve detection? |
| Nouf | PCA baseline, **AE + Isolation Forest** | Is the autoencoder's latent space a better signal than its reconstruction error? |
 
The Denoising and Sparse autoencoders reuse the Vanilla AE architecture and training settings and add one change each, so any difference in results comes from that change alone.
 
## Results
 
All models use the same data and the same threshold rule: the 99th percentile of anomaly scores on benign validation data. No attack label is used to train a model, tune it, or set its threshold.
 
| Model | Member | ROC-AUC | PR-AUC | F1 | Precision | Recall | False alarms |
|---|---|---|---|---|---|---|---|
| PCA (baseline) | Nouf | 0.938 | 0.835 | 0.761 | **0.959** | 0.631 | **0.37%** |
| Vanilla AE | Jana | 0.955 | 0.903 | 0.888 | 0.934 | 0.847 | 0.84% |
| Denoising AE | Omar | **0.961** | 0.903 | 0.840 | 0.829 | **0.850** | 2.46% |
| **Sparse AE** (deployed) | Raghad | 0.955 | **0.905** | **0.891** | 0.940 | 0.847 | 0.76% |
| AE + Isolation Forest | Nouf | 0.883 | 0.616 | 0.563 | 0.563 | 0.562 | 6.12% |
 
### Recall on the hardest attacks
 
| Attack | PCA | Vanilla AE | Denoising AE | Sparse AE | AE + Isolation Forest |
|---|---|---|---|---|---|
| MQTT Flood attacks (4 types) | 6–29% | **100%** | **100%** | **100%** | 0–1% |
| Recon PingSweep | 38% | 52% | 53% | 54% | **65%** |
| Recon VulScan | 40% | 47% | **51%** | 46% | 17% |
| MQTT Malformed Data | 4% | 24% | 25% | **26%** | 5% |
| ARP Spoofing | 1% | 2% | **3%** | 2% | 0% |
 
DoS, DDoS and the other Recon attacks (OSScan, PortScan) are detected at 88–100% by the Vanilla, Denoising and Sparse AEs.
 
### Findings
 
- **The autoencoder clearly beats PCA.** F1 rises from 0.761 to 0.888. The gain comes mostly from the four MQTT flood attacks (from 6–29% to 100%), which a linear model cannot separate from normal traffic.
- **Input noise improves ranking but not the final decision.** The Denoising AE has the highest ROC-AUC and improves every hard attack, but it raises false alarms at the shared threshold (2.46%), which lowers its F1.
- **A sparse bottleneck matches the Vanilla AE, with slightly fewer false alarms.** An L1 penalty on the bottleneck activations (strength 1e-6) made the code clearly sparser (17% near-zero latent values instead of 2%) and gave the best F1 (0.891) and PR-AUC (0.905), mostly through fewer false alarms (0.76% vs 0.84%). The gain is small: a second run of the same notebook, where a weaker penalty won the tuning, reached only F1 = 0.810.
- **Reconstruction error is a better signal than the latent space.** Isolation Forest on the latent features misses almost all MQTT attacks. The same autoencoder scored by reconstruction error reaches F1 = 0.888, matching the Vanilla AE.
- **A performance ceiling around F1 ≈ 0.89.** The Vanilla, Sparse and reconstruction-scored autoencoders all land between 0.888 and 0.891. Ten alternative training settings, five alternative scoring methods and a 100-trial architecture search all landed at the same level. ARP Spoofing, MQTT Malformed Data and some Recon attacks look like normal traffic in the available features, so better training alone does not catch them.

## Deployment
 
The best model by F1, the **Sparse AE**, is deployed. The choice is made in [`inference/compare_models.ipynb`](inference/compare_models.ipynb), which reloads every saved model from this repository and evaluates them all again with `evaluate.py` in one run. The rule: highest F1, then PR-AUC and false alarms when models are close. The Sparse AE wins on all three, by a small margin over the Vanilla AE.
 
| File | Content |
|---|---|
| `inference/compare_models.ipynb` | Re-evaluates every saved model, comparison table, choice of the deployed model |
| `inference/inference.py` | Inference pipeline: loads the Sparse AE and its threshold, `predict(X)` |
| `inference/demo.ipynb` | Bilingual (English / Arabic) Gradio demo with a shareable link |
 
`predict(X)` takes prepared rows (40 features, same order and scaling as `data/test.npz`) and returns, for each row, the anomaly `score`, the decision `is_anomaly`, and the 3 `top_features` with the largest reconstruction error, which explain the decision.
 
```python
import sys
sys.path.insert(0, "iomt-anomaly-detection/inference")
from inference import predict
 
results = predict(X_test[:3])    # one dict per row: score, is_anomaly, top_features
```
 
The demo lets anyone pick a test sample (one of three prepared examples, a random one, or by number) and shows the decision, how far the score is from the alarm line, the true traffic type with a short explanation, and the features that looked most abnormal. It runs in Colab: open `inference/demo.ipynb`, Run all, and open the link printed by the last cell.
 
## Dataset
 
**CICIoMT2024** (Canadian Institute for Cybersecurity, UNB): network traffic from 40 IoMT devices under 18 attacks in 5 categories (DDoS, DoS, Recon, MQTT, Spoofing). We use the WiFi + MQTT subset from the cleaned release on Zenodo (record 19452870). Each row describes a window of packets with 46 numeric features.
 
| Set | Rows | Content |
|---|---|---|
| Train | 150,000 | Benign only |
| Validation | 30,000 | Benign only |
| Test | 42,893 | 37,607 benign + 5,286 attacks (12.3%), 300 per attack type |
 
Main preparation decisions (details in [`data/README.md`](data/README.md)):
 
- The original train/test split is kept, so test rows come from different capture files.
- Profiling traffic (idle lab, single devices) is excluded: it was recorded in different conditions from the test traffic and raised false alarms to about 8%.
- 6 constant or duplicate columns removed (46 → 40 features); no single feature separates attacks on its own (leakage check).
- `log1p` on heavy-tailed columns, then MinMax scaling fitted on the training set only.

## Repository structure
 
```
data/
  01_data_prep.ipynb      data preparation (download, sampling, cleaning, scaling)
  README.md               dataset and preprocessing details
  train.npz, val.npz, test.npz, config.json
evaluate.py               shared evaluation used by every model (Omar)
models/
  baselines/              PCA baseline (Nouf)
  dense_ae/               Vanilla AE: notebook, trained model, metadata (Jana)
  denoising_ae/           Denoising AE (Omar)
  sparse_ae/              Sparse AE (Raghad)
  ae_isoforest/           AE + Isolation Forest (Nouf)
inference/                deployment of the best model (Raghad)
  compare_models.ipynb    re-evaluates all saved models and picks the best one
  inference.py            inference pipeline: load_model() and predict()
  demo.ipynb              Gradio demo
requirements.txt
```
 
Each model folder contains its notebook (full pipeline with outputs, including a `predict()` inference function) and the saved Keras model. The Vanilla and Sparse AE folders also hold a metadata file with the threshold and feature order. Not yet in the repository: the Denoising AE's threshold file and the AE + Isolation Forest's forest (`isolation_forest.pkl`) and metadata. PCA has no saved model; its notebook refits it in seconds.
 
## How to run
 
In Google Colab:
 
```python
!git clone https://github.com/JanaAlzobidi/iomt-anomaly-detection.git
import sys
sys.path.insert(0, "iomt-anomaly-detection")
from evaluate import load_data, reconstruction_errors, evaluate
 
X_train, X_val, X_test, y_test, attack_test = load_data("iomt-anomaly-detection/data")
```
 
Then open any notebook in `models/` and run all cells. The prepared data is in the repository, so the raw dataset does not need to be downloaded.
 
Using the trained Vanilla AE:
 
```python
import json, numpy as np, keras
from evaluate import per_feature_errors
 
model = keras.models.load_model("iomt-anomaly-detection/models/dense_ae/vanilla_ae.keras")
meta = json.load(open("iomt-anomaly-detection/models/dense_ae/vanilla_ae_metadata.json"))
 
scores = reconstruction_errors(model, X_test[:5])
print(scores > meta["threshold"])        # True = attack
```
 
## Limitations
 
- **Lab data.** All traffic comes from one testbed. A real hospital network would need retraining on its own benign traffic, as the profiling experiment showed.
- **Test composition.** 300 rows per attack type allows fair per-attack evaluation but does not reflect real attack frequencies.
- **Fixed threshold.** The 99th percentile assumes benign traffic does not change over time. The Denoising AE shows the threshold does not always transfer from validation to test.
- **Hard attacks.** ARP Spoofing and MQTT Malformed Data stay mostly undetected; catching them would likely need different features, such as packet sequences over time, rather than a different model.
- **Single run.** Results come from one random seed (42). Training is not fully reproducible even with a fixed seed, and the differences between the top autoencoders (about 0.003 F1) are smaller than the variation between two runs of the same notebook.

## Reference
 
S. Dadkhah, E. C. P. Neto, R. Ferreira, R. C. Molokwu, S. Sadeghi and A. A. Ghorbani, "CICIoMT2024: A benchmark dataset for multi-protocol security assessment in IoMT," *Internet of Things*, vol. 28, 2024.
 
