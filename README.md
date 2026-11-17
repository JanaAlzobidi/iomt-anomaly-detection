# IoMT Anomaly Detection (CICIoMT2024)

Unsupervised detection of cyberattacks on Internet of Medical Things (IoMT) devices.
Every model is trained on **benign traffic only** and flags attacks as anomalies, so it can detect attack types it has never seen.

We compare four autoencoder-based approaches on the same data, with the same threshold rule and the same metrics:

| Model | Idea |
|---|---|
| Dense Autoencoder | Learns to reconstruct benign traffic; high reconstruction error = attack |
| Denoising Autoencoder | Trained on noisy inputs to reconstruct the clean version, so it learns the structure of benign traffic instead of copying it |
| Sparse Autoencoder | L1 penalty on the bottleneck activations, so only the most important patterns are encoded |
| AE + Isolation Forest | Isolation Forest applied to the autoencoder's latent features instead of the reconstruction error |

## Dataset

**CICIoMT2024** (Canadian Institute for Cybersecurity, UNB): network traffic from 40 IoMT devices under 18 attacks in 5 categories (DDoS, DoS, Recon, MQTT, Spoofing). We use the WiFi + MQTT subset, from the cleaned release on Zenodo (record 19452870, `wifi_mqtt_clean.parquet`, 9,128,910 rows, 46 numeric features).

Each row describes a **window of packets** (not a single packet): rates, flag proportions, protocol shares and packet length statistics within that window.

## Data preparation (`data_prep/01_data_prep.ipynb`)

### Splits

| Set | Rows | Content | Source |
|---|---|---|---|
| Train | 150,000 | Benign only | original *train* split |
| Validation | 30,000 | Benign only | original *train* split |
| Test | 42,893 | 37,607 benign + 5,286 attacks (12.3%) | original *test* split |

- The original train/test split is kept, so test traffic comes from different capture files than training traffic.
- The test set contains all available benign test rows and 300 samples of each attack type (186 for Recon PingSweep, which has fewer). This keeps attacks rare, as in a real network, while allowing per-attack evaluation. The raw dataset is the opposite: over 90% of its rows are attacks.
- Duplicate rows were checked across all sets (none found).

### Profiling traffic is excluded

The dataset also contains 361,081 benign *profiling* rows, recorded in separate experiments (single devices powering on, an idle lab overnight, researchers interacting with devices). The benign traffic in the test set was recorded during the attack scenarios, in different network conditions.

Including profiling traffic in training raised the false alarm rate of both baselines from under 2% to about 8.5%, a clear case of **distribution shift**: the model learned "normal" from different conditions than the ones it was tested in. Training therefore uses only benign traffic from the original train split.

### Features (46 → 40)

| Removed | Reason |
|---|---|
| `drate`, `time_to_live` | Constant across the whole dataset |
| `srate` | Identical to `rate` in the training data (`drate` is always 0, so total rate = outgoing rate) |
| `llc` | Identical to `ipv` |
| `smtp`, `irc` | Always 0 in benign traffic, identical to `telnet` |

`iat` (mean inter-arrival time between packets in the window) is kept: it is an official behavioural feature and it does not separate attacks on its own.

**Leakage check:** single-feature ROC-AUC was computed for every feature. The strongest single feature is `header_length` (0.899), so no feature separates attacks almost perfectly on its own, and the models must combine several features.

### Scaling

- `log1p` on heavy-tailed columns (values from 0 to millions): `header_length`, `duration`, `rate`, `rst_count`, `tot_sum`, `min`, `max`, `avg`, `std`, `tot_size`, `iat`, `radius`, `covariance`, `weight`.
- MinMax scaling fitted on the training set only. Training data lies in [0, 1]; test values reach 8.0 for some attacks, which is expected and produces large reconstruction errors.
- The scaler and the preprocessing steps are saved (`scaler.pkl`, `config.json`) so new raw data can be transformed in exactly the same way at inference time.

## Evaluation (`evaluate.py`)

All models use the same evaluation:

- **Threshold:** 99th percentile of the anomaly scores on the benign validation set. No attack labels are used to choose it, so by design about 1% of normal traffic raises an alert.
- **Metrics:** ROC-AUC, PR-AUC, F1, precision, recall, false alarm rate, confusion matrix, and recall per attack type.
- `compare_percentiles()` shows how the 95th / 99th / 99.5th percentile changes the trade-off between missed attacks and false alarms.
- Each run is appended to `results.csv` and its plots are saved to `figures/`.

## Results

| Model | ROC-AUC | PR-AUC | F1 | Precision | Recall | False alarm rate | Weakest attack (recall) |
|---|---|---|---|---|---|---|---|
| Baseline: Isolation Forest | 0.860 | 0.417 | 0.045 | 0.178 | 0.026 | 1.66% | DDoS SYN Flood (0.00) |
| Baseline: PCA | 0.938 | 0.835 | 0.761 | 0.959 | 0.631 | 0.37% | ARP Spoofing (0.01) |
| Dense AE | | | | | | | |
| Denoising AE | | | | | | | |
| Sparse AE | | | | | | | |
| AE + Isolation Forest | | | | | | | |

**Baseline observations**

- **PCA** detects almost all flood attacks (DoS / DDoS: 99-100%) but misses most MQTT attacks and ARP Spoofing, which look closer to normal traffic. These are the attacks where a non-linear model can add value.
- **Isolation Forest** fails on flood attacks (recall ≈ 0). A likely reason: it isolates points with random splits inside the training range, so values far outside that range are not scored as more anomalous than the most extreme benign points. Reconstruction-based methods (PCA, autoencoders) do not have this limitation, since their error keeps growing with the distance from normal data.

## Using the shared data in a model notebook

The prepared files live in a shared Google Drive folder (they are not committed to the repository).

```python
from google.colab import drive; drive.mount("/content/drive")
import sys
DATA_DIR = "/content/drive/MyDrive/IoMT_project/data"
sys.path.insert(0, DATA_DIR)
from evaluate import load_data, reconstruction_errors, evaluate, compare_percentiles

X_train, X_val, X_test, y_test, attack_test = load_data(DATA_DIR)

# build and train the model on X_train, with validation_data=(X_val, X_val)

val_scores  = reconstruction_errors(model, X_val)
test_scores = reconstruction_errors(model, X_test)
evaluate(test_scores, y_test, attack_test, val_scores,
         model_name="Dense AE", person="1", save_dir=DATA_DIR)
```

Rules:

- Do not re-split or re-scale the data. Use the saved files as they are.
- Use `SEED = 42`.
- Use a `linear` output layer in the decoder: test values can exceed 1.
- For inference on raw rows, `preprocess_raw(df_raw, DATA_DIR)` applies the same preprocessing.
- AE + Isolation Forest: use `-iso.score_samples(Z)` as the score (higher = more anomalous).

## Repository structure

```
data_prep/
  01_data_prep.ipynb      data preparation (run once)
  README.md               dataset and preprocessing details
evaluate.py               shared evaluation used by every model
models/
  baselines/              PCA baseline
  dense_ae/
  denoising_ae/
  sparse_ae/
  ae_isoforest/
results/
  results.csv             comparison table
  figures/
requirements.txt
```

## Limitations

- **Single sample:** results come from one random sample (seed 42); a different seed would change the numbers slightly.
- **Test composition:** 300 samples per attack type is a choice for fair per-attack evaluation, not the real-world attack distribution.
- **Lab data:** all traffic comes from one testbed. Models trained here would likely need retraining on the benign traffic of a real hospital network, as the profiling experiment showed.
- **Dropped constant features:** columns that are constant in benign traffic (e.g. `smtp`) carry no signal for the attacks in this dataset, but in a real deployment a sudden non-zero value in such a column could be a strong sign of a new attack.
- **Threshold:** a fixed percentile assumes the benign traffic does not change over time; a deployed system would need periodic recalibration.

## References

- S. Dadkhah, E. C. P. Neto, R. Ferreira, R. C. Molokwu, S. Sadeghi and A. A. Ghorbani, "CICIoMT2024: A benchmark dataset for multi-protocol security assessment in IoMT," *Internet of Things*, vol. 28, 2024. Dataset: https://www.unb.ca/cic/datasets/iomt-dataset-2024.html
- Cleaned release used in this project: https://zenodo.org/records/19452870
