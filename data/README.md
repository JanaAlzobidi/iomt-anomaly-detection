# Data: CICIoMT2024 (WiFi + MQTT)

This folder holds the dataset shared by every model in the project, and the notebook that builds it. `01_data_prep.ipynb` downloads the raw data from Zenodo, builds the splits and saves the output files listed below, which are stored here (about 10 MB). The raw parquet file (9M rows) is not stored in the repository.

## Source

- **Dataset:** CICIoMT2024, Canadian Institute for Cybersecurity (UNB). Network traffic from 40 IoMT devices (25 real, 15 simulated) under 18 attacks in 5 categories.
- **Release used:** cleaned version on Zenodo (record 19452870), file `wifi_mqtt_clean.parquet` inside `Core Processed Data.zip`: 9,128,910 rows, 46 numeric features, plus `label`, `split`, `source_file` and `protocol` columns.
- **What a row is:** a window of packets, not a single packet. Features summarise the window: transmission rate, proportion of packets with each TCP flag, share of each protocol, and packet length statistics.

## Raw composition

| Category | Traffic type | Train | Test | Profiling |
|---|---|---:|---:|---:|
| Benign | Benign | 192,732 | 37,607 | 361,081 |
| DDoS | ICMP / SYN / TCP / UDP Flood | 4,779,187 | 1,066,762 | 0 |
| DoS | ICMP / SYN / TCP / UDP Flood | 1,805,529 | 416,676 | 0 |
| MQTT | DDoS/DoS Connect Flood, DDoS/DoS Publish Flood, Malformed Data | 262,938 | 63,715 | 0 |
| Recon | PingSweep, VulScan, PortScan, OSScan | 99,279 | 25,613 | 0 |
| Spoofing | ARP Spoofing | 16,047 | 1,744 | 0 |

Attacks make up about 93% of all rows, the opposite of a real network.

## Preparation steps

1. **Sampling.** The parquet file is read in batches of 500,000 rows and sampled per (split, traffic type), so the 9M rows never sit in memory.
2. **Profiling excluded.** Profiling traffic (devices powering on, idle lab overnight, researchers interacting with devices) was recorded in separate experiments. The benign test traffic was recorded during the attack scenarios. Training on profiling raised the PCA baseline's false alarm rate from 0.4% to about 8%, so only benign rows from the original *train* split are used for training and validation.
3. **Cleaning.** Infinite values replaced and rows with missing values dropped (none found). Exact duplicate rows removed, keeping the test copy so no row appears in both training and test (none found).
4. **Splits.** The original train/test split is kept: training and test rows come from different capture files.
5. **Feature selection (46 → 40).** Columns constant across the dataset or identical to another column in the training data are removed (see below).
6. **Leakage check.** Single-feature ROC-AUC for every feature. The strongest single feature is `header_length` (0.899), so no column separates attacks almost perfectly on its own.
7. **Transform.** `log1p` on heavy-tailed columns, then MinMax scaling fitted on the training set only.

## Final splits

| Set | Rows | Content |
|---|---:|---|
| Train | 150,000 | Benign only |
| Validation | 30,000 | Benign only |
| Test | 42,893 | 37,607 benign + 5,286 attacks (12.3%) |

The test set uses all benign test rows and 300 rows of each of the 18 attack types (186 for Recon PingSweep, which has fewer test rows). Attacks stay rare, as in practice, and every attack type can be evaluated on its own.

## Features

**Removed (6):**

| Column | Reason |
|---|---|
| `drate` | Constant (always 0) |
| `time_to_live` | Constant |
| `srate` | Identical to `rate` in the training data (incoming rate is always 0, so total rate = outgoing rate) |
| `llc` | Identical to `ipv` |
| `smtp`, `irc` | Always 0 in benign traffic, identical to `telnet` |

**Kept (40):** `header_length`, `protocol_type`, `duration`, `rate`, `fin_flag_number`, `syn_flag_number`, `rst_flag_number`, `psh_flag_number`, `ack_flag_number`, `ece_flag_number`, `cwr_flag_number`, `ack_count`, `syn_count`, `fin_count`, `rst_count`, `http`, `https`, `dns`, `telnet`, `ssh`, `tcp`, `udp`, `dhcp`, `arp`, `icmp`, `igmp`, `ipv`, `tot_sum`, `min`, `max`, `avg`, `std`, `tot_size`, `iat`, `number`, `magnitue`, `radius`, `covariance`, `variance`, `weight`.

`iat` is the mean inter-arrival time between packets in the window. It is an official behavioural feature and does not separate attacks on its own, so it is kept.

**log1p applied to (14):** `header_length`, `duration`, `rate`, `rst_count`, `tot_sum`, `min`, `max`, `avg`, `std`, `tot_size`, `iat`, `radius`, `covariance`, `weight`. These are non-negative columns whose values exceed 100 in the training data.

After scaling, training data lies in [0, 1]. Test values reach 8.0 for some attacks, which is expected: they fall far outside normal traffic and produce large reconstruction errors.

## Output files

| File | Content |
|---|---|
| `train.npz` | `X` (150,000 × 40, float32) |
| `val.npz` | `X` (30,000 × 40) |
| `test.npz` | `X` (42,893 × 40), `y` (0 = benign, 1 = attack), `attack_type`, `category` |
| `scaler.pkl` | MinMaxScaler fitted on the training set |
| `config.json` | Feature list and order, log1p columns, removed columns, split sizes, seed |

The class distribution and top-feature plots are shown in the notebook outputs. The notebook writes to `DATA_DIR` (a Google Drive folder in our runs); to regenerate the files, run it and copy the outputs into this folder.

`scaler.pkl` and `config.json` are needed at inference time: `preprocess_raw(df_raw, DATA_DIR)` in `evaluate.py` uses them to transform raw rows exactly like the training data.

## Loading

```python
!git clone https://github.com/JanaAlzobidi/iomt-anomaly-detection.git
import sys
REPO = "/content/iomt-anomaly-detection"
sys.path.insert(0, REPO)
DATA_DIR = f"{REPO}/data"
RESULTS_DIR = f"{REPO}/results"
from evaluate import load_data, reconstruction_errors, evaluate, compare_percentiles

SEED = 42
X_train, X_val, X_test, y_test, attack_test = load_data(DATA_DIR)
```

## Notes

- One random sample (seed 42). A different seed changes the exact rows but not the procedure.
- 300 rows per attack type is chosen for fair per-attack evaluation, not to match the real attack distribution.
- Columns that are always 0 in benign traffic (e.g. `smtp`) carry no signal for the attacks in this dataset, but in a real deployment a sudden non-zero value there could indicate a new attack.

## Reference

S. Dadkhah, E. C. P. Neto, R. Ferreira, R. C. Molokwu, S. Sadeghi and A. A. Ghorbani, "CICIoMT2024: A benchmark dataset for multi-protocol security assessment in IoMT," *Internet of Things*, vol. 28, 2024. https://www.unb.ca/cic/datasets/iomt-dataset-2024.html
