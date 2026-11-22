# Deployment: Sparse Autoencoder
 
This folder holds the deployment of the project's best model, the **Sparse AE** (`models/sparse_ae/`).
 
## Files
 
| File | Content |
|---|---|
| `compare_models.ipynb` | Reloads every saved model from `models/`, evaluates them all with `evaluate.py` in one run, and picks the model to deploy |
| `inference.py` | Inference pipeline: `load_model()` and `predict(X)` |
| `demo.ipynb` | Gradio demo in English and Arabic, with dark and light mode and a shareable link |
 
## 1. Choosing the model (`compare_models.ipynb`)
 
Every model is loaded from its saved `.keras` file and scored on the same validation and test data; `evaluate()` sets each threshold with the same rule (99th percentile of benign validation scores). PCA has no saved model file, so its numbers come from its notebook, and so do AE + Isolation Forest's while `isolation_forest.pkl` is not in the repository.
 
| Model | ROC-AUC | PR-AUC | F1 | False alarms |
|---|---|---|---|---|
| **Sparse AE** | 0.955 | **0.905** | **0.891** | 0.76% |
| Vanilla AE | 0.955 | 0.903 | 0.888 | 0.84% |
| Denoising AE | **0.961** | 0.903 | 0.840 | 2.46% |
| PCA (baseline) | 0.938 | 0.835 | 0.761 | **0.37%** |
| AE + Isolation Forest | 0.883 | 0.616 | 0.563 | 6.12% |
 
Rule: highest F1, then PR-AUC and false alarms when models are close. The Sparse AE wins on all three, by a small margin over the Vanilla AE. Its weakness: like every model, it misses almost all ARP Spoofing (2%) and most MQTT Malformed Data (26%), which look like normal traffic in the 40 features.
 
The notebook also checks that the threshold saved in `sparse_ae_metadata.json` equals the one `evaluate()` computes from the saved model, so the deployed model makes exactly the decisions in the table.
 
## 2. Inference pipeline (`inference.py`)
 
```python
import sys
sys.path.insert(0, "iomt-anomaly-detection/inference")
from inference import predict
 
results = predict(X_test[:3])
```
 
`predict(X)` takes one row or many, prepared like `data/test.npz` (40 features, same order and scaling), and returns one dict per row:
 
| Key | Meaning |
|---|---|
| `score` | Reconstruction error (higher = more abnormal) |
| `is_anomaly` | `True` if the score is above the saved threshold (= attack) |
| `top_features` | The 3 features with the largest reconstruction error, i.e. what looked abnormal |
 
The model and its metadata are loaded once and reused. Running the file directly (`python inference/inference.py`) prints predictions for 5 random test rows as a quick check.
 
## 3. Demo (`demo.ipynb`)
 
Open it in Google Colab and Run all; the last cell prints a public link (`share=True`) that works while the session runs. The demo:
 
- lets you pick a test sample: three prepared examples (a normal row, an easy attack that is caught, a hard attack that is missed), a random sample (half normal, half attacks), or any row by number
- shows the decision, a gauge with the sample's position relative to the alarm line, the true traffic type with a plain-language explanation, and the 3 features that looked most abnormal
- uses `predict()` from `inference.py` for every decision
- switches between English and Arabic (right-to-left) and between dark and light mode
The three examples are chosen automatically as the row with the median score of their group, so they are typical rather than extreme cases.
 
