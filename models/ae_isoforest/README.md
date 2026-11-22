# Autoencoder + Isolation Forest

An anomaly detection approach that combines an **Autoencoder (AE)** for feature representation learning with **Isolation Forest (IF)** for detecting unusual patterns in IoMT network traffic.

This experiment investigates whether anomaly detection in the autoencoder's **latent space** is more effective than using **reconstruction error**.

## Approach

Both models are trained using benign network traffic only.

The main pipeline follows these steps:

1. Train an Autoencoder to reconstruct normal traffic.
2. Extract compressed features from the Autoencoder's `latent` layer.
3. Train Isolation Forest on the benign latent representations.
4. Calculate anomaly scores using `-iso.score_samples()`.
5. Apply a threshold based on the 99th percentile of benign validation scores.

The final AE + IF anomaly score comes from Isolation Forest, not reconstruction error.

## Files

| File | Description |
|---|---|
| `AE_IsolationForest.ipynb` | Main AE + Isolation Forest implementation, tuning, evaluation, and comparison |
| `AE_IsolationForest_wReError.ipynb` | Additional experiment comparing latent-space Isolation Forest with AE reconstruction error |
| `ae_isoforest.keras` | Saved Autoencoder model from the main implementation |
| `README.md` | Documentation of the approach, experiments, and findings |

## Main Experiment Results

The main experiment selected an Autoencoder with 32 latent dimensions and an Isolation Forest with 400 estimators and `max_samples=8192`.

| Metric | PCA Baseline | AE + Isolation Forest |
|---|---:|---:|
| ROC-AUC | 0.938 | 0.883 |
| PR-AUC | 0.835 | 0.616 |
| F1 Score | 0.761 | 0.563 |
| Precision | 0.959 | 0.563 |
| Recall | 0.631 | 0.562 |
| False Alarm Rate | 0.37% | 6.12% |

The AE + Isolation Forest model did not outperform PCA overall. However, it improved detection of Recon PingSweep attacks compared with the PCA baseline.

## Additional Experiment: Reconstruction Error vs. Isolation Forest

To further investigate the performance difference, a separate experiment compared two anomaly scoring methods using the **same trained Autoencoder**:

- **Latent Isolation Forest:** Uses the encoded latent features to calculate anomaly scores.
- **AE Reconstruction Error:** Uses the mean squared error (MSE) between the original input and its reconstruction.

This additional experiment uses a different Autoencoder architecture from the main implementation.

| Metric | Latent Isolation Forest | AE Reconstruction Error |
|---|---:|---:|
| ROC-AUC | 0.7983 | **0.9616** |
| PR-AUC | 0.2399 | **0.9048** |
| F1 Score | 0.0135 | **0.8747** |
| False Alarm Rate | 1.60% | 0.87% |

The reconstruction-error approach achieved substantially better anomaly detection performance in this comparison.

## Key Findings

- **Reconstruction error was a stronger anomaly signal** than Isolation Forest applied to latent features in the evaluated experiments.
- A low Autoencoder reconstruction loss does not necessarily mean its latent representation will work well for Isolation Forest.
- The main AE + IF model showed better recall on Recon PingSweep than PCA, despite lower overall performance.
- Adding model complexity does not guarantee improved anomaly detection.
- The results suggest that the choice of anomaly scoring method can significantly affect detection performance.

## Conclusion

The experiments highlight the importance of evaluating both the learned feature representation and the anomaly scoring method.

Although the AE + Isolation Forest approach did not outperform the PCA baseline, the additional reconstruction-error comparison provided useful evidence that **reconstruction-based detection was more effective for this dataset**.

The main implementation remains the latent-space AE + Isolation Forest model, while reconstruction error is documented as an additional comparative experiment.
