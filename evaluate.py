"""
evaluate.py - shared evaluation for the CICIoMT2024 anomaly detection project.

Every model is evaluated with this file so the comparison is fair:
same data, same threshold rule, same metrics.

Score convention: higher score = more anomalous.
If your method returns the opposite (e.g. IsolationForest.score_samples), multiply by -1.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

SEED = 42


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_data(data_dir: str):
    """Returns X_train, X_val, X_test, y_test, attack_test."""
    tr = np.load(os.path.join(data_dir, "train.npz"))
    va = np.load(os.path.join(data_dir, "val.npz"))
    te = np.load(os.path.join(data_dir, "test.npz"), allow_pickle=True)
    X_train, X_val, X_test = tr["X"], va["X"], te["X"]
    y_test, attack_test = te["y"], te["attack_type"]
    print(f"X_train {X_train.shape} (benign only) | X_val {X_val.shape} (benign only) | "
          f"X_test {X_test.shape}, attacks: {int(y_test.sum())} ({y_test.mean():.1%})")
    return X_train, X_val, X_test, y_test, attack_test


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------
def reconstruction_errors(model, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    """Per-sample MSE between input and reconstruction (any Keras model)."""
    recon = model.predict(X, batch_size=batch_size, verbose=0)
    return np.mean(np.square(X - recon), axis=1)


def per_feature_errors(model, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    """Squared error per feature, useful to show which features caused an alert."""
    recon = model.predict(X, batch_size=batch_size, verbose=0)
    return np.square(X - recon)


# ---------------------------------------------------------------------------
# Threshold: computed from benign validation scores only (no labels used)
# ---------------------------------------------------------------------------
def threshold_from_val(val_scores: np.ndarray, percentile: float = 99.0) -> float:
    return float(np.percentile(val_scores, percentile))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(test_scores, y_test, attack_test, val_scores,
             model_name: str, person: str = "", percentile: float = 99.0,
             save_dir: str | None = None, plot: bool = True) -> dict:
    """
    test_scores : anomaly scores for the test set (higher = more anomalous)
    y_test      : 0 = benign, 1 = attack
    attack_test : traffic type for each test sample ("Benign" or attack name)
    val_scores  : scores for the validation set (all benign), used for the threshold
    """
    from sklearn.metrics import (average_precision_score, confusion_matrix,
                                 f1_score, precision_score, recall_score,
                                 roc_auc_score)

    test_scores = np.asarray(test_scores, dtype=float).ravel()
    val_scores = np.asarray(val_scores, dtype=float).ravel()
    y_test = np.asarray(y_test).astype(int).ravel()
    attack_test = np.asarray(attack_test).astype(str).ravel()
    assert len(test_scores) == len(y_test) == len(attack_test), "length mismatch"

    thr = threshold_from_val(val_scores, percentile)
    y_pred = (test_scores > thr).astype(int)

    res = {
        "model": model_name,
        "person": person,
        "threshold": thr,
        "percentile": percentile,
        "roc_auc": roc_auc_score(y_test, test_scores),
        "pr_auc": average_precision_score(y_test, test_scores),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "false_alarm_rate": float(y_pred[y_test == 0].mean()),
    }
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    res["confusion_matrix"] = cm.tolist()

    # recall per attack type
    df = pd.DataFrame({"type": attack_test, "y": y_test, "pred": y_pred})
    per_type = (df.groupby("type")
                  .agg(n=("pred", "size"), detected=("pred", "mean"))
                  .sort_values("detected"))
    attacks_only = per_type.drop(index=[t for t in per_type.index if t.lower() == "benign"],
                                 errors="ignore")
    if len(attacks_only):
        res["weakest_attack"] = attacks_only.index[0]
        res["weakest_attack_recall"] = float(attacks_only["detected"].iloc[0])
    res["per_attack_recall"] = attacks_only["detected"].round(4).to_dict()

    print(f"\n===== {model_name} =====")
    print(f"Threshold (P{percentile:g} of validation scores): {thr:.6g}")
    for k in ["roc_auc", "pr_auc", "f1", "precision", "recall", "false_alarm_rate"]:
        print(f"{k:>17}: {res[k]:.4f}")
    print("\nConfusion matrix (rows = true, cols = predicted)")
    print(pd.DataFrame(cm, index=["Benign", "Attack"], columns=["Benign", "Attack"]))
    print("\nRecall per attack type (weakest first):")
    print(attacks_only.rename(columns={"detected": "recall"}).to_string(
        formatters={"recall": "{:.3f}".format}))

    if plot:
        _plot(test_scores, y_test, val_scores, thr, attacks_only, model_name, save_dir)
    if save_dir:
        _save_result(res, save_dir)
    return res


def compare_percentiles(test_scores, y_test, val_scores, percentiles=(95, 99, 99.5)):
    """Shows how the threshold percentile affects F1, recall and false alarms."""
    from sklearn.metrics import f1_score, recall_score
    y_test = np.asarray(y_test).astype(int)
    rows = []
    for p in percentiles:
        thr = threshold_from_val(val_scores, p)
        pred = (np.asarray(test_scores) > thr).astype(int)
        rows.append({"percentile": p, "threshold": thr,
                     "f1": f1_score(y_test, pred, zero_division=0),
                     "recall": recall_score(y_test, pred, zero_division=0),
                     "false_alarm_rate": pred[y_test == 0].mean()})
    out = pd.DataFrame(rows)
    print(out.to_string(index=False, float_format="{:.4f}".format))
    return out


# ---------------------------------------------------------------------------
# Inference: apply the same preprocessing to raw rows (original column names)
# ---------------------------------------------------------------------------
def preprocess_raw(df_raw: pd.DataFrame, data_dir: str) -> np.ndarray:
    import joblib
    with open(os.path.join(data_dir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    scaler = joblib.load(os.path.join(data_dir, "scaler.pkl"))
    X = df_raw[cfg["features"]].astype("float64").replace([np.inf, -np.inf], np.nan)
    if X.isna().any().any():
        raise ValueError("input contains NaN or inf values")
    for c in cfg["log_cols"]:
        X[c] = np.log1p(X[c].clip(lower=0))
    return scaler.transform(X.values).astype("float32")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _plot(test_scores, y_test, val_scores, thr, attacks_only, model_name, save_dir):
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    eps = 1e-12
    bins = np.logspace(np.log10(max(min(test_scores.min(), val_scores.min()), eps)),
                       np.log10(max(test_scores.max(), eps * 10)), 80)
    a = ax[0, 0]
    a.hist(test_scores[y_test == 0] + eps, bins=bins, alpha=.6, label="Benign", density=True)
    a.hist(test_scores[y_test == 1] + eps, bins=bins, alpha=.6, label="Attack", density=True)
    a.axvline(thr, color="k", ls="--", label=f"threshold={thr:.3g}")
    a.set_xscale("log"); a.set_title("Anomaly score distribution (test)"); a.legend()

    fpr, tpr, _ = roc_curve(y_test, test_scores)
    ax[0, 1].plot(fpr, tpr); ax[0, 1].plot([0, 1], [0, 1], "k:", lw=.8)
    ax[0, 1].set(title="ROC curve", xlabel="False positive rate", ylabel="True positive rate")

    pr, rc, _ = precision_recall_curve(y_test, test_scores)
    ax[1, 0].plot(rc, pr)
    ax[1, 0].axhline(y_test.mean(), color="k", ls=":", lw=.8, label="random")
    ax[1, 0].set(title="Precision-Recall curve", xlabel="Recall", ylabel="Precision")
    ax[1, 0].legend()

    a = ax[1, 1]
    if len(attacks_only):
        a.barh(attacks_only.index, attacks_only["detected"])
        a.set_xlim(0, 1); a.set_title("Recall per attack type")
        a.tick_params(axis="y", labelsize=8)
    fig.suptitle(model_name, fontsize=14)
    fig.tight_layout()
    if save_dir:
        fig_dir = os.path.join(save_dir, "figures")
        os.makedirs(fig_dir, exist_ok=True)
        fig.savefig(os.path.join(fig_dir, f"eval_{_slug(model_name)}.png"), dpi=120)
    plt.show()


def _save_result(res: dict, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "results.csv")
    cols = ["model", "person", "roc_auc", "pr_auc", "f1", "precision", "recall",
            "false_alarm_rate", "weakest_attack", "weakest_attack_recall",
            "threshold", "percentile"]
    new = pd.DataFrame([{k: res.get(k) for k in cols}])
    if os.path.exists(path):
        old = pd.read_csv(path)
        old = old[old["model"] != res["model"]]  # re-running a model replaces its row
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)
    with open(os.path.join(save_dir, f"result_{_slug(res['model'])}.json"), "w",
              encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=float)
    print(f"\nSaved to {path}")


def _slug(s: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in s).strip("_").lower()
