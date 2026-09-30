# 🛠️ Phase 4 — Baseline Modeling & Dataset Troubleshooting

> **Phase Focus:** Initial Multilayer Perceptron (MLP) baseline implementation, analysis of early 99%+ overfitting, the constant cell temperature hypothesis, the `pvlib` synthetic dataset attempt, dataset re-validation, and integration of the DKASC longitudinal dataset.

---

## 📁 Files Included in This Phase Folder
* 🐍 **[`study_time_series.py`](study_time_series.py):** Initial baseline time-series LSTM benchmark script.

---

## 1. Initial Baseline MLP Approach & Overfitting

We initiated our empirical modeling phase using a standard **Multilayer Perceptron (MLP)** / Artificial Neural Network architecture to map historical weather parameters directly to solar power generation.

```
Input Features (8 cols) ---> MLP Dense Layers (128 -> 64 -> 32) ---> Power (P_mp)
```

### The Initial Failure (Anomalous 99%+ R²):
* **Observation:** During early training, the MLP achieved an anomalously high $R^2 > 0.99$ on the training split, but failed completely on out-of-sample test evaluation.
* **Diagnosis:** The network was rote-memorizing training data points (lookup table behavior) rather than learning generalizable non-linear weather dynamics.

---

## 2. The Constant Cell Temperature Misdiagnosis

Investigating the root cause of early model failure, we initially formed a hypothesis based on a misinterpretation of baseline literature:

* **Hypothesis:** We mistakenly assumed that cell temperature ($T_{\text{cell}}$) was treated as a constant value throughout the primary dataset (`xSi12922.csv`), leading us to believe that the lack of thermal variance was causing the model to overfit.

---

## 3. The `pvlib` Synthetic Dataset Attempt & Failure

To bypass the supposed constant cell temperature flaw, we built an automated simulation pipeline using the **`pvlib`** Python library to synthesize a clean mathematical dataset.

```
Synthetic Weather ---> pvlib Single-Diode Solver ---> Hardcoded Target Power
```

### Unexpected Limitation of Synthetic Data:
* **Outcome:** When trained on `pvlib` data, the neural network suffered from a new, severe form of overfitting: because the synthetic data was generated directly by hardcoded physical equations, the neural network simply **memorized the algebraic formulas** instead of learning real-world weather-driven variances!
* **Resolution:** Re-examining the original `xSi12922.csv` telemetry distributions revealed that cell temperature was **naturally distributed and dynamic** ($T_{\text{cell}} \in [5^\circ\text{C}, 65^\circ\text{C}]$). The original telemetry was entirely valid, allowing us to safely discard the synthetic `pvlib` approach.

---

## 4. Dataset Re-Validation & DKASC Expansion

Having re-validated `xSi12922.csv` (XS1) as our primary baseline foundation, we expanded our experimental framework by discovering and integrating the **DKASC** (`mSi0166.csv`) dataset as a second robust longitudinal source:

| Dataset | Cell Technology | Sample Count | Peak Capacity | Dynamic Range ($\operatorname{Var}(y)$) | Location |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **`xSi12922.csv` (XS1)** | Monocrystalline Silicon | 35,861 | 70 W | **634.4** | Golden, Colorado |
| **`mSi0166.csv` (DKASC)** | Multicrystalline Silicon | 33,899 | 38 W | **211.2** | Alice Springs, Australia |

This dual-dataset framework established the foundation for rigorous cross-asset validation and overfitting analysis in later phases.
