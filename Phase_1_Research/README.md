# 📚 Phase 1 — Research & Literature Survey

> **Phase Focus:** Comprehensive academic literature review, analysis of existing empirical solar forecasting platforms, deep-dive into three foundational research papers, technology comparison (black-box AI vs. PINNs), and formal identification of the research gap.

---

## 📁 Files Included in This Phase Folder
* 📄 **[`literature_review.md`](literature_review.md):** 15-paper chronological literature review trace (2019–2026) documenting the evolution of PV modeling.
* 📄 **[`reproduction.md`](reproduction.md):** Empirical reproduction report for the baseline PKINN paper (Pei et al., 2026).
* 📄 **[`review_one.md`](review_one.md):** Deep-dive summary and critical analysis of foundational research papers.
* 📄 **[`PV Physics-Informed ML Review.pdf`](PV%20Physics-Informed%20ML%20Review.pdf):** Compiled reference literature trace document.

---

## 📖 1. Academic Literature Survey

Our literature survey traced the transition of photovoltaic (PV) power forecasting across three generations of technology:

1. **First-Generation Empirical Models (1990s–2010s):** Statistical regression (ARIMA, persistent models) and basic physical equations (Sandia, PVForm). High physical transparency, but failed under rapid weather volatility.
2. **Second-Generation Pure Data-Driven AI (2015–2023):** Standard Machine Learning (MLPs, LSTMs, CNNs, XGBoost, Transformers). Excellent curve-fitting under clear skies, but vulnerable to extreme weather drift, over-smoothing cloud edge drops, and generating physically impossible outputs (negative power at night).
3. **Third-Generation Physics-Informed Neural Networks (2024–Present):** Hybrid architectures embedding governing semiconductor and thermodynamic differential equations into the neural loss function ($\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda \mathcal{L}_{\text{physics}}$).

---

## 🔬 2. Foundational Research Papers

Based on our survey, three primary peer-reviewed papers serve as the scientific foundation for this project:

### Paper 1: PKINN Baseline Architecture
* **Citation:** Pei et al., *"Photovoltaic Knowledge Informed Neural Network (PKINN) for Power Prediction Under Volatile Weather"*, **Energy and AI** (Elsevier), 2026. [ScienceDirect Link](https://www.sciencedirect.com/science/article/pii/S2666546826000091)
* **Core Contribution:** Proves that embedding the Single-Diode Model (SDM) circuit equation ($\mathcal{L}_{\text{diode}}$) directly into the neural loss function prevents unphysical over-smoothing during rapid weather fluctuations.
* **Our Empirical Reproduction:**
  * Unconstrained Baseline Deep Learning: $R^2 = 0.3379$ ($\text{MAE} = 16.84\text{ W}$)
  * Single-Branch PKINN ($\mathcal{L}_{\text{QEM}}$): $R^2 = 0.8011$ ($\text{MAE} = 6.62\text{ W}$)
  * 3-Branch FAM-PKINN Ensemble: $R^2 = 0.8424$ ($\text{MAE} = 4.85\text{ W}$, $\text{RMSE} = 10.06\text{ W}$) $\rightarrow$ **71.2% Error Drop!**

### Paper 2: Arrhenius Degradation Kinetics
* **Citation:** Poddar et al., *"Accelerated degradation of photovoltaic modules under a future warmer climate"*, **Progress in Photovoltaics: Research and Applications**, 2024.
* **Core Contribution:** Formulates Arrhenius thermal stress kinetics to mathematically model how thermal cycling, humidity, and cumulative operating temperature degrade panel solder joints and increase series resistance ($R_s$) over time:
  
  $$\frac{dR_s}{dt} = A \cdot \exp\left( -\frac{E_a}{k_B \cdot T_{\text{cell}}} \right) \cdot (1 + \gamma_{\text{rh}} \cdot RH)$$

### Paper 3: Thermal Sensitivity & Convective Cooling
* **Citation:** Wang et al., *"Efficient estimation of convection cooling of photovoltaic arrays for thermal management"*, **Energy and AI**, 2025.
* **Core Contribution:** Quantifies heat dissipation balance on solar panel surface. Shows that panel junction temperature ($T_{\text{cell}}$) scales with irradiance ($G$) and is cooled by ambient wind velocity ($v_w$):
  
  $$T_{\text{cell}} = T_{\text{amb}} + \frac{\text{POA}}{u_0 + u_1 \cdot v_w}$$

---

## ⚖️ 3. Technology Comparison Matrix

| Attribute / Feature | Pure Data-Driven AI (LSTM / MLP) | Traditional Differential Solvers | Physics-Informed Neural Networks (PINNs) |
| :--- | :--- | :--- | :--- |
| **Inference Speed** | ⚡ Real-time ($<1\text{ ms}$) | 🐌 Slow numerical iterations | ⚡ Real-time ($<2\text{ ms}$) |
| **Physical Interpretability** | ❌ None ("Black Box") | 🎯 100% Deterministic | 🎯 High (Inverse State Diagnostics) |
| **Safety Under Ramps** | ❌ Predicts negative / impossible power | 🎯 Strictly bounded | 🎯 Strictly physically bounded |
| **Needs Internal Parameters** | ❌ No | ⚠️ Yes (Requires measuring $R_s, I_0, n$) | 🎯 **No (Estimates unobservables inverse-wise)** |
| **Cross-Asset Transferability** | ❌ Low (Overfits site quirks) | ⚠️ Moderate (Needs recalibration) | 🏆 **High ($R^2 > 0.90$ zero-shot transfer)** |

---

## 💡 4. Identification of Research Gap

While existing literature covers basic PINNs, **a critical research gap remains**:

> **The Research Gap:** Existing PINN implementations treat all loss components on a uniform row-by-row timescale, leading to severe gradient conflicts between instantaneous weather ramps (seconds/minutes) and cumulative material degradation (months/years). Furthermore, there is a lack of lightweight, real-time physics-informed architectures capable of handling severe non-stationary weather transitions while simultaneously executing inverse health diagnostics ($R_s(t)$) and zero-shot cross-asset domain adaptation without retraining.

This project directly addresses this gap through **Decoupled 12-Month Macro Cumulative Physics**, **Physics-Preserving Feature Reduction (KPCA)**, and **Structural Wavelet-Analytical Circuit Hybridization**.
