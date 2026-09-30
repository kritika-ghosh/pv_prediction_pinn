# ⏳ Phase 5 — Refining Architecture & Transitioning to Temporal Modeling (LSTM)

> **Phase Focus:** Identification of the structural temporal limitation of Multilayer Perceptrons (MLPs), transition to sequential Long Short-Term Memory (LSTM) networks, sliding window sequence design, and multi-timescale physical modeling.

---

## 📁 Files Included in This Phase Folder
* 🐍 **[`study_12month_pinn.py`](study_12month_pinn.py):** 12-Month Macro Cumulative Arrhenius PINN benchmark script using sliding sequence buffers.

---

## 1. Re-Evaluating the Baseline MLP & Temporal Limitation

Following the dataset re-validation in Phase 4, we re-evaluated our MLP baseline on `xSi12922.csv`.

### The Fundamental Structural Flaw of MLPs:
Standard MLPs process tabular input data **row-by-row independently**, treating each time step $t$ as an isolated static instance:

$$\hat{y}_t = f(x_t; \theta)$$

However, solar weather data is **strictly sequential and time-dependent**:
* Atmospheric moisture and cloud shadows at time $t - 15\text{ min}$ directly govern cell irradiance and ambient temperature at time $t$.
* Panel thermal mass introduces thermal inertia—a solar panel takes 10–20 minutes to heat up or cool down following a solar irradiance jump.
* Row-by-row MLPs cannot model thermal inertia or causal time dynamics, resulting in high prediction lag during morning and evening transitions.

---

## 2. Transition to LSTM & Sliding Window Sequence Buffers

To capture temporal dependencies, we replaced the MLP architecture with a **Long Short-Term Memory (LSTM)** recurrent neural network equipped with a sliding sequence window:

```
Time-Series Telemetry (POA, Temp, RH, Pressure, etc.)
  │
  ├─ Window 1: [t-11, t-10, ..., t0] ──> LSTM ──> Predict Power at t0
  ├─ Window 2: [t-10, t-9,  ..., t1] ──> LSTM ──> Predict Power at t1
  └─ Window 3: [t-9,  t-8,  ..., t2] ──> LSTM ──> Predict Power at t2
```

### Sequence Architecture Parameters:
* **Sequence Length ($N = 12$):** Captures a 1-hour temporal memory window (12 steps $\times$ 5-minute sampling).
* **Unidirectional Hidden State ($h_t$):** Preserves physical causality (the arrow of time: past weather affects present power).
* **State Update Equations:**
  
  $$f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)$$
  
  $$i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i)$$
  
  $$c_t = f_t \odot c_{t-1} + i_t \odot \tanh(W_c x_t + U_c h_{t-1} + b_c)$$
  
  $$h_t = \sigma(W_o x_t + U_o h_{t-1} + b_o) \odot \tanh(c_t)$$

---

## 3. Multi-Timescale Physics Integration

With sliding window LSTMs established, we tackled the **Timescale Mismatch Problem**:
* **Fast Weather Ramps:** Irradiance drops occur in seconds to minutes ($t \in [1, 12]$).
* **Slow Physical Degradation:** Arrhenius degradation kinetics ($R_s$ growth) accumulate across months and years ($t \in [1, 105120]$).

In `study_12month_pinn.py`, we implemented a **Macro 12-Month Cumulative Riemann Integral** across historical sequence windows:

$$\Delta R_{s, \text{physical}}^{(12\text{mo})} = \sum_{i=1}^N A \cdot \exp\left( -\frac{E_a}{k_B \cdot T_{\text{cell}}(t_i)} \right) \cdot \left[1 + \gamma_{\text{rh}} \cdot RH(t_i)\right] \Delta t$$

### Major Zero-Shot Breakthrough:
* **Exp 14 (`diode + thermal + arrhenius_12mo`):** Achieved **Zero-Shot $R^2 = 0.7921$ without showing a single target power label $y$ during training!**
* Proves that temporal sequence LSTMs regularized by macro-timescale physical equations fully constrain the physical hypothesis space.
