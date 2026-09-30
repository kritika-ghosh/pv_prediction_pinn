# 🔬 Phase 2 — Theoretical Foundations & Core Concepts

> **Phase Focus:** In-depth solid-state semiconductor physics, solar cell operating principles, Single-Diode equivalent circuit modeling, mathematical proof of black-box AI limitations, and the core inductive bias theory behind Physics-Informed Neural Networks (PINNs).

---

## 📁 Files Included in This Phase Folder
* 📓 **[`PINN.ipynb`](PINN.ipynb):** Foundational Jupyter notebook exploring initial PINN formulation, data structures, and baseline symbolic physics loss prototyping.

---

## ☀️ 1. Semiconductor & Solar Cell Working Principles

A photovoltaic (PV) solar cell is a solid-state semiconductor device (typically monocrystalline or multicrystalline silicon) containing a **p-n junction**.

```
Sunlight Photons (hν)
      │
      ▼
┌──────────────┐  <-- N-Type Layer (Emitter)
├──────────────┤  <-- Depletion Region (Built-in Electric Field)
└──────────────┘  <-- P-Type Layer (Base)
      │
      ▼
Electron-Hole Pairs Separated -> Photocurrent (I_ph)
```

1. **Photovoltaic Conversion:** When incoming photons with energy greater than the silicon bandgap ($E_g \approx 1.12\text{ eV}$ at $300\text{ K}$) strike the cell, electrons are excited from the valence band to the conduction band, creating electron-hole pairs.
2. **Built-in Electric Field:** The built-in potential across the p-n junction drives excited electrons toward the n-side and holes toward the p-side, generating a direct photocurrent ($I_{\text{ph}}$).
3. **Operational Environmental Dynamics:**
   * **Solar Irradiance ($G$ / $\text{POA}$):** Governs photocurrent generation ($I_{\text{ph}} \propto G$). Higher irradiance directly increases generated output current.
   * **Junction Temperature ($T_{\text{cell}}$):** Causes the semiconductor bandgap to narrow slightly ($dE_g/dT < 0$), slightly increasing photocurrent but drastically increasing diode saturation current ($I_0$), causing open-circuit voltage ($V_{\text{oc}}$) and output power efficiency to drop ($\gamma_p \approx -0.4\%/^\circ\text{C}$).

---

## ⚡ 2. The Single-Diode Equivalent Circuit Model (SDM)

To model an operational solar cell or module electrically, electrical engineers use the **Single-Diode Equivalent Circuit**:

```
           I_ph           D (Diode)         R_sh           R_s
   ┌─────────┼──────────────┬────────────────┼────────────████──────┐ (+)
   │         │              │                │                      │
  (↑)       ---            / \              ████                   [LOAD]
 I_ph       \ / Diode      ---              ████ R_sh               │
   │         │              │                │                      │
   └─────────┴──────────────┴────────────────┴──────────────────────┘ (-)
```

Applying Kirchhoff's Current Law (KCL) to the top circuit node yields the fundamental **Shockley Single-Diode Equation**:

$$I = I_{\text{ph}} - I_0 \left[ \exp\left( \frac{V + I R_s}{n V_t} \right) - 1 \right] - \frac{V + I R_s}{R_{\text{sh}}}$$

Where:
* $I_{\text{ph}}$: Light-generated photocurrent ($\text{A}$).
* $I_0$: Reverse diode saturation current ($\text{A}$).
* $n$: Diode ideality factor ($1.0 \le n \le 2.0$ for silicon).
* $V_t = \frac{k_B T_{\text{cell}}}{q}$: Thermal voltage ($\text{V}$), where $k_B$ is Boltzmann's constant and $q$ is electron charge.
* $R_s$: Internal series resistance ($\Omega$), representing contact grid and solder joint resistance.
* $R_{\text{sh}}$: Shunt resistance ($\Omega$), representing PN junction leakage currents.

---

## 🤖 3. Limitations of Pure Black-Box Machine Learning

Standard deep learning architectures (MLPs, LSTMs, Transformers) treat solar forecasting purely as a statistical curve-fitting problem ($y = f(x; \theta)$).

### Why Pure AI Fails Under Real-World Weather Drift:
1. **No Understanding of Energy Conservation:** Pure neural networks can output negative power at night ($P < 0$) or output power exceeding the solar constant ($P > G \cdot A$).
2. **Lag Behind Cloud Transients:** Standard mean squared error ($\text{MSE}$) losses penalize extreme predictions, causing black-box models to output over-smoothed averages during rapid cumulus cloud coverage.
3. **Rote-Memorization & Overfitting:** When given unconstrained parameter capacity, deep models memorizing specific training telemetry numbers fail when evaluated on unseen seasonal weather or different geographical sites.

---

## 🛡️ 4. Why Physics-Informed Neural Networks (PINNs)?

**Physics-Informed Neural Networks (PINNs)** eliminate black-box failures by embedding governing physical equations directly into the neural network's loss function.

```
                           ┌────────────────────────┐
                           │   Input Weather (X)    │
                           └───────────┬────────────┘
                                       │
                                       ▼
                           ┌────────────────────────┐
                           │ Neural Network Weights │
                           └───────────┬────────────┘
                                       │
                ┌──────────────────────┴──────────────────────┐
                ▼                                             ▼
   ┌─────────────────────────┐                   ┌─────────────────────────┐
   │ Empirical Supervised    │                   │ Physical Loss Penalties  │
   │ Loss (L_data)           │                   │ (L_sde, L_thermal, etc.)│
   └────────────┬────────────┘                   └────────────┬────────────┘
                │                                             │
                └──────────────────────┬──────────────────────┘
                                       ▼
                     Total Objective: L = L_data + λ L_physics
```

By adding physical loss penalties ($\lambda \mathcal{L}_{\text{physics}}$), the neural network's optimization search space is strictly constrained to weight combinations that comply with the laws of semiconductor physics and thermodynamics.
