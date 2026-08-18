import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.autograd import grad
import torch.nn as nn
import torch.optim as optim

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================================
# 1. DATA INGESTION & NORMALIZATION
# ==========================================================
dataset_path = "pinn_training_dataset.csv"
df = pd.read_csv(dataset_path)

# Parse time and derive continuous normalized time [0, 1]
df["time"] = pd.to_datetime(df["time"], utc=True)
df.sort_values("time", inplace=True)

t_sec = (df["time"] - df["time"].iloc[0]).dt.total_seconds().values
t_norm_val = t_sec.max() if t_sec.max() > 0 else 1.0
t_normalized = t_sec / t_norm_val

# Extract weather & operational surface telemetry
G = df["poa_global"].values  # W/m^2
Tamb = df["temp_air"].values + 273.15  # Kelvin
wind = df["wind_speed"].values  # m/s
V = df["V_mp"].values  # Volts
I = df["I_mp"].values  # Amps
P = df["P_mp"].values  # Watts
Tcell_meas = df["temp_cell"].values + 273.15  # Kelvin

# Convert to PyTorch Tensors
t_t = (
    torch.tensor(t_normalized, dtype=torch.float32, device=device)
    .unsqueeze(1)
    .requires_grad_(True)
)
G_t = torch.tensor(G, dtype=torch.float32, device=device).unsqueeze(1)
Tamb_t = torch.tensor(Tamb, dtype=torch.float32, device=device).unsqueeze(1)
wind_t = torch.tensor(wind, dtype=torch.float32, device=device).unsqueeze(1)
V_t = torch.tensor(V, dtype=torch.float32, device=device).unsqueeze(1)
I_t = torch.tensor(I, dtype=torch.float32, device=device).unsqueeze(1)
P_t = torch.tensor(P, dtype=torch.float32, device=device).unsqueeze(1)
Tcell_t = torch.tensor(Tcell_meas, dtype=torch.float32, device=device).unsqueeze(
    1
)


# ==========================================================
# 2. MULTI-PHYSICS PINN ARCHITECTURE (Digital Twin)
# ==========================================================
class PVMultiPhysicsPINN(nn.Module):

  def __init__(self):
    super(PVMultiPhysicsPINN, self).__init__()
    # Ingests [t, G, T_ambient, wind_speed, V, I]
    self.net = nn.Sequential(
        nn.Linear(6, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, 4),  # Predicts: [P(t), T_cell(t), R_s(t), R_sh(t)]
    )

  def forward(self, t, G, T_amb, wind, V, I):
    # Normalized dimensionless inputs
    inputs = torch.cat(
        [
            t,
            G / 1000.0,
            (T_amb - 273.15) / 50.0,
            wind / 10.0,
            V / 40.0,
            I / 10.0,
        ],
        dim=1,
    )
    out = self.net(inputs)

    # Physically bounded outputs
    P_pred = torch.sigmoid(out[:, 0:1]) * 250.0  # Power [0, 250] W
    T_cell_pred = (
        273.15 + 15.0 + torch.sigmoid(out[:, 1:2]) * 55.0
    )  # Temp [15°C, 70°C]
    R_s_pred = (
        0.1 + torch.sigmoid(out[:, 2:3]) * 1.2
    )  # Series Resistance [0.1, 1.3] Ohm[cite: 1]
    R_sh_pred = (
        50.0 + torch.sigmoid(out[:, 3:4]) * 800.0
    )  # Shunt Resistance [50, 850] Ohm[cite: 1]

    return P_pred, T_cell_pred, R_s_pred, R_sh_pred


model = PVMultiPhysicsPINN().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
mse = nn.MSELoss()

# ==========================================================
# 3. PHYSICAL CONSTANTS (STC Calibrated)
# ==========================================================
# Electrical Single-Diode Parameters (Module scale)
a_ref = 1.6234  # Diode thermal voltage modifier (V)
I_ph_ref = 8.87  # STC Photocurrent (A)
I_0_ref = 2.1e-9  # STC Reverse saturation current (A)
alpha_Isc = 0.0048  # A/K
k_B_J = 1.380649e-23  # J/K
E_g = 1.121 * 1.60218e-19  # Silicon bandgap in Joules

# Thermal Heat Dissipation Parameters
C_th = 20000.0  # Heat capacity J / (K m^2)
A_mod = 1.63  # Module area (m^2)
alpha_abs = 0.9  # Absorptivity

# Arrhenius Aging Parameters
T_ref = 298.15  # Reference STC temp (25°C)
E_a_eV = 0.85  # Activation energy (eV)[cite: 1]
k_B_eV = 8.617333e-5  # Boltzmann constant (eV/K)
nominal_annual_degradation = 0.02  # Nominal dRs/dt (~0.02 Ohm/year)[cite: 1]

# Loss weighting coefficients
lambda_elec = 0.10
lambda_thermal = 0.05
lambda_aging = 1.00

# ==========================================================
# 4. TRAINING LOOP WITH COMPOSITE LOSS
# ==========================================================
EPOCHS = 500
print("Starting Multi-Physics PINN Digital Twin Training...\n")

for epoch in range(1, EPOCHS + 1):
  optimizer.zero_grad()

  # Forward pass
  P_pred, T_cell_pred, R_s_pred, R_sh_pred = model(
      t_t, G_t, Tamb_t, wind_t, V_t, I_t
  )

  # 1. Supervised Data Loss (L_data)
  L_data_P = mse(P_pred, P_t) / (100.0**2)
  L_data_T = mse(T_cell_pred, Tcell_t) / (20.0**2)
  L_data = L_data_P + L_data_T

  # 2. Electrical Single-Diode Constraint (L_elec)[cite: 1]
  a_T = a_ref * (T_cell_pred / 298.15)
  I_ph = (G_t / 1000.0) * (I_ph_ref + alpha_Isc * (T_cell_pred - 298.15))
  T_ratio = T_cell_pred / 298.15
  I_0 = (
      I_0_ref
      * (T_ratio**3)
      * torch.exp((E_g / (k_B_J * 298.15)) * (1.0 - 1.0 / T_ratio))
  )

  diode_v = V_t + I_t * R_s_pred
  exp_arg = torch.clamp(diode_v / a_T, min=-20.0, max=25.0)  # Prevents overflow
  I_calc = I_ph - I_0 * (torch.exp(exp_arg) - 1.0) - (diode_v / R_sh_pred)

  day_mask = (G_t > 20.0).float()
  L_elec = torch.mean(torch.square(day_mask * (I_t - I_calc))) / (5.0**2)

  # 3. Dynamic Thermal Energy Balance Constraint (L_thermal)[cite: 1]
  dT_dt = (
      grad(
          T_cell_pred,
          t_t,
          grad_outputs=torch.ones_like(T_cell_pred),
          create_graph=True,
          retain_graph=True,
      )[0]
      / t_norm_val
  )

  h_loss = 5.7 + 3.8 * wind_t  # Convective dissipation
  Q_in = alpha_abs * G_t
  P_density = P_pred / A_mod
  thermal_residual = (C_th * dT_dt) - (
      Q_in - P_density - h_loss * (T_cell_pred - Tamb_t)
  )
  L_thermal = torch.mean(torch.square(thermal_residual / 500.0))

  # 4. Arrhenius Physical Aging ODE Constraint (L_aging)[cite: 1]
  dRs_dt_norm = grad(
      R_s_pred,
      t_t,
      grad_outputs=torch.ones_like(R_s_pred),
      create_graph=True,
      retain_graph=True,
  )[0]
  arrhenius_acceleration = torch.exp(
      (-E_a_eV / k_B_eV) * ((1.0 / T_cell_pred) - (1.0 / T_ref))
  )
  target_aging_rate = (
      nominal_annual_degradation
      * arrhenius_acceleration
      * (t_norm_val / (365.25 * 86400.0))
  )
  L_aging = torch.mean(torch.square(dRs_dt_norm - target_aging_rate))

  # Total Multi-Physics Loss
  L_total = (
      L_data
      + (lambda_elec * L_elec)
      + (lambda_thermal * L_thermal)
      + (lambda_aging * L_aging)
  )

  # Backpropagation with gradient clipping
  L_total.backward()
  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
  optimizer.step()

  if epoch % 50 == 0 or epoch == 1:
    print(
        f"Epoch {epoch:3d}/{EPOCHS} | "
        f"L_total: {L_total.item():.5f} | "
        f"L_data: {L_data.item():.5f} | "
        f"L_elec: {L_elec.item():.4e} | "
        f"L_therm: {L_thermal.item():.4e} | "
        f"L_aging: {L_aging.item():.4e}"
    )

# ==========================================================
# 5. MODEL EVALUATION & PARAMETER DISCOVERY
# ==========================================================
model.eval()
with torch.no_grad():
  P_pred, T_cell_pred, R_s_pred, R_sh_pred = model(
      t_t, G_t, Tamb_t, wind_t, V_t, I_t
  )

print("\n--- Digital Twin Evaluation ---")
print(
    f"Mean Absolute Power Error: {torch.mean(torch.abs(P_pred - P_t)).item():.2f} W"
)
print(
    f"Mean Absolute Cell Temp Error: {torch.mean(torch.abs(T_cell_pred - Tcell_t)).item():.2f} K"
)
print(
    f"Extracted Series Resistance R_s(t=0): {R_s_pred[0].item():.4f} Ohm[cite: 1]"
)
print(
    f"Extracted Series Resistance R_s(t=end): {R_s_pred[-1].item():.4f} Ohm[cite: 1]"
)
print(f"Extracted Shunt Resistance R_sh (Mean): {R_sh_pred.mean().item():.2f} Ohm[cite: 1]")

import matplotlib.pyplot as plt

# ==========================================================
# 6. VISUALIZATION & DIAGNOSTIC PLOTS
# ==========================================================
model.eval()
with torch.no_grad():
  P_eval, T_eval, Rs_eval, Rsh_eval = model(
      t_t, G_t, Tamb_t, wind_t, V_t, I_t
  )

# Convert tensors to numpy arrays
time_axis = df["time"]
p_pred_np = P_eval.cpu().numpy().flatten()
p_true_np = P
t_pred_np = T_eval.cpu().numpy().flatten() - 273.15  # Convert Kelvin to °C
t_true_np = df["temp_cell"].values
rs_pred_np = Rs_eval.cpu().numpy().flatten()

# Create 3-panel figure
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# 1. Power Output Comparison
axes[0].plot(
    time_axis,
    p_true_np,
    label="Ground Truth Power ($P_{mp}$)",
    color="black",
    alpha=0.6,
    linewidth=1.5,
)
axes[0].plot(
    time_axis,
    p_pred_np,
    label="PINN Predicted Power ($P_{pred}$)",
    color="crimson",
    linestyle="--",
    linewidth=1.5,
)
axes[0].set_ylabel("Power (W)", fontsize=11)
axes[0].set_title(
    "Multi-Physics PINN Digital Twin Performance",
    fontsize=14,
    fontweight="bold",
)
axes[0].grid(True, linestyle=":", alpha=0.6)
axes[0].legend(loc="upper right")

# 2. PV Cell Temperature Tracking
axes[1].plot(
    time_axis,
    t_true_np,
    label="Actual Cell Temp ($T_{cell}$)",
    color="tab:blue",
    alpha=0.7,
    linewidth=1.5,
)
axes[1].plot(
    time_axis,
    t_pred_np,
    label="PINN Predicted Temp ($T_{pred}$)",
    color="darkorange",
    linestyle="--",
    linewidth=1.5,
)
axes[1].set_ylabel("Cell Temp (°C)", fontsize=11)
axes[1].grid(True, linestyle=":", alpha=0.6)
axes[1].legend(loc="upper right")

# 3. Dynamic Series Resistance (Degradation State)
axes[2].plot(
    time_axis,
    rs_pred_np,
    label="Extracted $R_s(t)$ (Degradation Trajectory)",
    color="purple",
    linewidth=2,
)
axes[2].set_ylabel("Series Resistance ($\Omega$)", fontsize=11)
axes[2].set_xlabel("Timestamp", fontsize=11)
axes[2].grid(True, linestyle=":", alpha=0.6)
axes[2].legend(loc="upper left")

plt.tight_layout()
plt.savefig("pinn_digital_twin_results.png", dpi=300)
plt.show()