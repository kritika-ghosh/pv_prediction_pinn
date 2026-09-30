
def resolve_dataset_path(csv_name):
    import os
    if os.path.exists(csv_name):
        return csv_name
    candidates = [
        os.path.join("datasets", csv_name),
        os.path.join("..", csv_name),
        os.path.join("..", "datasets", csv_name),
        os.path.join("..", "..", "datasets", csv_name),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return csv_name

import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

# Set device and seeds
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

# -------------------------------------------------------------
# 1. Dataset & Windowing Preprocessor
# -------------------------------------------------------------
class PVDailySequenceDataset(Dataset):
    def __init__(self, X, y, seq_len=12):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        return (
            self.X[idx : idx + self.seq_len],
            self.y[idx + self.seq_len],
            self.X[idx + self.seq_len],  # Current step physical features
        )


def prepare_pv_data(csv_path="xSi12922.csv", seq_len=12, split_ratio=0.8):
    df = pd.read_csv(csv_path)

    # Core operational & environmental features
    features = [
        "POA",
        "Dry bulb temperature (degC)",
        "Relative humidity (%RH)",
        "Atmospheric pressure (mb)",
    ]
    target = "Pmp (W)"

    # Clean non-positive/night rows
    df = df[df[target] > 0.5].reset_index(drop=True)

    X_raw = df[features].values
    y_raw = df[target].values.reshape(-1, 1)

    split_idx = int(len(df) * split_ratio)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_X.fit_transform(X_raw[:split_idx])
    y_train = scaler_y.fit_transform(y_raw[:split_idx])

    X_test = scaler_X.transform(X_raw[split_idx:])
    y_test = scaler_y.transform(y_raw[split_idx:])

    train_ds = PVDailySequenceDataset(X_train, y_train, seq_len)
    test_ds = PVDailySequenceDataset(X_test, y_test, seq_len)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    return train_loader, test_loader, scaler_X, scaler_y


# -------------------------------------------------------------
# 2. Multi-Head LSTM Model
# -------------------------------------------------------------
class PVMultiHeadLSTM(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1
        )
        self.fc_shared = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
        )
        # Head 1: Normalized Power Output (P_hat)
        self.head_power = nn.Linear(32, 1)
        # Head 2: Cell Temperature in Celsius (T_cell_hat)
        self.head_thermal = nn.Linear(32, 1)
        # Head 3: Instantaneous Degradation Rate (dD_hat / dt)
        self.head_aging = nn.Sequential(nn.Linear(32, 1), nn.Softplus())

    def forward(self, x):
        out, _ = self.lstm(x)
        features = self.fc_shared(out[:, -1, :])
        p_hat = self.head_power(features)
        t_cell_hat = self.head_thermal(features)
        dD_dt_hat = self.head_aging(features)
        return p_hat, t_cell_hat, dD_dt_hat


# -------------------------------------------------------------
# 3. Modular Physics Loss Formulations
# -------------------------------------------------------------
class MultiPhysicsLossEngine:
    def __init__(self, scaler_X, scaler_y):
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        self.mse = nn.MSELoss()

        # Unscale constants for physical domain
        self.poa_mean = scaler_X.mean_[0]
        self.poa_scale = scaler_X.scale_[0]
        self.tamb_mean = scaler_X.mean_[1]
        self.tamb_scale = scaler_X.scale_[1]
        self.rh_mean = scaler_X.mean_[2]
        self.rh_scale = scaler_X.scale_[2]
        self.p_mean = scaler_y.mean_[0]
        self.p_scale = scaler_y.scale_[0]

        # Physical coefficients
        self.eta_stc = 0.16  # Baseline conversion efficiency
        self.gamma_p = -0.004  # Power temperature coefficient (-0.4%/C)
        self.area = 0.6  # Panel aperture area (m^2)
        self.u0 = 26.9  # Faiman heat transfer coeff constant
        self.u1 = 1.06  # Faiman wind velocity coeff
        self.v_w = 1.5  # Constant wind velocity proxy (m/s)
        self.A_arrh = 1e-4  # Kinetic pre-exponential factor
        self.E_a = 0.35  # Apparent activation energy (eV)
        self.k_B = 8.617333e-5  # Boltzmann constant (eV/K)
        self.gamma_rh = 0.005  # Humidity degradation acceleration

    def loss_data(self, p_hat_norm, y_norm):
        return self.mse(p_hat_norm, y_norm)

    def loss_sde(self, p_hat_norm, current_x_norm, t_cell_hat):
        """Single-Diode / Power Conversion Envelope constraint."""
        # Denormalize POA
        poa = current_x_norm[:, 0:1] * self.poa_scale + self.poa_mean
        poa = torch.clamp(poa, min=0.0)

        # Theoretical power baseline P_QEM
        p_ref = (
            poa
            * self.area
            * self.eta_stc
            * (1.0 + self.gamma_p * (t_cell_hat - 25.0))
        )
        p_ref = torch.clamp(p_ref, min=0.0)

        # Denormalize predicted power
        p_pred = p_hat_norm * self.p_scale + self.p_mean
        p_loss = torch.mean((p_pred - p_ref) ** 2) / (self.p_scale**2)
        relu_penalty = torch.mean(torch.relu(-p_pred))
        return p_loss + relu_penalty

    def loss_thermal(self, t_cell_hat, current_x_norm):
        """First Law Thermodynamic heat dissipation equilibrium."""
        poa = current_x_norm[:, 0:1] * self.poa_scale + self.poa_mean
        t_amb = current_x_norm[:, 1:2] * self.tamb_scale + self.tamb_mean

        t_expected = t_amb + (poa / (self.u0 + self.u1 * self.v_w))
        return torch.mean((t_cell_hat - t_expected) ** 2) / 100.0

    def loss_arrhenius(self, dD_dt_hat, t_cell_hat, current_x_norm):
        """Arrhenius hydrolytic & thermal kinetic drift constraint."""
        rh = current_x_norm[:, 2:3] * self.rh_scale + self.rh_mean
        rh = torch.clamp(rh, min=0.0, max=100.0)
        t_kelvin = t_cell_hat + 273.15

        degrad_target = (
            self.A_arrh
            * torch.exp(-self.E_a / (self.k_B * t_kelvin))
            * (1.0 + self.gamma_rh * rh)
        )
        return torch.mean((dD_dt_hat - degrad_target) ** 2) * 1e6


# -------------------------------------------------------------
# 4. Combinatorial Training & Evaluation Loop
# -------------------------------------------------------------
def train_and_evaluate(active_losses, train_loader, test_loader, loss_engine):
    model = PVMultiHeadLSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = 6

    # Fixed regularization weights
    weights = {"sde": 0.1, "thermal": 0.05, "arrhenius": 0.01}

    model.train()
    for _ in range(epochs):
        for seq_x, y_target, curr_x in train_loader:
            seq_x = seq_x.to(device)
            y_target = y_target.to(device)
            curr_x = curr_x.to(device)

            optimizer.zero_grad()
            p_hat, t_hat, dD_hat = model(seq_x)

            # Compute selected losses
            total_loss = torch.tensor(0.0, device=device)

            if "data" in active_losses:
                total_loss += loss_engine.loss_data(p_hat, y_target)
            if "sde" in active_losses:
                total_loss += weights["sde"] * loss_engine.loss_sde(
                    p_hat, curr_x, t_hat
                )
            if "thermal" in active_losses:
                total_loss += weights["thermal"] * loss_engine.loss_thermal(
                    t_hat, curr_x
                )
            if "arrhenius" in active_losses:
                total_loss += weights["arrhenius"] * loss_engine.loss_arrhenius(
                    dD_hat, t_hat, curr_x
                )

            total_loss.backward()
            optimizer.step()

    # Evaluation on test set
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for seq_x, y_target, _ in test_loader:
            seq_x = seq_x.to(device)
            p_hat, _, _ = model(seq_x)
            preds.append(p_hat.cpu().numpy())
            actuals.append(y_target.numpy())

    preds = np.vstack(preds)
    actuals = np.vstack(actuals)

    # Denormalize to true Watts before calculating R2
    preds_w = preds * loss_engine.p_scale + loss_engine.p_mean
    actuals_w = actuals * loss_engine.p_scale + loss_engine.p_mean

    return r2_score(actuals_w, preds_w)


# -------------------------------------------------------------
# 5. Run All Combinations (Singles, Pairs, Triplets, All-4)
# -------------------------------------------------------------
if __name__ == "__main__":
    train_loader, test_loader, scaler_X, scaler_y = prepare_pv_data(
        "xSi12922.csv"
    )
    engine = MultiPhysicsLossEngine(scaler_X, scaler_y)

    loss_terms = ["data", "sde", "thermal", "arrhenius"]
    combinations = []

    # Generate all sub-combinations (size 1 to 4)
    for k in range(1, 5):
        combinations.extend(list(itertools.combinations(loss_terms, k)))

    results = []
    print(f"{'Experiment':<5} | {'Active Losses':<40} | {'Test R² Score':<10}")
    print("-" * 65)

    for i, combo in enumerate(combinations, start=1):
        score = train_and_evaluate(combo, train_loader, test_loader, engine)
        combo_str = " + ".join(combo)
        results.append((combo_str, score))
        print(f"{i:<5} | {combo_str:<40} | {score:10.4f}")