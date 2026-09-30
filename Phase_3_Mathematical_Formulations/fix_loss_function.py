
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)


# -------------------------------------------------------------
# 1. Temporal Feature Engineering & Time-Series Windowing
# -------------------------------------------------------------
def build_temporal_features(df):
    dt = pd.to_datetime(df["date"])
    df["dt"] = dt
    df["hour"] = dt.dt.hour + dt.dt.minute / 60.0
    df["day_of_year"] = dt.dt.dayofyear
    df["day_id"] = dt.dt.date

    # Cyclic diurnal and seasonal encodings
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["sin_day"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["cos_day"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    return df


class PVTimeSeriesDataset(Dataset):
    def __init__(self, sequences_x, targets_y, current_x):
        self.X = torch.tensor(sequences_x, dtype=torch.float32)
        self.y = torch.tensor(targets_y, dtype=torch.float32)
        self.curr_x = torch.tensor(current_x, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.curr_x[idx]


def prepare_time_series_data(
    csv_path="xSi12922.csv", seq_len=12, split_ratio=0.8
):
    df = pd.read_csv(csv_path)
    df = build_temporal_features(df)

    # 4 Environmental + 4 Temporal Encodings = 8 time-series features
    feature_cols = [
        "POA",
        "Dry bulb temperature (degC)",
        "Relative humidity (%RH)",
        "Atmospheric pressure (mb)",
        "sin_hour",
        "cos_hour",
        "sin_day",
        "cos_day",
    ]
    target_col = "Pmp (W)"

    # Chronological date split
    unique_days = df["day_id"].unique()
    split_point = int(len(unique_days) * split_ratio)
    train_days = set(unique_days[:split_point])
    test_days = set(unique_days[split_point:])

    df_train = df[df["day_id"].isin(train_days)].copy()
    df_test = df[df["day_id"].isin(test_days)].copy()

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    scaler_X.fit(df_train[feature_cols].values)
    scaler_y.fit(df_train[[target_col]].values)

    def extract_day_windows(sub_df):
        seqs, targets, curr_features = [], [], []
        # Group by day so window slices never cross midnight
        for _, day_group in sub_df.groupby("day_id"):
            if len(day_group) <= seq_len:
                continue

            x_scaled = scaler_X.transform(day_group[feature_cols].values)
            y_scaled = scaler_y.transform(day_group[[target_col]].values)

            for i in range(len(day_group) - seq_len):
                seqs.append(x_scaled[i : i + seq_len])
                targets.append(y_scaled[i + seq_len])
                curr_features.append(x_scaled[i + seq_len])

        return np.array(seqs), np.array(targets), np.array(curr_features)

    X_train_seq, y_train_seq, curr_train = extract_day_windows(df_train)
    X_test_seq, y_test_seq, curr_test = extract_day_windows(df_test)

    train_ds = PVTimeSeriesDataset(X_train_seq, y_train_seq, curr_train)
    test_ds = PVTimeSeriesDataset(X_test_seq, y_test_seq, curr_test)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    return train_loader, test_loader, scaler_X, scaler_y


# -------------------------------------------------------------
# REVISED 2. Time-Series Multi-Head LSTM Model (Physically Coupled)
# -------------------------------------------------------------
class PVTimeSeriesLSTM(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1
        )
        self.fc_shared = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.Tanh(),  # Tanh is better for tracking continuous physical states than ReLU
        )

        # State Heads
        self.head_thermal = nn.Linear(32, 1)  # Predicts T_cell directly in Celsius

        # We predict a Cumulative Degradation State D (bounded strictly between 0 and 1)
        self.head_degradation = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # Residual tracking for the derivative dD/dt
        self.head_aging_rate = nn.Sequential(
            nn.Linear(32, 1),
            nn.Softplus(),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        context = self.fc_shared(out[:, -1, :])

        t_cell_hat = self.head_thermal(context)
        D_cumulative_hat = self.head_degradation(context)
        dD_dt_hat = self.head_aging_rate(context)

        return t_cell_hat, D_cumulative_hat, dD_dt_hat


# -------------------------------------------------------------
# REVISED 3. Physics Loss Engine (Strictly Bounded & Normalized)
# -------------------------------------------------------------
class MultiPhysicsLossEngine:
    def __init__(self, scaler_X, scaler_y):
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        self.mse = nn.MSELoss()

        # Unscale variables accurately
        self.poa_mean = scaler_X.mean_[0]
        self.poa_scale = scaler_X.scale_[0]
        self.tamb_mean = scaler_X.mean_[1]
        self.tamb_scale = scaler_X.scale_[1]
        self.rh_mean = scaler_X.mean_[2]
        self.rh_scale = scaler_X.scale_[2]
        self.p_mean = scaler_y.mean_[0]
        self.p_scale = scaler_y.scale_[0]

        # Domain physics coefficients (Standardized SI Units)
        self.eta_stc = 0.16
        self.gamma_p = -0.004  # Temperature coefficient (%/degC)
        self.area = 0.6
        self.u0 = 26.9
        self.u1 = 1.06
        self.v_w = 1.5

        # Arrhenius Constants calibrated for eV calculations
        self.A_arrh = 1e-4
        self.E_a = 0.35
        self.k_B = 8.617333e-5  # eV/K
        self.gamma_rh = 0.005

    def forward_physics_power(self, poa, t_cell_hat, D_cumulative_hat):
        """Calculates physical power explicitly bounded by the Single-Diode formulation
        and directly penalised by the degradation state factor."""
        poa_cleared = torch.clamp(poa, min=0.0)

        # Base physical power capacity matching Shockley/QEM envelope
        p_ideal = (
            poa_cleared
            * self.area
            * self.eta_stc
            * (1.0 + self.gamma_p * (t_cell_hat - 25.0))
        )
        p_ideal = torch.clamp(p_ideal, min=0.0)

        # D_cumulative_hat = 0 means healthy, 1 means dead asset.
        # Forces coupling: High degradation structurally suppresses output power capacity.
        p_physical = (1.0 - D_cumulative_hat) * p_ideal
        return p_physical

    def loss_data(self, t_cell_hat, D_cumulative_hat, current_x_norm, y_norm):
        """Calculates structural data error using the physical equation forward pass."""
        poa = current_x_norm[:, 0:1] * self.poa_scale + self.poa_mean

        # Generate prediction through the physical laws
        p_physical = self.forward_physics_power(
            poa, t_cell_hat, D_cumulative_hat
        )

        # Normalize back to scale cleanly with y_norm
        p_physical_norm = (p_physical - self.p_mean) / self.p_scale

        # Primary Data Loss + Night-time Negative Check
        data_mse = self.mse(p_physical_norm, y_norm)
        relu_penalty = torch.mean(torch.relu(-p_physical))

        return data_mse + relu_penalty

    def loss_thermal(self, t_cell_hat, current_x_norm):
        """Enforces heat balance without arbitrary divisions."""
        poa = current_x_norm[:, 0:1] * self.poa_scale + self.poa_mean
        t_amb = current_x_norm[:, 1:2] * self.tamb_scale + self.tamb_mean

        # Thermodynamic expectation
        t_expected = t_amb + (poa / (self.u0 + self.u1 * self.v_w))

        # Normalized loss based on a standard summer temperature variance expectation (~10 degC)
        return torch.mean((t_cell_hat - t_expected) ** 2) / 100.0

    def loss_arrhenius(self, dD_dt_hat, t_cell_hat, current_x_norm):
        """Enforces physical kinetics via automatic Kelvin mapping."""
        rh = current_x_norm[:, 2:3] * self.rh_scale + self.rh_mean
        rh = torch.clamp(rh, min=0.0, max=100.0)

        # Absolute Temperature conversion protects against vanishing gradients
        t_kelvin = t_cell_hat + 273.15
        t_kelvin = torch.clamp(t_kelvin, min=200.0)  # Lower bound safety floor

        degrad_target = (
            self.A_arrh
            * torch.exp(-self.E_a / (self.k_B * t_kelvin))
            * (1.0 + self.gamma_rh * rh)
        )

        # Log-space MSE normalization prevents gradient explosion/nan failures from 1e6 scalers
        log_pred = torch.log1p(dD_dt_hat)
        log_target = torch.log1p(degrad_target)
        return self.mse(log_pred, log_target)


# -------------------------------------------------------------
# 4. Training & Combinatoric Evaluation Loop
# -------------------------------------------------------------
def train_and_evaluate(active_losses, train_loader, test_loader, loss_engine):
    model = PVTimeSeriesLSTM(input_dim=8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = 6

    weights = {"thermal": 0.05, "arrhenius": 0.01}

    model.train()
    for _ in range(epochs):
        for seq_x, y_target, curr_x in train_loader:
            seq_x, y_target, curr_x = (
                seq_x.to(device),
                y_target.to(device),
                curr_x.to(device),
            )
            optimizer.zero_grad()
            t_hat, D_hat, dD_dt_hat = model(seq_x)

            total_loss = torch.tensor(0.0, device=device)
            if "data" in active_losses:
                total_loss += loss_engine.loss_data(
                    t_hat, D_hat, curr_x, y_target
                )
            if "thermal" in active_losses:
                total_loss += weights[
                    "thermal"
                ] * loss_engine.loss_thermal(t_hat, curr_x)
            if "arrhenius" in active_losses:
                total_loss += weights[
                    "arrhenius"
                ] * loss_engine.loss_arrhenius(dD_dt_hat, t_hat, curr_x)

            total_loss.backward()
            optimizer.step()

    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for seq_x, y_target, curr_x in test_loader:
            seq_x, curr_x = seq_x.to(device), curr_x.to(device)
            t_hat, D_hat, _ = model(seq_x)
            poa = curr_x[:, 0:1] * loss_engine.poa_scale + loss_engine.poa_mean
            p_physical = loss_engine.forward_physics_power(poa, t_hat, D_hat)
            preds.append(p_physical.cpu().numpy())
            actuals.append(
                y_target.numpy() * loss_engine.p_scale + loss_engine.p_mean
            )

    preds = np.vstack(preds)
    actuals = np.vstack(actuals)

    return r2_score(actuals, preds)


# -------------------------------------------------------------
# 5. Run Benchmark
# -------------------------------------------------------------
if __name__ == "__main__":
    train_loader, test_loader, scaler_X, scaler_y = prepare_time_series_data(
        "xSi12922.csv"
    )
    engine = MultiPhysicsLossEngine(scaler_X, scaler_y)

    loss_terms = ["data", "thermal", "arrhenius"]
    combinations = []
    for k in range(1, len(loss_terms) + 1):
        combinations.extend(list(itertools.combinations(loss_terms, k)))

    print(
        f"{'Experiment':<5} | {'Active Loss Components':<35} | {'Test R² Score':<10}"
    )
    print("-" * 58)
    for i, combo in enumerate(combinations, start=1):
        score = train_and_evaluate(combo, train_loader, test_loader, engine)
        combo_str = " + ".join(combo)
        print(f"{i:<5} | {combo_str:<35} | {score:10.4f}")
