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
# 2. Time-Series Multi-Head LSTM Model
# -------------------------------------------------------------
class PVTimeSeriesLSTM(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1
        )
        self.fc_shared = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
        )
        # Power forecast: \hat{P}_{t}
        self.head_power = nn.Linear(32, 1)
        # Thermal forecast: \hat{T}_{cell, t}
        self.head_thermal = nn.Linear(32, 1)
        # Degradation rate: d\hat{D}/dt
        self.head_aging = nn.Sequential(nn.Linear(32, 1), nn.Softplus())

    def forward(self, x):
        out, _ = self.lstm(x)
        # Temporal context vector from the final recurrent step
        context = self.fc_shared(out[:, -1, :])
        p_hat = self.head_power(context)
        t_cell_hat = self.head_thermal(context)
        dD_dt_hat = self.head_aging(context)
        return p_hat, t_cell_hat, dD_dt_hat


# -------------------------------------------------------------
# 3. Physics Loss Engine
# -------------------------------------------------------------
class MultiPhysicsLossEngine:
    def __init__(self, scaler_X, scaler_y):
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        self.mse = nn.MSELoss()

        # Unscale parameters
        self.poa_mean = scaler_X.mean_[0]
        self.poa_scale = scaler_X.scale_[0]
        self.tamb_mean = scaler_X.mean_[1]
        self.tamb_scale = scaler_X.scale_[1]
        self.rh_mean = scaler_X.mean_[2]
        self.rh_scale = scaler_X.scale_[2]
        self.p_mean = scaler_y.mean_[0]
        self.p_scale = scaler_y.scale_[0]

        # Domain physics coefficients
        self.eta_stc = 0.16
        self.gamma_p = -0.004
        self.area = 0.6
        self.u0 = 26.9
        self.u1 = 1.06
        self.v_w = 1.5
        self.A_arrh = 1e-4
        self.E_a = 0.35
        self.k_B = 8.617333e-5
        self.gamma_rh = 0.005

    def loss_data(self, p_hat_norm, y_norm):
        return self.mse(p_hat_norm, y_norm)

    def loss_sde(self, p_hat_norm, current_x_norm, t_cell_hat):
        poa = current_x_norm[:, 0:1] * self.poa_scale + self.poa_mean
        poa = torch.clamp(poa, min=0.0)

        # Baseline envelope prediction
        p_ref = (
            poa
            * self.area
            * self.eta_stc
            * (1.0 + self.gamma_p * (t_cell_hat - 25.0))
        )
        p_ref = torch.clamp(p_ref, min=0.0)

        p_pred = p_hat_norm * self.p_scale + self.p_mean
        p_loss = torch.mean((p_pred - p_ref) ** 2) / (self.p_scale**2)
        relu_penalty = torch.mean(torch.relu(-p_pred))
        return p_loss + relu_penalty

    def loss_thermal(self, t_cell_hat, current_x_norm):
        poa = current_x_norm[:, 0:1] * self.poa_scale + self.poa_mean
        t_amb = current_x_norm[:, 1:2] * self.tamb_scale + self.tamb_mean
        t_expected = t_amb + (poa / (self.u0 + self.u1 * self.v_w))
        return torch.mean((t_cell_hat - t_expected) ** 2) / 100.0

    def loss_arrhenius(self, dD_dt_hat, t_cell_hat, current_x_norm):
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
# 4. Training & Combinatoric Evaluation Loop
# -------------------------------------------------------------
def train_and_evaluate(active_losses, train_loader, test_loader, loss_engine):
    model = PVTimeSeriesLSTM(input_dim=8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = 6

    weights = {"sde": 0.1, "thermal": 0.05, "arrhenius": 0.01}

    model.train()
    for _ in range(epochs):
        for seq_x, y_target, curr_x in train_loader:
            seq_x, y_target, curr_x = (
                seq_x.to(device),
                y_target.to(device),
                curr_x.to(device),
            )
            optimizer.zero_grad()
            p_hat, t_hat, dD_hat = model(seq_x)

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
# 5. Run Benchmark
# -------------------------------------------------------------
if __name__ == "__main__":
    train_loader, test_loader, scaler_X, scaler_y = prepare_time_series_data(
        "xSi12922.csv"
    )
    engine = MultiPhysicsLossEngine(scaler_X, scaler_y)

    loss_terms = ["data", "sde", "thermal", "arrhenius"]
    combinations = []
    for k in range(1, 5):
        combinations.extend(list(itertools.combinations(loss_terms, k)))

    print(
        f"{'Experiment':<5} | {'Active Loss Components':<42} | {'Test R² Score':<10}"
    )
    print("-" * 65)
    for i, combo in enumerate(combinations, start=1):
        score = train_and_evaluate(combo, train_loader, test_loader, engine)
        combo_str = " + ".join(combo)
        print(f"{i:<5} | {combo_str:<42} | {score:10.4f}")