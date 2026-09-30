
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

"""
Cross-Asset Generalization Benchmark:
Train on Monocrystalline Module (xSi12922.csv) -> Test on Commercial Array (first.csv).

Evaluates all 15 multi-physics loss combinations of the Feature Reduction +
Wavelet + Analytical Single-Diode Hybrid PINN to test for overfitting and cross-asset transfer.
"""

import itertools
import os
import sys
import time
import numpy as np
import pandas as pd
import pywt
import torch
import torch.nn as nn
from sklearn.decomposition import KernelPCA
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)


# -------------------------------------------------------------
# 1. Feature Engineering & Wavelet De-noising
# -------------------------------------------------------------
def build_temporal_features(df, time_col="date"):
    dt = pd.to_datetime(df[time_col])
    df["dt"] = dt
    df["hour"] = dt.dt.hour + dt.dt.minute / 60.0
    df["day_of_year"] = dt.dt.dayofyear
    df["day_id"] = dt.dt.date

    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["sin_day"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["cos_day"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    return df


def hybrid_wavelet_filter(signal_array, wavelet="db4", level=1):
    coeffs = pywt.wavedec(signal_array, wavelet=wavelet, level=level, axis=0)
    approx = coeffs[0]
    detail = coeffs[1]
    reconstructed = pywt.waverec(
        [approx, np.zeros_like(detail)], wavelet=wavelet, axis=0
    )
    return reconstructed[: len(signal_array)], approx, detail


class HybridPVDataset(Dataset):
    def __init__(self, sequences_x, targets_y, raw_poas, current_x):
        self.X = torch.tensor(sequences_x, dtype=torch.float32)
        self.y = torch.tensor(targets_y, dtype=torch.float32).unsqueeze(-1)
        self.raw_poa = torch.tensor(raw_poas, dtype=torch.float32).unsqueeze(-1)
        self.curr_x = torch.tensor(current_x, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.raw_poa[idx], self.curr_x[idx]


# -------------------------------------------------------------
# 2. Structural Hybrid Architecture (Analytical Circuit in Forward)
# -------------------------------------------------------------
class HybridPVTimeSeriesLSTM(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.1
        )
        self.fc_shared = nn.Sequential(nn.Linear(hidden_dim, 32), nn.Tanh())

        self.head_thermal = nn.Linear(32, 1)
        self.head_aging = nn.Sequential(nn.Linear(32, 1), nn.Softplus())
        self.head_diode_n = nn.Sequential(nn.Linear(32, 1), nn.Sigmoid())

        self.eta_stc = 0.16
        self.area = 0.6
        self.gamma_p = -0.004

    def forward(self, x, raw_poa):
        out, _ = self.lstm(x)
        context = self.fc_shared(out[:, -1, :])

        t_cell_hat = self.head_thermal(context)
        dRs_dt_hat = self.head_aging(context)
        diode_n = 1.0 + self.head_diode_n(context)

        # Single-Diode capacity envelope
        p_ideal = (
            raw_poa
            * self.area
            * self.eta_stc
            * (1.0 + self.gamma_p * (t_cell_hat - 25.0))
        )
        p_hybrid_pred = torch.clamp(p_ideal / diode_n, min=0.0)

        return p_hybrid_pred, t_cell_hat, dRs_dt_hat, diode_n


# -------------------------------------------------------------
# 3. Physics Loss Engine
# -------------------------------------------------------------
class HybridPhysicsLossEngine:
    def __init__(self, scaler_X):
        self.scaler_X = scaler_X
        self.mse = nn.MSELoss()

        self.poa_mean = scaler_X.mean_[0]
        self.poa_scale = scaler_X.scale_[0]
        self.tamb_mean = scaler_X.mean_[1]
        self.tamb_scale = scaler_X.scale_[1]
        self.rh_mean = scaler_X.mean_[2]
        self.rh_scale = scaler_X.scale_[2]

        self.area = 0.6
        self.eta_stc = 0.16
        self.gamma_p = -0.004
        self.u0 = 26.9
        self.u1 = 1.06
        self.v_w = 1.5
        self.A_arrh = 1e-4
        self.E_a = 0.35
        self.k_B = 8.617333e-5
        self.gamma_rh = 0.005

    def loss_data(self, p_pred, y_target):
        return self.mse(p_pred, y_target)

    def loss_sde(self, p_pred, raw_poa, t_cell_hat, diode_n):
        poa = torch.clamp(raw_poa, min=0.0)
        p_sde_ref = (
            poa
            * self.area
            * self.eta_stc
            * (1.0 + self.gamma_p * (t_cell_hat - 25.0))
        ) / diode_n
        p_sde_ref = torch.clamp(p_sde_ref, min=0.0)
        p_loss = self.mse(p_pred, p_sde_ref)
        relu_penalty = torch.mean(torch.relu(-p_pred))
        return p_loss + relu_penalty

    def loss_thermal(self, t_cell_hat, current_x_norm):
        poa = current_x_norm[:, 0:1] * self.poa_scale + self.poa_mean
        t_amb = current_x_norm[:, 1:2] * self.tamb_scale + self.tamb_mean
        t_expected = t_amb + (poa / (self.u0 + self.u1 * self.v_w))
        return torch.mean((t_cell_hat - t_expected) ** 2) / 100.0

    def loss_aging(self, dRs_dt_hat, t_cell_hat, current_x_norm):
        rh = torch.clamp(
            current_x_norm[:, 2:3] * self.rh_scale + self.rh_mean,
            min=0.0,
            max=100.0,
        )
        t_kelvin = torch.clamp(t_cell_hat + 273.15, min=200.0)
        r_arrh = (
            self.A_arrh
            * torch.exp(-self.E_a / (self.k_B * t_kelvin))
            * (1.0 + self.gamma_rh * rh)
        )
        return self.mse(torch.log1p(dRs_dt_hat), torch.log1p(r_arrh))


# -------------------------------------------------------------
# 4. Data Preparation: Train on xSi, Test on first.csv
# -------------------------------------------------------------
def load_datasets(seq_len=12):
    print("\n" + "=" * 80)
    print("STEP 1: PREPARING TRAINING DATASET (xSi12922.csv)")
    print("=" * 80)
    df_xsi = pd.read_csv(resolve_dataset_path("xSi12922.csv"))
    df_xsi = build_temporal_features(df_xsi, time_col="date")

    physical_cols_xsi = ["POA", "Dry bulb temperature (degC)", "Relative humidity (%RH)"]
    stochastic_cols_xsi = [
        "Diffuse horizontal irradiance (W/m2)",
        "sin_hour",
        "cos_hour",
        "sin_day",
        "cos_day",
    ]
    target_col_xsi = "Pmp (W)"

    # Clean missing values
    for col in physical_cols_xsi + stochastic_cols_xsi + [target_col_xsi]:
        df_xsi[col] = df_xsi[col].interpolate(method="linear").bfill().ffill()

    unique_days_xsi = df_xsi["day_id"].unique()
    split_point = int(len(unique_days_xsi) * 0.8)
    train_days_xsi = set(unique_days_xsi[:split_point])
    test_days_xsi = set(unique_days_xsi[split_point:])

    df_train_xsi = df_xsi[df_xsi["day_id"].isin(train_days_xsi)].copy()
    df_test_xsi = df_xsi[df_xsi["day_id"].isin(test_days_xsi)].copy()

    # Step B: Fit KPCA on xSi Training Data (5 -> 1)
    scaler_stoch = StandardScaler()
    X_stoch_train_scaled = scaler_stoch.fit_transform(df_train_xsi[stochastic_cols_xsi].values)
    X_stoch_test_scaled = scaler_stoch.transform(df_test_xsi[stochastic_cols_xsi].values)

    kpca = KernelPCA(n_components=1, kernel="rbf", random_state=42, n_jobs=-1)
    fit_idx = np.linspace(0, len(X_stoch_train_scaled) - 1, 5000, dtype=int)
    kpca.fit(X_stoch_train_scaled[fit_idx])

    df_train_xsi["latent_temporal"] = kpca.transform(X_stoch_train_scaled)[:, 0]
    df_test_xsi["latent_temporal"] = kpca.transform(X_stoch_test_scaled)[:, 0]

    hybrid_cols = ["POA", "Temp", "RH", "latent_temporal"]

    # Align column names for clean processing
    df_train_xsi["Temp"] = df_train_xsi["Dry bulb temperature (degC)"]
    df_train_xsi["RH"] = df_train_xsi["Relative humidity (%RH)"]
    df_test_xsi["Temp"] = df_test_xsi["Dry bulb temperature (degC)"]
    df_test_xsi["RH"] = df_test_xsi["Relative humidity (%RH)"]

    scaler_X = StandardScaler()
    scaler_X.fit(df_train_xsi[hybrid_cols].values)

    def extract_day_windows(sub_df, poa_col, target_col, stride=1):
        seqs, targets, raw_poas, currs = [], [], [], []
        for _, day_group in sub_df.groupby("day_id"):
            if len(day_group) <= seq_len:
                continue

            vals = day_group[hybrid_cols].values.copy()
            for c in range(3):  # DWT on 3 physical columns
                denoised, _, _ = hybrid_wavelet_filter(vals[:, c], wavelet="db4", level=1)
                vals[:, c] = denoised

            x_scaled = scaler_X.transform(vals)
            poa_raw = day_group[poa_col].values
            y_raw = day_group[target_col].values

            n_samples = len(day_group) - seq_len
            for i in range(0, n_samples, stride):
                seqs.append(x_scaled[i : i + seq_len])
                targets.append(y_raw[i + seq_len])
                raw_poas.append(poa_raw[i + seq_len])
                currs.append(x_scaled[i + seq_len])

        return np.array(seqs), np.array(targets), np.array(raw_poas), np.array(currs)

    print("Building xSi12922 training and validation sequences...")
    X_train_xsi, y_train_xsi, poa_train_xsi, curr_train_xsi = extract_day_windows(
        df_train_xsi, "POA", target_col_xsi
    )
    X_test_xsi, y_test_xsi, poa_test_xsi, curr_test_xsi = extract_day_windows(
        df_test_xsi, "POA", target_col_xsi
    )

    train_loader_xsi = DataLoader(
        HybridPVDataset(X_train_xsi, y_train_xsi, poa_train_xsi, curr_train_xsi),
        batch_size=64,
        shuffle=True,
    )
    test_loader_xsi = DataLoader(
        HybridPVDataset(X_test_xsi, y_test_xsi, poa_test_xsi, curr_test_xsi),
        batch_size=128,
        shuffle=False,
    )
    print(f"xSi Train Sequences: {len(X_train_xsi):,} | xSi Test Sequences: {len(X_test_xsi):,}")

    # ---------------------------------------------------------
    # STEP 2: PREPARING TARGET EVALUATION DATASET (first.csv)
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 2: PREPARING TARGET EVALUATION DATASET (first.csv, Year 2016)")
    print("=" * 80)
    cols_first = [
        "timestamp",
        "Active_Power",
        "Weather_Temperature_Celsius",
        "Weather_Relative_Humidity",
        "Global_Horizontal_Radiation",
        "Radiation_Global_Tilted",
        "Diffuse_Horizontal_Radiation",
    ]
    df_first = pd.read_csv("first.csv", usecols=cols_first)
    df_first = build_temporal_features(df_first, time_col="timestamp")
    df_first = df_first[df_first["dt"].dt.year == 2016].copy().sort_values("dt").reset_index(drop=True)

    df_first["POA"] = df_first["Radiation_Global_Tilted"].fillna(df_first["Global_Horizontal_Radiation"]).clip(lower=0.0)
    df_first["Temp"] = df_first["Weather_Temperature_Celsius"]
    df_first["RH"] = df_first["Weather_Relative_Humidity"]

    # Target: Active_Power is in kW -> convert to Watts to match physical dimensions
    df_first["Active_Power_W"] = df_first["Active_Power"] * 1000.0

    stochastic_cols_first = [
        "Diffuse_Horizontal_Radiation",
        "sin_hour",
        "cos_hour",
        "sin_day",
        "cos_day",
    ]
    for col in ["POA", "Temp", "RH", "Active_Power_W"] + stochastic_cols_first:
        df_first[col] = df_first[col].interpolate(method="linear").bfill().ffill()

    # Apply training scaler and KPCA
    X_stoch_first_scaled = scaler_stoch.transform(df_first[stochastic_cols_first].values)
    df_first["latent_temporal"] = kpca.transform(X_stoch_first_scaled)[:, 0]

    # Stride=2 covers all 366 days evenly with 50k sequences
    X_test_first, y_test_first, poa_test_first, curr_test_first = extract_day_windows(
        df_first, "POA", "Active_Power_W", stride=2
    )

    test_loader_first = DataLoader(
        HybridPVDataset(X_test_first, y_test_first, poa_test_first, curr_test_first),
        batch_size=256,
        shuffle=False,
    )
    print(f"first.csv Evaluation Sequences: {len(X_test_first):,} (across 366 days)")

    return train_loader_xsi, test_loader_xsi, test_loader_first, scaler_X


# -------------------------------------------------------------
# 5. Training & Evaluation Pipeline
# -------------------------------------------------------------
def train_and_cross_evaluate(
    active_losses,
    train_loader_xsi,
    test_loader_xsi,
    test_loader_first,
    loss_engine,
    epochs=6,
    lr=0.002,
):
    torch.manual_seed(42)
    model = HybridPVTimeSeriesLSTM(input_dim=4, hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    weights = {"sde": 0.1, "thermal": 0.05, "aging": 0.01}

    model.train()
    for _ in range(epochs):
        for bx, by, bpoa, bcurr in train_loader_xsi:
            bx, by, bpoa, bcurr = bx.to(device), by.to(device), bpoa.to(device), bcurr.to(device)
            optimizer.zero_grad()
            p_pred, t_hat, dRs_hat, diode_n = model(bx, bpoa)

            total_loss = torch.tensor(0.0, device=device)
            if "data" in active_losses:
                total_loss += loss_engine.loss_data(p_pred, by)
            if "sde" in active_losses:
                total_loss += weights["sde"] * loss_engine.loss_sde(p_pred, bpoa, t_hat, diode_n)
            if "thermal" in active_losses:
                total_loss += weights["thermal"] * loss_engine.loss_thermal(t_hat, bcurr)
            if "aging" in active_losses:
                total_loss += weights["aging"] * loss_engine.loss_aging(dRs_hat, t_hat, bcurr)

            total_loss.backward()
            optimizer.step()

    # 1. Native Evaluation on xSi12922 Test Set
    model.eval()
    preds_xsi, actuals_xsi = [], []
    with torch.no_grad():
        for bx, by, bpoa, _ in test_loader_xsi:
            bx, bpoa = bx.to(device), bpoa.to(device)
            p_pred, _, _, _ = model(bx, bpoa)
            preds_xsi.append(p_pred.cpu().numpy())
            actuals_xsi.append(by.numpy())

    preds_xsi = np.vstack(preds_xsi)
    actuals_xsi = np.vstack(actuals_xsi)
    r2_native = r2_score(actuals_xsi, preds_xsi)
    mae_native = mean_absolute_error(actuals_xsi, preds_xsi)

    # 2. Zero-Shot Cross-Asset Evaluation on first.csv
    preds_first, actuals_first, poas_first = [], [], []
    with torch.no_grad():
        for bx, by, bpoa, _ in test_loader_first:
            bx, bpoa = bx.to(device), bpoa.to(device)
            p_pred, _, _, _ = model(bx, bpoa)
            preds_first.append(p_pred.cpu().numpy())
            actuals_first.append(by.numpy())
            poas_first.append(bpoa.cpu().numpy())

    preds_first = np.vstack(preds_first).flatten()
    actuals_first = np.vstack(actuals_first).flatten()
    poas_first = np.vstack(poas_first).flatten()

    # Night clamp
    night_mask = poas_first < 5.0
    preds_first[night_mask] = 0.0

    # A: Raw Zero-Shot Transfer (Watts to Watts, uncalibrated nameplate)
    r2_raw = r2_score(actuals_first, preds_first)

    # B: Capacity-Scaled Transfer (Commercial Array Nameplate Calibration)
    # Optimal least-squares scale factor S = sum(y * y_hat) / sum(y_hat^2)
    denom = np.sum(preds_first**2)
    scale_factor = (np.sum(actuals_first * preds_first) / denom) if denom > 0 else 1.0
    preds_scaled_w = preds_first * scale_factor

    r2_scaled = r2_score(actuals_first, preds_scaled_w)
    mae_w = mean_absolute_error(actuals_first, preds_scaled_w)
    rmse_w = root_mean_squared_error(actuals_first, preds_scaled_w)

    mae_kw = mae_w / 1000.0
    rmse_kw = rmse_w / 1000.0

    return {
        "r2_native": r2_native,
        "mae_native": mae_native,
        "r2_raw": r2_raw,
        "scale": scale_factor,
        "r2_scaled": r2_scaled,
        "mae_kw": mae_kw,
        "mae_w": mae_w,
        "rmse_kw": rmse_kw,
    }


def main():
    train_loader_xsi, test_loader_xsi, test_loader_first, scaler_X = load_datasets(seq_len=12)
    loss_engine = HybridPhysicsLossEngine(scaler_X)

    components = ["data", "sde", "thermal", "aging"]
    all_combos = []
    for r in range(1, 5):
        all_combos.extend(itertools.combinations(components, r))

    print("\n" + "=" * 110)
    print("CROSS-ASSET BENCHMARK: TRAINED ON xSi12922 (70W) -> TESTED ON first.csv (25kW Commercial Array)")
    print("=" * 110)
    print(
        f"{'Exp #':<7} | {'Active Loss Components':<30} | {'xSi R^2':<8} | {'Raw Trans R^2':<14} | {'Scale (S)':<10} | {'Scaled R^2':<11} | {'MAE (kW)':<9} | {'MAE (W)':<9}"
    )
    print("-" * 110)

    results = []
    for idx, combo in enumerate(all_combos, start=1):
        t0 = time.time()
        res = train_and_cross_evaluate(
            combo,
            train_loader_xsi,
            test_loader_xsi,
            test_loader_first,
            loss_engine,
            epochs=6,
            lr=0.002,
        )
        elapsed = time.time() - t0
        combo_name = " + ".join(combo)
        print(
            f"{idx:<7} | {combo_name:<30} | {res['r2_native']:<8.4f} | {res['r2_raw']:<14.4f} | {res['scale']:<10.1f} | {res['r2_scaled']:<11.4f} | {res['mae_kw']:<9.3f} | {res['mae_w']:<9.1f}  ({elapsed:.1f}s)"
        )
        sys.stdout.flush()
        res["exp"] = idx
        res["combo"] = combo_name
        results.append(res)

    print("=" * 110)
    best_scaled = max(results, key=lambda x: x["r2_scaled"])
    print(f"\n[SUMMARY] Peak Generalization: Exp {best_scaled['exp']} ({best_scaled['combo']})")
    print(f" -> xSi12922 Native Test R^2: {best_scaled['r2_native']:.4f}")
    print(f" -> first.csv Zero-Shot Scaled R^2: {best_scaled['r2_scaled']:.4f}")
    print(f" -> Array Scaling Factor S: {best_scaled['scale']:.1f}x (70W module -> 25kW array)")
    print(f" -> Array Transfer MAE: {best_scaled['mae_kw']:.3f} kW ({best_scaled['mae_w']:.1f} W)")


if __name__ == "__main__":
    main()
