
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
Training Pipeline for Feature Reduction + Hybridization PINN on first.csv.

Applies:
1. Physics-Preserving Feature Split:
   - Physical Core (Pristine): POA (Radiation_Global_Tilted / Global_Horizontal_Radiation),
     Weather_Temperature_Celsius, Weather_Relative_Humidity.
   - Stochastic Background: Diffuse_Horizontal_Radiation, sin_hour, cos_hour, sin_day, cos_day.
2. Kernel PCA Dimensionality Reduction:
   - Compresses 5 stochastic features down to 1 non-linear latent temporal vector.
   - Input dimension reduced from 8 to 4 features.
3. Model Hybridization:
   - Hybridization A (Wavelet DWT): Decomposes physical core features using 'db4'
     discrete wavelet transform to isolate high-frequency turbulence from diurnal thermodynamics.
   - Hybridization B (Structural Circuit Hybrid): Hard-coded analytical Single-Diode capacity
     equation driven by latent neural parameter heads (T_cell, aging dRs/dt, diode ideality n).
4. Multi-Physics Loss Function:
   - Supervised data loss (MSE)
   - Single-Diode Equation circuit loss (loss_sde)
   - Thermodynamic heat dissipation loss (loss_thermal)
   - Arrhenius degradation kinetics loss (loss_aging)
"""

import argparse
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


def build_temporal_features(df):
    """Generates cyclic diurnal and annual trigonometric encodings."""
    dt = pd.to_datetime(df["timestamp"])
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
    """Decomposes physical series into high-frequency details and low-frequency

    approximations using Discrete Wavelet Transform (DWT).
    Reconstructs smooth de-noised baseline to prevent sensor noise from confusing
    recurrent hidden states.
    """
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
        return (
            self.X[idx],
            self.y[idx],
            self.raw_poa[idx],
            self.curr_x[idx],
        )


def prepare_first_csv_data(
    csv_path="first.csv",
    year=2016,
    seq_len=12,
    split_ratio=0.8,
    sample_stride=1,
):
    """Prepares first.csv data with Physics-Preserving Feature Split, KPCA, and Wavelet DWT."""
    print(f"\n[DATA PREP] Loading {csv_path} (Year {year})...")
    cols = [
        "timestamp",
        "Active_Power",
        "Weather_Temperature_Celsius",
        "Weather_Relative_Humidity",
        "Global_Horizontal_Radiation",
        "Radiation_Global_Tilted",
        "Diffuse_Horizontal_Radiation",
    ]
    df = pd.read_csv(csv_path, usecols=cols)
    df = build_temporal_features(df)

    if year is not None:
        df = df[df["dt"].dt.year == year].copy()

    df = df.sort_values("dt").reset_index(drop=True)

    # Physical Core Irradiance: Tilt POA with GHI fallback
    df["POA"] = df["Radiation_Global_Tilted"].fillna(
        df["Global_Horizontal_Radiation"]
    )
    df["POA"] = df["POA"].clip(lower=0.0)

    # Impute missing values
    for c in [
        "Diffuse_Horizontal_Radiation",
        "Weather_Temperature_Celsius",
        "Weather_Relative_Humidity",
        "Active_Power",
    ]:
        df[c] = df[c].interpolate(method="linear").bfill().ffill()

    physical_cols = [
        "POA",
        "Weather_Temperature_Celsius",
        "Weather_Relative_Humidity",
    ]
    stochastic_cols = [
        "Diffuse_Horizontal_Radiation",
        "sin_hour",
        "cos_hour",
        "sin_day",
        "cos_day",
    ]
    target_col = "Active_Power"

    print(f"[DATA PREP] Loaded {len(df):,} rows across {df['day_id'].nunique()} unique days.")

    # Train / Test Chronological Day Split
    unique_days = df["day_id"].unique()
    split_point = int(len(unique_days) * split_ratio)
    train_days = set(unique_days[:split_point])
    test_days = set(unique_days[split_point:])

    df_train = df[df["day_id"].isin(train_days)].copy()
    df_test = df[df["day_id"].isin(test_days)].copy()

    print(f"[DATA PREP] Train Days: {len(train_days)} | Test Days: {len(test_days)}")

    # Step B: Kernel PCA Dimension Reduction on Stochastic Noise (5 -> 1)
    print("[KPCA] Fitting Kernel PCA (RBF kernel, 1 component) on stochastic features...")
    scaler_stoch = StandardScaler()
    X_stoch_train_scaled = scaler_stoch.fit_transform(
        df_train[stochastic_cols].values
    )
    X_stoch_test_scaled = scaler_stoch.transform(
        df_test[stochastic_cols].values
    )

    kpca = KernelPCA(n_components=1, kernel="rbf", random_state=42, n_jobs=-1)
    fit_idx = np.linspace(0, len(X_stoch_train_scaled) - 1, min(5000, len(X_stoch_train_scaled)), dtype=int)
    kpca.fit(X_stoch_train_scaled[fit_idx])

    df_train["latent_temporal"] = kpca.transform(X_stoch_train_scaled)[:, 0]
    df_test["latent_temporal"] = kpca.transform(X_stoch_test_scaled)[:, 0]
    print("[KPCA] Dimensionality reduced: 5 stochastic features -> 1 latent temporal vector!")

    hybrid_cols = physical_cols + ["latent_temporal"]

    scaler_X = StandardScaler()
    scaler_X.fit(df_train[hybrid_cols].values)

    # Step C: Wavelet DWT De-noising & Sequence Extraction
    print("[WAVELET] Applying Wavelet DWT ('db4') de-noising & building sequence windows...")

    def extract_day_windows(sub_df):
        seqs, targets, raw_poas, currs = [], [], [], []
        for _, day_group in sub_df.groupby("day_id"):
            if len(day_group) <= seq_len:
                continue

            vals = day_group[hybrid_cols].values.copy()
            # De-noise physical columns with DWT
            for c in range(len(physical_cols)):
                denoised_col, _, _ = hybrid_wavelet_filter(
                    vals[:, c], wavelet="db4", level=1
                )
                vals[:, c] = denoised_col

            x_scaled = scaler_X.transform(vals)
            poa_raw = day_group["POA"].values
            y_raw = day_group[target_col].values

            n_samples = len(day_group) - seq_len
            for i in range(0, n_samples, sample_stride):
                seqs.append(x_scaled[i : i + seq_len])
                targets.append(y_raw[i + seq_len])
                raw_poas.append(poa_raw[i + seq_len])
                currs.append(x_scaled[i + seq_len])

        return (
            np.array(seqs),
            np.array(targets),
            np.array(raw_poas),
            np.array(currs),
        )

    X_train, y_train, poa_train, curr_train = extract_day_windows(df_train)
    X_test, y_test, poa_test, curr_test = extract_day_windows(df_test)

    print(f"[DATA PREP] Extracted Train Sequences: {len(X_train):,} | Test Sequences: {len(X_test):,}")

    train_ds = HybridPVDataset(X_train, y_train, poa_train, curr_train)
    test_ds = HybridPVDataset(X_test, y_test, poa_test, curr_test)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    return train_loader, test_loader, scaler_X, df_test


class HybridCommercialPVLSTM(nn.Module):
    """Hybrid Structural Architecture for 25kW Commercial PV Array (first.csv).

    Eliminates unconstrained black-box regression head.
    Predicts unobservable physical states (T_cell, aging rate dRs/dt, diode ideality n)
    and evaluates analytical Single-Diode circuit power envelope inside forward pass.
    """

    def __init__(self, input_dim=4, hidden_dim=64, capacity_kw=25.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.1
        )
        self.fc_shared = nn.Sequential(nn.Linear(hidden_dim, 32), nn.Tanh())

        # Unobservable Physical State Estimators
        self.head_thermal = nn.Linear(32, 1)  # Junction cell temperature
        self.head_aging = nn.Sequential(nn.Linear(32, 1), nn.Softplus())  # dRs/dt
        self.head_diode_n = nn.Sequential(nn.Linear(32, 1), nn.Sigmoid())  # Ideality factor

        # Commercial Array Nameplate Parameters
        # At STC (1000 W/m2, 25 C), P_nom ~ 25 kW.
        # With diode_n in [1.0, 2.0], base capacity envelope coefficient:
        self.base_capacity = (capacity_kw / 1000.0) * 1.15
        self.gamma_p = -0.004

    def forward(self, x, raw_poa):
        out, _ = self.lstm(x)
        context = self.fc_shared(out[:, -1, :])

        t_cell_hat = self.head_thermal(context)
        dRs_dt_hat = self.head_aging(context)

        # Diode ideality strictly bounded within [1.0, 2.0] for Silicon
        diode_n = 1.0 + self.head_diode_n(context)

        # Analytical Single-Diode capacity envelope (kW)
        p_ideal = (
            raw_poa * self.base_capacity * (1.0 + self.gamma_p * (t_cell_hat - 25.0))
        )

        # Structurally bounded power prediction (kW)
        p_hybrid_pred = torch.clamp(p_ideal / diode_n, min=0.0)

        return p_hybrid_pred, t_cell_hat, dRs_dt_hat, diode_n, p_ideal


class MultiPhysicsLossEngine:
    def __init__(self, scaler_X):
        self.mse = nn.MSELoss()
        self.relu = nn.ReLU()

        self.poa_mean = scaler_X.mean_[0]
        self.poa_scale = scaler_X.scale_[0]
        self.tamb_mean = scaler_X.mean_[1]
        self.tamb_scale = scaler_X.scale_[1]
        self.rh_mean = scaler_X.mean_[2]
        self.rh_scale = scaler_X.scale_[2]

        self.E_a = 0.45
        self.k_B = 8.617333e-5
        self.A_arrh = 1000.0
        self.gamma_rh = 0.005

    def compute_losses(
        self,
        p_pred,
        y_true,
        t_cell_hat,
        dRs_dt_hat,
        diode_n,
        p_ideal,
        raw_poa,
        curr_x,
    ):
        # 1. Supervised Target Loss
        loss_data = self.mse(p_pred, y_true)

        # 2. Single-Diode Circuit Loss (loss_sde)
        p_circuit_target = p_ideal / diode_n
        loss_sde = self.mse(p_pred, p_circuit_target) + self.relu(-p_pred).mean()

        # 3. Thermodynamic Heat Dissipation Loss
        t_amb = curr_x[:, 1:2] * self.tamb_scale + self.tamb_mean
        t_cell_expected = t_amb + raw_poa * 0.03125
        loss_thermal = 0.01 * self.mse(t_cell_hat, t_cell_expected)

        # 4. Arrhenius Aging Kinetics Loss
        rh_val = curr_x[:, 2:3] * self.rh_scale + self.rh_mean
        t_kelvin = torch.clamp(t_cell_hat + 273.15, min=250.0, max=400.0)
        arrh_rate = (
            self.A_arrh
            * torch.exp(-self.E_a / (self.k_B * t_kelvin))
            * (1.0 + self.gamma_rh * rh_val)
        )
        loss_aging = self.mse(
            torch.log1p(dRs_dt_hat),
            torch.log1p(arrh_rate),
        )

        return loss_data, loss_sde, loss_thermal, loss_aging


def train_and_evaluate(
    train_loader,
    test_loader,
    scaler_X,
    active_losses=("data", "sde", "thermal", "aging"),
    epochs=6,
    lr=1e-3,
):
    model = HybridCommercialPVLSTM(input_dim=4, hidden_dim=64, capacity_kw=25.0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_engine = MultiPhysicsLossEngine(scaler_X)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for x_batch, y_batch, poa_batch, curr_batch in train_loader:
            x_b = x_batch.to(device)
            y_b = y_batch.to(device)
            poa_b = poa_batch.to(device)
            curr_b = curr_batch.to(device)

            optimizer.zero_grad()
            p_pred, t_cell_hat, dRs_dt_hat, diode_n, p_ideal = model(x_b, poa_b)

            l_data, l_sde, l_thermal, l_aging = loss_engine.compute_losses(
                p_pred, y_b, t_cell_hat, dRs_dt_hat, diode_n, p_ideal, poa_b, curr_b
            )

            total_loss = torch.tensor(0.0, device=device)
            if "data" in active_losses:
                total_loss = total_loss + l_data
            if "sde" in active_losses:
                total_loss = total_loss + l_sde
            if "thermal" in active_losses:
                total_loss = total_loss + l_thermal
            if "aging" in active_losses:
                total_loss = total_loss + l_aging

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += total_loss.item()

    # Evaluation
    model.eval()
    preds, actuals, poas = [], [], []
    with torch.no_grad():
        for x_batch, y_batch, poa_batch, _ in test_loader:
            x_b = x_batch.to(device)
            poa_b = poa_batch.to(device)
            p_pred, _, _, _, _ = model(x_b, poa_b)
            preds.extend(p_pred.squeeze(-1).cpu().numpy())
            actuals.extend(y_batch.squeeze(-1).cpu().numpy())
            poas.extend(poa_batch.squeeze(-1).cpu().numpy())

    preds = np.array(preds)
    actuals = np.array(actuals)
    poas = np.array(poas)

    # Force night zero
    night_mask = poas < 5.0
    preds[night_mask] = 0.0

    r2 = r2_score(actuals, preds)
    mae_kw = mean_absolute_error(actuals, preds)
    rmse_kw = root_mean_squared_error(actuals, preds)
    mae_w = mae_kw * 1000.0
    rmse_w = rmse_kw * 1000.0

    return r2, mae_kw, mae_w, rmse_kw, rmse_w, model


def main():
    parser = argparse.ArgumentParser(description="Train Hybrid KPCA-Wavelet PINN on first.csv")
    parser.add_argument("--csv", type=str, default="first.csv")
    parser.add_argument("--year", type=int, default=2016)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--full_15", action="store_true", help="Run all 15 loss combinations")
    args = parser.parse_args()

    train_loader, test_loader, scaler_X, df_test = prepare_first_csv_data(
        csv_path=args.csv, year=args.year
    )

    if args.full_15:
        components = ["data", "sde", "thermal", "aging"]
        combos = []
        for r in range(1, 5):
            combos.extend(itertools.combinations(components, r))
    else:
        # Key representative benchmark combinations
        combos = [
            ("data",),
            ("sde",),
            ("data", "sde"),
            ("data", "thermal"),
            ("data", "sde", "thermal"),
            ("data", "sde", "thermal", "aging"),
        ]

    print("\n" + "=" * 90)
    print(f"HYBRID KPCA + WAVELET + SDE PINN BENCHMARK ON {args.csv.upper()} (Year {args.year})")
    print("=" * 90)
    print(f"{'Exp #':<7} | {'Active Loss Components':<30} | {'Test R^2':<10} | {'MAE (kW)':<10} | {'MAE (Watts)':<12} | {'RMSE (kW)':<10}")
    print("-" * 90)

    results = []
    for idx, combo in enumerate(combos, start=1):
        t_start = time.time()
        r2, mae_kw, mae_w, rmse_kw, rmse_w, _ = train_and_evaluate(
            train_loader, test_loader, scaler_X, active_losses=combo, epochs=args.epochs, lr=args.lr
        )
        elapsed = time.time() - t_start
        combo_name = " + ".join(combo)
        print(f"{idx:<7} | {combo_name:<30} | {r2:<10.4f} | {mae_kw:<10.3f} | {mae_w:<12.1f} | {rmse_kw:<10.3f}  ({elapsed:.1f}s)")
        results.append({
            "exp": idx,
            "combo": combo_name,
            "r2": r2,
            "mae_kw": mae_kw,
            "mae_w": mae_w,
            "rmse_kw": rmse_kw,
            "rmse_w": rmse_w,
        })

    print("=" * 90)
    best_exp = max(results, key=lambda x: x["r2"])
    print(f"Top Performer: Exp {best_exp['exp']} ({best_exp['combo']}) -> R^2: {best_exp['r2']:.4f}, MAE: {best_exp['mae_kw']:.3f} kW ({best_exp['mae_w']:.1f} W)")


if __name__ == "__main__":
    main()
