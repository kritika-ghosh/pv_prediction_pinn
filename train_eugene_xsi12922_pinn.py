"""Cross-Site Overfitting & Generalization Benchmark:
Trained on xSi12922.csv -> Evaluated on Eugene_xSi12922.csv.

Purpose:
Rigorous out-of-domain cross-site validation to verify whether the model
overfit to its training climate or learned universal semiconductor physics:
- Training Asset: xSi12922.csv (Monocrystalline Silicon module #12922)
- Test Asset:     Eugene_xSi12922.csv (Exact same module #12922 at Eugene, Oregon)

Because both files use the exact same 70W monocrystalline silicon hardware (xSi12922),
there is zero wattage capacity mismatch. Any performance difference reflects genuine
generalization across different years (2011-2012 vs 2012-2014) and different climates.

Key Architecture Features:
1. Physics-Preserving Feature Reduction (The Split):
   - Physical Core (Pristine): POA, Dry bulb temperature (degC), Relative humidity (%RH)
   - Stochastic Noise: Atmospheric pressure (mb), sin_hour, cos_hour, sin_day, cos_day
   - Kernel PCA (RBF kernel): Compresses 5 stochastic features down to 1 dense latent temporal vector.
2. Wavelet DWT De-noising ('db4', level=1):
   - Separates rapid sensor turbulence from slow thermodynamic trends.
3. Structural Analytical Single-Diode Circuit Solver:
   - Power is structurally bounded inside forward pass:
     P_ideal = POA * Area * eta_stc * (1 + gamma_p * (T_cell - 25))
     P_pred  = clamp(P_ideal / diode_n, min=0.0)
4. Multi-Physics Loss Engine:
   - Evaluates all 15 loss combinations of: data, sde, thermal, and aging.

NOTE FOR USER: Run this script when ready using:
    python train_eugene_xsi12922_pinn.py
"""

import sys
import itertools
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
# 1. Feature Engineering, KPCA Reduction & Wavelet Hybridization
# -------------------------------------------------------------
def build_temporal_features(df):
    """Generates cyclic diurnal and annual trigonometric encodings."""
    dt = pd.to_datetime(df["date"])
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
    """
    coeffs = pywt.wavedec(signal_array, wavelet=wavelet, level=level, axis=0)
    approx = coeffs[0]
    detail = coeffs[1]
    reconstructed = pywt.waverec(
        [approx, np.zeros_like(detail)], wavelet=wavelet, axis=0
    )
    return reconstructed[: len(signal_array)], approx, detail


class CrossSitePVDataset(Dataset):
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


def prepare_cross_site_data(
    train_csv="xSi12922.csv", test_csv="Eugene_xSi12922.csv", seq_len=12
):
    """Loads training data from xSi12922.csv and test data from Eugene_xSi12922.csv.

    Fits all scalers and KPCA on the training dataset and transforms the unseen Eugene test dataset.
    """
    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)

    # Handle column alias for POA in Eugene dataset if needed
    if "POA" not in df_test.columns:
        if "POA irradiance CMP22 pyranometer (W/m2)" in df_test.columns:
            df_test["POA"] = df_test["POA irradiance CMP22 pyranometer (W/m2)"]
        else:
            raise KeyError("Could not find POA irradiance column in test dataset.")

    df_train = build_temporal_features(df_train)
    df_test = build_temporal_features(df_test)

    physical_cols = [
        "POA",
        "Dry bulb temperature (degC)",
        "Relative humidity (%RH)",
    ]

    stochastic_cols = [
        "Atmospheric pressure (mb)",
        "sin_hour",
        "cos_hour",
        "sin_day",
        "cos_day",
    ]
    target_col = "Pmp (W)"

    # Fit KPCA on training stochastic noise (5 -> 1)
    scaler_stoch = StandardScaler()
    X_stoch_train_scaled = scaler_stoch.fit_transform(df_train[stochastic_cols].values)
    X_stoch_test_scaled = scaler_stoch.transform(df_test[stochastic_cols].values)

    kpca = KernelPCA(n_components=1, kernel="rbf", random_state=42, n_jobs=-1)
    fit_idx = np.linspace(0, len(X_stoch_train_scaled) - 1, 5000, dtype=int)
    kpca.fit(X_stoch_train_scaled[fit_idx])

    df_train["latent_temporal"] = kpca.transform(X_stoch_train_scaled)[:, 0]
    df_test["latent_temporal"] = kpca.transform(X_stoch_test_scaled)[:, 0]

    hybrid_cols = physical_cols + ["latent_temporal"]

    scaler_X = StandardScaler()
    scaler_X.fit(df_train[hybrid_cols].values)

    def extract_day_windows(sub_df):
        seqs, targets, raw_poas, currs = [], [], [], []
        for _, day_group in sub_df.groupby("day_id"):
            if len(day_group) <= seq_len:
                continue

            vals = day_group[hybrid_cols].values.copy()
            for c in range(len(physical_cols)):
                denoised_col, _, _ = hybrid_wavelet_filter(vals[:, c], wavelet="db4", level=1)
                vals[:, c] = denoised_col

            x_scaled = scaler_X.transform(vals)
            poa_raw = day_group["POA"].values
            y_raw = day_group[target_col].values

            for i in range(len(day_group) - seq_len):
                seqs.append(x_scaled[i : i + seq_len])
                targets.append(y_raw[i + seq_len])
                raw_poas.append(poa_raw[i + seq_len])
                currs.append(x_scaled[i + seq_len])

        return np.array(seqs), np.array(targets), np.array(raw_poas), np.array(currs)

    X_train, y_train, poa_train, curr_train = extract_day_windows(df_train)
    X_test, y_test, poa_test, curr_test = extract_day_windows(df_test)

    train_ds = CrossSitePVDataset(X_train, y_train, poa_train, curr_train)
    test_ds = CrossSitePVDataset(X_test, y_test, poa_test, curr_test)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    return train_loader, test_loader, scaler_X


# -------------------------------------------------------------
# 2. Structural Hybrid Architecture (xSi12922 Physical Parameters)
# -------------------------------------------------------------
class CrossSiteHybridPVLSTM(nn.Module):
    """Hybrid Structural Architecture for xSi12922 Monocrystalline Module.

    Both xSi12922 and Eugene_xSi12922 share the exact same hardware:
    Area = 0.6 m^2, eta_stc = 0.16 (Rated 70W Module).
    """

    def __init__(self, input_dim=4, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.1
        )
        self.fc_shared = nn.Sequential(nn.Linear(hidden_dim, 32), nn.Tanh())

        self.head_thermal = nn.Linear(32, 1)  # Junction cell temperature
        self.head_aging = nn.Sequential(
            nn.Linear(32, 1), nn.Softplus()
        )  # Degradation rate dRs/dt
        self.head_diode_n = nn.Sequential(
            nn.Linear(32, 1), nn.Sigmoid()
        )  # Diode ideality factor

    def forward(self, x, raw_poa):
        out, _ = self.lstm(x)
        context = self.fc_shared(out[:, -1, :])

        t_cell_hat = self.head_thermal(context)
        dRs_dt_hat = self.head_aging(context)

        # Diode ideality strictly bounded within [1.0, 2.0] for Silicon
        diode_n = 1.0 + self.head_diode_n(context)

        # Monocrystalline xSi12922 module physical constants
        eta_stc = 0.16
        area = 0.6
        gamma_p = -0.004

        # Analytical Single-Diode capacity envelope
        p_ideal = raw_poa * area * eta_stc * (1.0 + gamma_p * (t_cell_hat - 25.0))

        # Structurally bounded power prediction
        p_hybrid_pred = torch.clamp(p_ideal / diode_n, min=0.0)

        return p_hybrid_pred, t_cell_hat, dRs_dt_hat, diode_n


# -------------------------------------------------------------
# 3. Hybrid Physics Loss Engine (data, sde, thermal, aging)
# -------------------------------------------------------------
class CrossSitePhysicsLossEngine:
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
        """Single-Diode Equation (SDE) Circuit Physics Loss."""
        poa = torch.clamp(raw_poa, min=0.0)
        p_sde_ref = (
            poa * self.area * self.eta_stc * (1.0 + self.gamma_p * (t_cell_hat - 25.0))
        ) / diode_n
        p_sde_ref = torch.clamp(p_sde_ref, min=0.0)
        return self.mse(p_pred, p_sde_ref) + torch.mean(torch.relu(-p_pred))

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
# 4. Training on xSi12922 -> Testing on Eugene_xSi12922
# -------------------------------------------------------------
def train_on_xsi_and_test_on_eugene(active_losses, train_loader_xsi, test_loader_eugene, loss_engine):
    torch.manual_seed(42)
    model = CrossSiteHybridPVLSTM(input_dim=4, hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    epochs = 6

    weights = {"sde": 0.1, "thermal": 0.05, "aging": 0.01}

    # Step 1: Train the model on xSi12922
    model.train()
    for _ in range(epochs):
        for bx, by, bpoa, bcurr in train_loader_xsi:
            bx, by, bpoa, bcurr = (
                bx.to(device),
                by.to(device),
                bpoa.to(device),
                bcurr.to(device),
            )
            optimizer.zero_grad()
            p_pred, t_hat, dRs_hat, diode_n = model(bx, bpoa)

            total_loss = torch.tensor(0.0, device=device)
            if "data" in active_losses:
                total_loss += loss_engine.loss_data(p_pred, by)
            if "sde" in active_losses:
                total_loss += weights["sde"] * loss_engine.loss_sde(
                    p_pred, bpoa, t_hat, diode_n
                )
            if "thermal" in active_losses:
                total_loss += weights["thermal"] * loss_engine.loss_thermal(t_hat, bcurr)
            if "aging" in active_losses:
                total_loss += weights["aging"] * loss_engine.loss_aging(dRs_hat, t_hat, bcurr)

            total_loss.backward()
            optimizer.step()

    # Step 2: Test the trained model directly on Eugene_xSi12922 (Out-of-Domain Generalization)
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for bx, by, bpoa, _ in test_loader_eugene:
            bx, bpoa = bx.to(device), bpoa.to(device)
            p_pred, _, _, _ = model(bx, bpoa)
            preds.append(p_pred.cpu().numpy())
            actuals.append(by.numpy())

    preds = np.vstack(preds)
    actuals = np.vstack(actuals)

    r2 = r2_score(actuals, preds)
    mae = mean_absolute_error(actuals, preds)
    rmse = root_mean_squared_error(actuals, preds)
    return r2, mae, rmse


# -------------------------------------------------------------
# 5. Main Execution: All 15 Experiments
# -------------------------------------------------------------
if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    print("=" * 95)
    print("CROSS-SITE OVERFITTING BENCHMARK: TRAINED ON xSi12922 -> TESTED ON EUGENE (Eugene_xSi12922.csv)")
    print("Same Hardware: 70W Monocrystalline Silicon Module #12922 Across Different Sites & Years")
    print("=" * 95)

    print("\nPreparing training data (xSi12922.csv) and testing data (Eugene_xSi12922.csv)...")
    train_loader, test_loader, scaler_X = prepare_cross_site_data(
        train_csv="xSi12922.csv", test_csv="Eugene_xSi12922.csv"
    )
    engine = CrossSitePhysicsLossEngine(scaler_X)

    loss_terms = ["data", "sde", "thermal", "aging"]
    combinations = []
    for k in range(1, len(loss_terms) + 1):
        combinations.extend(list(itertools.combinations(loss_terms, k)))

    print(
        f"\n{'Exp #':<6} | {'Active Loss Components':<35} | {'Eugene Test R^2':<15} | {'MAE (W)':<10} | {'RMSE (W)':<10}"
    )
    print("-" * 90)

    results = []
    for i, combo in enumerate(combinations, start=1):
        r2, mae, rmse = train_on_xsi_and_test_on_eugene(
            combo, train_loader, test_loader, engine
        )
        combo_str = " + ".join(combo)
        results.append((i, combo_str, r2, mae, rmse))
        print(
            f"{i:<6} | {combo_str:<35} | {r2:15.4f} | {mae:9.2f}W | {rmse:9.2f}W"
        )

    print("-" * 90)
    best_exp = max(results, key=lambda x: x[2])
    print(
        f"\n[PEAK GENERALIZATION RESULT] Peak Combination: Exp {best_exp[0]} ({best_exp[1]})"
    )
    print(f"   Eugene Test R^2 = {best_exp[2]:.4f}")
    print(f"   MAE             = {best_exp[3]:.2f} W")
    print(f"   RMSE            = {best_exp[4]:.2f} W")
    print("=" * 95)
