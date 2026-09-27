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

    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["sin_day"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["cos_day"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    return df


class PVTimeSeriesDataset(Dataset):
    def __init__(self, sequences_x, targets_y, current_x, telemetry_raw):
        self.X = torch.tensor(sequences_x, dtype=torch.float32)
        self.y = torch.tensor(targets_y, dtype=torch.float32)
        self.curr_x = torch.tensor(current_x, dtype=torch.float32)
        self.raw_telemetry = torch.tensor(telemetry_raw, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            self.X[idx],
            self.y[idx],
            self.curr_x[idx],
            self.raw_telemetry[idx],
        )


def prepare_single_diode_data(
    csv_path="xSi12922.csv", seq_len=12, split_ratio=0.8
):
    df = pd.read_csv(csv_path)
    df = build_temporal_features(df)

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

    # Telemetry columns for Single-Diode Equation: Vmp, Imp, Isc, Voc, T_cell
    telemetry_cols = [
        "Vmp (V)",
        "Imp (A)",
        "Isc (A)",
        "Voc (V)",
        "PV module back surface temperature (degC)",
    ]

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
        seqs, targets, curr_features, raw_telem = [], [], [], []
        for _, day_group in sub_df.groupby("day_id"):
            if len(day_group) <= seq_len:
                continue

            x_scaled = scaler_X.transform(day_group[feature_cols].values)
            y_scaled = scaler_y.transform(day_group[[target_col]].values)
            telem_vals = day_group[telemetry_cols].values

            for i in range(len(day_group) - seq_len):
                seqs.append(x_scaled[i : i + seq_len])
                targets.append(y_scaled[i + seq_len])
                curr_features.append(x_scaled[i + seq_len])
                raw_telem.append(telem_vals[i + seq_len])

        return (
            np.array(seqs),
            np.array(targets),
            np.array(curr_features),
            np.array(raw_telem),
        )

    X_train_seq, y_train_seq, curr_train, telem_train = extract_day_windows(
        df_train
    )
    X_test_seq, y_test_seq, curr_test, telem_test = extract_day_windows(
        df_test
    )

    train_ds = PVTimeSeriesDataset(
        X_train_seq, y_train_seq, curr_train, telem_train
    )
    test_ds = PVTimeSeriesDataset(X_test_seq, y_test_seq, curr_test, telem_test)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    return train_loader, test_loader, scaler_X, scaler_y


# -------------------------------------------------------------
# 2. Deep Learning Multi-Head LSTM + Single-Diode Parameter Generator
# -------------------------------------------------------------
class SingleDiodePINNLSTM(nn.Module):
    """Deep Learning Architecture that extracts dynamic Single-Diode Model parameters
    (I_ph, I_0, R_s, R_sh, n) from temporal sequence windows."""

    def __init__(self, input_dim=8, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1
        )
        self.fc_shared = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
        )

        # Direct Power Forecast Head (Neural Black-Box Branch)
        self.head_power_nn = nn.Linear(32, 1)

        # Thermal Head
        self.head_thermal = nn.Linear(32, 1)

        # Dynamic Single-Diode Parameter Extraction Heads (Deep Learning Physics Inversion)
        self.head_Rs = nn.Sequential(
            nn.Linear(32, 1), nn.Softplus()
        )  # Series Resistance Rs > 0
        self.head_Rsh = nn.Sequential(
            nn.Linear(32, 1), nn.Softplus()
        )  # Shunt Resistance Rsh > 0
        self.head_n = nn.Sequential(
            nn.Linear(32, 1), nn.Sigmoid()
        )  # Diode Ideality n in (1, 2)
        self.head_I0 = nn.Sequential(
            nn.Linear(32, 1), nn.Sigmoid()
        )  # Reverse saturation scale
        self.head_aging_rate = nn.Sequential(
            nn.Linear(32, 1), nn.Softplus()
        )  # Aging rate dD/dt

    def forward(self, x):
        out, _ = self.lstm(x)
        context = self.fc_shared(out[:, -1, :])

        p_nn_norm = self.head_power_nn(context)
        t_cell_hat = self.head_thermal(context)

        # Single-Diode Parameter Outputs (scaled to physical ranges)
        Rs_hat = self.head_Rs(context) * 0.1 + 0.001  # Rs in (0.001, 0.1) Ohms
        Rsh_hat = self.head_Rsh(context) * 1000.0 + 100.0  # Rsh in Ohms
        n_hat = self.head_n(context) * 1.0 + 1.0  # n in (1.0, 2.0)
        I0_hat = self.head_I0(context) * 1e-6 + 1e-9  # Reverse saturation
        dD_dt_hat = self.head_aging_rate(context)

        return (
            p_nn_norm,
            t_cell_hat,
            Rs_hat,
            Rsh_hat,
            n_hat,
            I0_hat,
            dD_dt_hat,
        )


# -------------------------------------------------------------
# 3. Single-Diode Circuit & Multi-Physics Loss Engine
# -------------------------------------------------------------
class SingleDiodePhysicsLossEngine:
    def __init__(self, scaler_X, scaler_y):
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        self.mse = nn.MSELoss()

        self.poa_mean = scaler_X.mean_[0]
        self.poa_scale = scaler_X.scale_[0]
        self.tamb_mean = scaler_X.mean_[1]
        self.tamb_scale = scaler_X.scale_[1]
        self.rh_mean = scaler_X.mean_[2]
        self.rh_scale = scaler_X.scale_[2]
        self.p_mean = scaler_y.mean_[0]
        self.p_scale = scaler_y.scale_[0]

        # Thermal voltage constant V_t = k*T/q ~ 0.026V at 300K for 1 cell
        # For a standard module with Ns=60 cells in series: Ns * V_t ~ 1.56V
        self.Ns_Vt = 1.56
        self.k_B = 8.617333e-5
        self.E_a = 0.35
        self.A_arrh = 1e-4
        self.gamma_rh = 0.005

    def loss_single_diode_qem(
        self, p_nn_norm, Rs_hat, Rsh_hat, n_hat, I0_hat, raw_telem
    ):
        """Calculates exact Single-Diode Model (SDM) Circuit Loss (L_QEM / L_diode).
        Evaluates current-voltage implicit diode equation:
        I = Iph - I0 * [exp((V + I*Rs) / (n*V_t)) - 1] - (V + I*Rs)/Rsh
        """
        Vmp = raw_telem[:, 0:1]
        Imp = raw_telem[:, 1:2]
        Isc = raw_telem[:, 2:3]

        # Photo-generated current Iph ~ Isc
        Iph = Isc

        # Diode thermal voltage scaled by ideality factor n
        nV_t = n_hat * self.Ns_Vt

        # Exponential diode term
        arg = (Vmp + Imp * Rs_hat) / nV_t
        arg = torch.clamp(arg, max=50.0)  # Numerical safety floor
        exp_term = torch.exp(arg) - 1.0

        # Shunt current
        I_sh = (Vmp + Imp * Rs_hat) / Rsh_hat

        # Calculated current via Single-Diode Equation
        I_calc = Iph - I0_hat * exp_term - I_sh
        P_diode_calc = Vmp * I_calc

        # Normalize predicted power from neural network
        p_pred_watts = p_nn_norm * self.p_scale + self.p_mean

        # Single-Diode Circuit Loss
        diode_loss = self.mse(p_pred_watts, P_diode_calc)
        return diode_loss / (self.p_scale**2)

    def loss_data(self, p_nn_norm, y_norm):
        """Supervised data fitting loss."""
        return self.mse(p_nn_norm, y_norm)

    def loss_thermal(self, t_cell_hat, current_x_norm):
        """Heat balance loss."""
        poa = current_x_norm[:, 0:1] * self.poa_scale + self.poa_mean
        t_amb = current_x_norm[:, 1:2] * self.tamb_scale + self.tamb_mean
        t_expected = t_amb + (poa / (26.9 + 1.06 * 1.5))
        return torch.mean((t_cell_hat - t_expected) ** 2) / 100.0

    def loss_arrhenius(self, dD_dt_hat, t_cell_hat, current_x_norm):
        """Arrhenius degradation loss."""
        rh = current_x_norm[:, 2:3] * self.rh_scale + self.rh_mean
        rh = torch.clamp(rh, min=0.0, max=100.0)
        t_kelvin = torch.clamp(t_cell_hat + 273.15, min=200.0)
        degrad_target = (
            self.A_arrh
            * torch.exp(-self.E_a / (self.k_B * t_kelvin))
            * (1.0 + self.gamma_rh * rh)
        )
        return self.mse(torch.log1p(dD_dt_hat), torch.log1p(degrad_target))


# -------------------------------------------------------------
# 4. Training & Combinatoric Evaluation Loop
# -------------------------------------------------------------
def train_and_evaluate(active_losses, train_loader, test_loader, loss_engine):
    model = SingleDiodePINNLSTM(input_dim=8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = 6

    weights = {"diode": 0.1, "thermal": 0.05, "arrhenius": 0.01}

    model.train()
    for _ in range(epochs):
        for seq_x, y_target, curr_x, raw_telem in train_loader:
            seq_x, y_target, curr_x, raw_telem = (
                seq_x.to(device),
                y_target.to(device),
                curr_x.to(device),
                raw_telem.to(device),
            )
            optimizer.zero_grad()
            (
                p_nn,
                t_hat,
                Rs_hat,
                Rsh_hat,
                n_hat,
                I0_hat,
                dD_dt_hat,
            ) = model(seq_x)

            total_loss = torch.tensor(0.0, device=device)
            if "data" in active_losses:
                total_loss += loss_engine.loss_data(p_nn, y_target)
            if "diode" in active_losses:
                total_loss += weights[
                    "diode"
                ] * loss_engine.loss_single_diode_qem(
                    p_nn, Rs_hat, Rsh_hat, n_hat, I0_hat, raw_telem
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
        for seq_x, y_target, _, _ in test_loader:
            seq_x = seq_x.to(device)
            p_nn, _, _, _, _, _, _ = model(seq_x)
            preds.append(p_nn.cpu().numpy())
            actuals.append(y_target.numpy())

    preds = np.vstack(preds)
    actuals = np.vstack(actuals)

    preds_w = preds * loss_engine.p_scale + loss_engine.p_mean
    actuals_w = actuals * loss_engine.p_scale + loss_engine.p_mean

    return r2_score(actuals_w, preds_w)


# -------------------------------------------------------------
# 5. Run Benchmark
# -------------------------------------------------------------
if __name__ == "__main__":
    train_loader, test_loader, scaler_X, scaler_y = prepare_single_diode_data(
        "xSi12922.csv"
    )
    engine = SingleDiodePhysicsLossEngine(scaler_X, scaler_y)

    loss_terms = ["data", "diode", "thermal", "arrhenius"]
    combinations = []
    for k in range(1, len(loss_terms) + 1):
        combinations.extend(list(itertools.combinations(loss_terms, k)))

    print(
        f"{'Experiment':<5} | {'Active Loss Components':<42} | {'Test R² Score':<10}"
    )
    print("-" * 65)
    for i, combo in enumerate(combinations, start=1):
        score = train_and_evaluate(combo, train_loader, test_loader, engine)
        combo_str = " + ".join(combo)
        print(f"{i:<5} | {combo_str:<42} | {score:10.4f}")
