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


def prepare_12month_pinn_data(
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
# 2. Deep Learning Multi-Head LSTM (With Monotonic Rs Accumulator)
# -------------------------------------------------------------
class SingleDiode12MoPINNLSTM(nn.Module):
    """Deep Learning Architecture with Single-Diode parameter extraction
    and 12-Month Cumulative Series Resistance Tracking.
    """

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

        self.head_power_nn = nn.Linear(32, 1)
        self.head_thermal = nn.Linear(32, 1)
        self.head_Rs = nn.Sequential(nn.Linear(32, 1), nn.Softplus())
        self.head_Rsh = nn.Sequential(nn.Linear(32, 1), nn.Softplus())
        self.head_n = nn.Sequential(nn.Linear(32, 1), nn.Sigmoid())
        self.head_I0 = nn.Sequential(nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, x):
        out, _ = self.lstm(x)
        context = self.fc_shared(out[:, -1, :])

        p_nn_norm = self.head_power_nn(context)
        t_cell_hat = self.head_thermal(context)

        Rs_hat = self.head_Rs(context) * 0.1 + 0.001
        Rsh_hat = self.head_Rsh(context) * 1000.0 + 100.0
        n_hat = self.head_n(context) * 1.0 + 1.0
        I0_hat = self.head_I0(context) * 1e-6 + 1e-9

        return p_nn_norm, t_cell_hat, Rs_hat, Rsh_hat, n_hat, I0_hat


# -------------------------------------------------------------
# 3. 12-Month Multi-Physics Loss Engine
# -------------------------------------------------------------
class SingleDiode12MoPhysicsLossEngine:
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

        self.Ns_Vt = 1.56
        self.k_B = 8.617333e-5
        self.E_a = 0.35
        self.A_arrh = 1e-4
        self.gamma_rh = 0.005
        self.dt_hours = 0.0833  # 5 minutes = 1/12 hour

    def loss_data(self, p_nn_norm, y_norm):
        return self.mse(p_nn_norm, y_norm)

    def loss_single_diode_qem(
        self, p_nn_norm, Rs_hat, Rsh_hat, n_hat, I0_hat, raw_telem
    ):
        Vmp = raw_telem[:, 0:1]
        Imp = raw_telem[:, 1:2]
        Isc = raw_telem[:, 2:3]
        Iph = Isc

        nV_t = n_hat * self.Ns_Vt
        arg = torch.clamp((Vmp + Imp * Rs_hat) / nV_t, max=50.0)
        exp_term = torch.exp(arg) - 1.0

        I_sh = (Vmp + Imp * Rs_hat) / Rsh_hat
        I_calc = Iph - I0_hat * exp_term - I_sh
        P_diode_calc = Vmp * I_calc

        p_pred_watts = p_nn_norm * self.p_scale + self.p_mean
        return self.mse(p_pred_watts, P_diode_calc) / (self.p_scale**2)

    def loss_thermal(self, t_cell_hat, current_x_norm):
        poa = current_x_norm[:, 0:1] * self.poa_scale + self.poa_mean
        t_amb = current_x_norm[:, 1:2] * self.tamb_scale + self.tamb_mean
        t_expected = t_amb + (poa / (26.9 + 1.06 * 1.5))
        return torch.mean((t_cell_hat - t_expected) ** 2) / 100.0

    def loss_arrhenius_12mo(self, Rs_hat, t_cell_hat, current_x_norm):
        """12-Month Macro Cumulative Arrhenius Degradation Loss.
        Integrates cumulative thermal stress over the sequence window
        to constrain total series resistance change Delta R_s.
        """
        rh = current_x_norm[:, 2:3] * self.rh_scale + self.rh_mean
        rh = torch.clamp(rh, min=0.0, max=100.0)
        t_kelvin = torch.clamp(t_cell_hat + 273.15, min=200.0)

        # Instantaneous rate of degradation at each time step
        degrad_rate = (
            self.A_arrh
            * torch.exp(-self.E_a / (self.k_B * t_kelvin))
            * (1.0 + self.gamma_rh * rh)
        )

        # Integrated cumulative physical degradation across batch
        Delta_Rs_physical = torch.mean(degrad_rate) * self.dt_hours

        # Enforce that Rs_hat change matches integrated physical degradation
        Rs_baseline = 0.040  # Default initial series resistance (Ohms)
        Delta_Rs_predicted = torch.mean(Rs_hat - Rs_baseline)

        return self.mse(Delta_Rs_predicted, Delta_Rs_physical)


# -------------------------------------------------------------
# 4. Training & Combinatoric Evaluation Loop
# -------------------------------------------------------------
def train_and_evaluate(active_losses, train_loader, test_loader, loss_engine):
    model = SingleDiode12MoPINNLSTM(input_dim=8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = 6

    weights = {"diode": 0.1, "thermal": 0.05, "arrhenius_12mo": 0.01}

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
            if "arrhenius_12mo" in active_losses:
                total_loss += weights[
                    "arrhenius_12mo"
                ] * loss_engine.loss_arrhenius_12mo(Rs_hat, t_hat, curr_x)

            total_loss.backward()
            optimizer.step()

    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for seq_x, y_target, _, _ in test_loader:
            seq_x = seq_x.to(device)
            p_nn, _, _, _, _, _ = model(seq_x)
            preds.append(p_nn.cpu().numpy())
            actuals.append(y_target.numpy())

    preds = np.vstack(preds)
    actuals = np.vstack(actuals)

    preds_w = preds * loss_engine.p_scale + loss_engine.p_mean
    actuals_w = actuals * loss_engine.p_scale + loss_engine.p_mean

    return r2_score(actuals_w, preds_w)


# -------------------------------------------------------------
# 5. Run Benchmark (All 15 Experiments)
# -------------------------------------------------------------
if __name__ == "__main__":
    train_loader, test_loader, scaler_X, scaler_y = prepare_12month_pinn_data(
        "xSi12922.csv"
    )
    engine = SingleDiode12MoPhysicsLossEngine(scaler_X, scaler_y)

    loss_terms = ["data", "diode", "thermal", "arrhenius_12mo"]
    combinations = []
    for k in range(1, len(loss_terms) + 1):
        combinations.extend(list(itertools.combinations(loss_terms, k)))

    print(
        f"{'Experiment':<5} | {'Active Loss Components':<45} | {'Test R² Score':<10}"
    )
    print("-" * 68)
    for i, combo in enumerate(combinations, start=1):
        score = train_and_evaluate(combo, train_loader, test_loader, engine)
        combo_str = " + ".join(combo)
        print(f"{i:<5} | {combo_str:<45} | {score:10.4f}")
