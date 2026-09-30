
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

"""Evaluation of Study 7 PINN on mSi0166.csv.

Covers two rigorous evaluations:
Evaluation A (Cross-Asset Zero-Shot Transfer):
  Takes the model trained on xSi12922.csv and evaluates directly on mSi0166.csv.
Evaluation B (Native Retraining Benchmark):
  Trains the hybrid PINN architecture directly on mSi0166.csv across all 15 loss combinations.
"""

import sys
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
from torch.utils.data import DataLoader, Dataset
import study_hybrid_kpca_wavelet_pinn as base

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    sys.stdout.reconfigure(line_buffering=True)
    print("=" * 85)
    print("Cross-Asset Evaluation on mSi0166.csv (15 Experiments)")
    print("=" * 85)

    # 1. Prepare data for both modules
    print("Preparing xSi12922 and mSi0166 datasets...")
    train_loader_x, test_loader_x, scaler_X_x = base.prepare_hybrid_pinn_data("xSi12922.csv")
    engine_x = base.HybridPhysicsLossEngine(scaler_X_x)

    train_loader_m, test_loader_m, scaler_X_m = base.prepare_hybrid_pinn_data("mSi0166.csv")
    engine_m = base.HybridPhysicsLossEngine(scaler_X_m)

    loss_terms = ["data", "sde", "thermal", "aging"]
    combinations = []
    for k in range(1, len(loss_terms) + 1):
        combinations.extend(list(itertools.combinations(loss_terms, k)))

    weights = {"sde": 0.1, "thermal": 0.05, "aging": 0.01}
    epochs = 6

    # -------------------------------------------------------------
    # Evaluation A: Trained on xSi12922 -> Evaluated on mSi0166
    # -------------------------------------------------------------
    print("\n" + "=" * 85)
    print("EVALUATION A: Models Trained on xSi12922 -> Evaluated Directly on mSi0166 (Zero-Shot Transfer)")
    print("=" * 85)
    print(f"{'Exp #':<6} | {'Active Loss Components':<35} | {'mSi0166 R^2':<12} | {'MAE (W)':<10} | {'RMSE (W)':<10}")
    print("-" * 85)

    transfer_results = []
    for i, combo in enumerate(combinations, 1):
        torch.manual_seed(42)
        model = base.HybridPVTimeSeriesLSTM(input_dim=4, hidden_dim=64).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

        # Train on xSi12922
        model.train()
        for _ in range(epochs):
            for bx, by, bpoa, bcurr in train_loader_x:
                bx, by, bpoa, bcurr = bx.to(device), by.to(device), bpoa.to(device), bcurr.to(device)
                optimizer.zero_grad()
                p_pred, t_hat, dRs_hat, diode_n = model(bx, bpoa)
                total_loss = torch.tensor(0.0, device=device)
                if "data" in combo:
                    total_loss += engine_x.loss_data(p_pred, by)
                if "sde" in combo:
                    total_loss += weights["sde"] * engine_x.loss_sde(p_pred, bpoa, t_hat, diode_n)
                if "thermal" in combo:
                    total_loss += weights["thermal"] * engine_x.loss_thermal(t_hat, bcurr)
                if "aging" in combo:
                    total_loss += weights["aging"] * engine_x.loss_aging(dRs_hat, t_hat, bcurr)
                total_loss.backward()
                optimizer.step()

        # Test directly on mSi0166 test data
        model.eval()
        preds, actuals = [], []
        with torch.no_grad():
            for bx, by, bpoa, _ in test_loader_m:
                bx, bpoa = bx.to(device), bpoa.to(device)
                p_pred, _, _, _ = model(bx, bpoa)
                preds.append(p_pred.cpu().numpy())
                actuals.append(by.numpy())

        preds = np.vstack(preds)
        actuals = np.vstack(actuals)
        r2 = r2_score(actuals, preds)
        mae = mean_absolute_error(actuals, preds)
        rmse = root_mean_squared_error(actuals, preds)
        combo_str = " + ".join(combo)
        transfer_results.append((i, combo_str, r2, mae, rmse))
        print(f"{i:<6} | {combo_str:<35} | {r2:12.4f} | {mae:9.2f}W | {rmse:9.2f}W")

    # -------------------------------------------------------------
    # Evaluation B: Trained and Evaluated Directly on mSi0166
    # -------------------------------------------------------------
    print("\n" + "=" * 85)
    print("EVALUATION B: Models Trained Directly on mSi0166 -> Evaluated on mSi0166 (Native Benchmark)")
    print("=" * 85)
    print(f"{'Exp #':<6} | {'Active Loss Components':<35} | {'mSi0166 R^2':<12} | {'MAE (W)':<10} | {'RMSE (W)':<10}")
    print("-" * 85)

    native_results = []
    for i, combo in enumerate(combinations, 1):
        r2, mae, rmse = base.train_and_evaluate(combo, train_loader_m, test_loader_m, engine_m)
        combo_str = " + ".join(combo)
        native_results.append((i, combo_str, r2, mae, rmse))
        print(f"{i:<6} | {combo_str:<35} | {r2:12.4f} | {mae:9.2f}W | {rmse:9.2f}W")

    print("\n" + "=" * 85)
    print("ALL 15 EXPERIMENT READINGS COMPLETED FOR BOTH EVALUATION MODES!")
    print("=" * 85)

if __name__ == "__main__":
    main()
