# =============================================================================
# SCRIPT 2: INFERENCE & PREDICTION AGGREGATION
# =============================================================================
# This script loads the trained models and generates predictions for the held-out test set.
# It performs the following critical functions:
# 1. Reconstructs the exact training state (required for ExactGP inference).
# 2. Computes both point predictions (Mean) and uncertainty intervals (Confidence Intervals).
# 3. Inverse-transforms data back to the original physical scale (W/m^2).
# 4. Aggregates results from all models into a single 'Golden CSV' for analysis.

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import gpytorch
import pickle
import os
import sys

# CONFIGURATION
# Enforce double precision to match training stability requirements.
torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"EXPORTING PREDICTIONS + UNCERTAINTY ON: {device}")
os.makedirs("results", exist_ok=True)

# ==========================================
# 1. MODEL CLASS DEFINITIONS
# ==========================================
# These classes must exactly match the architecture defined in the training pipeline
# to successfully load the state dictionaries.

class StandardGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, like):
        super().__init__(train_x, train_y, like)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))

class MaternGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, like):
        super().__init__(train_x, train_y, like)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5))
    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))

class ReferenceLSTM(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 32, 3, batch_first=True)
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
    def forward(self, x):
        out, _ = self.lstm(x)
        x = self.relu(self.fc1(out[:, -1, :]))
        return self.fc2(x)

# ==========================================
# 2. INFERENCE LOGIC
# ==========================================
def export_dataset(csv_path, dataset_name):
    print(f"\nPROCESSING: {dataset_name}")
    if not os.path.exists(csv_path): return

    # A. RECOVER DATA & SPLIT
    # We must replicate the exact data loading and splitting process used in training.
    # Any deviation here would result in data leakage (training on test data) or shape mismatches.
    df = pd.read_csv(csv_path)
    
    # Standardize Time Columns
    if 'Year' in df.columns and 'Minute' in df.columns:
        df['Time'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])
    
    col_map = {'period_end':'Time','ghi':'GHI','air_temp':'AirTemp','wind_speed_10m':'WindSpeed',
               'zenith':'Zenith','clearsky_ghi':'GHI_ClearSky','cloud_opacity':'CloudOpacity',
               'Tamb':'AirTemp','WindVel':'WindSpeed','Zenith':'Zenith','Cloudopacity':'CloudOpacity'}
    for col in df.columns:
        if col in col_map: df.rename(columns={col: col_map[col]}, inplace=True)

    # Estimate Clear Sky GHI if missing (using basic Cosine Zenith model)
    if 'GHI_ClearSky' not in df.columns:
        z_rad = np.radians(df['Zenith'])
        df['GHI_ClearSky'] = 950 * (np.cos(z_rad) ** 1.15)
        df['GHI_ClearSky'] = df['GHI_ClearSky'].clip(lower=10)

    df['Time'] = pd.to_datetime(df['Time'])
    df.sort_values('Time', inplace=True)
    
    # Feature Engineering (Physics Constraints)
    df['IsDay'] = df['Zenith'] < 85
    df['CSI'] = df['GHI'] / df['GHI_ClearSky']
    df['CSI'] = df['CSI'].clip(0, 2.0).replace([np.inf, -np.inf], 0)
    
    # Create Smart Persistence Baseline (Persistence of Clearsky Index)
    # This assumes the cloudiness at t-1 persists to t.
    df['CSI_Persist'] = df['CSI'].shift(1)

    # Re-apply Sampling Mask to identify Train/Test sets
    df_day = df[df['IsDay']].reset_index(drop=True)
    np.random.seed(42) # CRITICAL: Must match training seed exactly
    n_samples = 15000 if len(df_day) > 15000 else len(df_day)
    indices = np.sort(np.random.choice(len(df_day), n_samples, replace=False))
    df_day = df_day.iloc[indices].reset_index(drop=True)
    
    split_idx = int(len(df_day) * 0.8)
    train_df = df_day.iloc[:split_idx]
    test_df = df_day.iloc[split_idx:].copy()
    
    # Calculate GHI Persistence for the Test Set
    test_df['GHI_Persist'] = test_df['CSI_Persist'] * test_df['GHI_ClearSky']
    test_df.dropna(subset=['GHI_Persist'], inplace=True)

    # Initialize Output Container
    out_df = pd.DataFrame()
    out_df['Time'] = test_df['Time'].values
    out_df['Actual_GHI'] = test_df['GHI'].values
    out_df['Smart_Persistence'] = test_df['GHI_Persist'].values

    # B. LOAD ARTIFACTS
    # Load the scalers fitted during training to ensure valid inverse transformation.
    with open(f"scalers/{dataset_name}_scalers.pkl", "rb") as f:
        sc = pickle.load(f)

    cols_X = ['AirTemp', 'WindSpeed', 'Zenith', 'CloudOpacity']
    
    # RECONSTRUCT TRAINING TENSORS
    # ExactGPs differ from Deep Learning: they are non-parametric. 
    # To make a prediction on a new point x*, the model requires the kernel matrix 
    # computed between x* and ALL training points X_train.
    # Therefore, we must reload X_train and y_train onto the device.
    train_x = torch.tensor(sc['X'].transform(train_df[cols_X])).to(device)
    train_y_csi = torch.tensor(sc['y_phys'].transform(train_df[['CSI']])).view(-1).to(device)
    train_y_ghi = torch.tensor(sc['y_base'].transform(train_df[['GHI']])).view(-1).to(device)
    
    test_x = torch.tensor(sc['X'].transform(test_df[cols_X])).to(device)

    # C. RUN INFERENCE FOR ALL MODELS
    
    # 1. BASELINE GP (Raw GHI)
    # This model predicts GHI directly. We capture both Mean and Uncertainty (Confidence Interval).
    print("   Predicting Baseline GP (with Uncertainty)...")
    like = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = StandardGP(train_x, train_y_ghi, like).to(device)
    ckpt = torch.load(f"saved_models/{dataset_name}_baseline_gp_raw.pth")
    model.load_state_dict(ckpt['model']); like.load_state_dict(ckpt['like'])
    
    model.eval(); like.eval()
    with torch.no_grad(), gpytorch.settings.max_cg_iterations(2000):
        # 'like(model(x))' adds the observational noise to the predictive distribution
        out = like(model(test_x))
        mean, std = out.mean, out.stddev
        
        # Inverse Transform Mean
        out_df['Pred_Baseline_Raw'] = sc['y_base'].inverse_transform(mean.cpu().numpy().reshape(-1,1)).ravel()
        
        # Calculate & Inverse Transform Uncertainty Bounds (95% CI)
        # Note: We transform the bounds themselves, which handles scaling correctly.
        low_raw = (mean - 1.96*std).cpu().numpy().reshape(-1,1)
        high_raw = (mean + 1.96*std).cpu().numpy().reshape(-1,1)
        out_df['Baseline_Lower_CI'] = sc['y_base'].inverse_transform(low_raw).ravel()
        out_df['Baseline_Upper_CI'] = sc['y_base'].inverse_transform(high_raw).ravel()

    # 2. LSTM (Raw GHI)
    print("   Predicting LSTM Raw...")
    lstm = ReferenceLSTM(4).double().to(device)
    lstm.load_state_dict(torch.load(f"saved_models/{dataset_name}_lstm_raw.pth"))
    lstm.eval()
    with torch.no_grad():
        raw = lstm(test_x.unsqueeze(1)).cpu().numpy()
        out_df['Pred_LSTM_Raw'] = sc['y_base'].inverse_transform(raw).ravel()

    # 3. LSTM (Physics-Informed CSI)
    print("   Predicting LSTM CSI...")
    lstm.load_state_dict(torch.load(f"saved_models/{dataset_name}_lstm_csi.pth"))
    lstm.eval()
    with torch.no_grad():
        raw = lstm(test_x.unsqueeze(1)).cpu().numpy()
        csi = sc['y_phys'].inverse_transform(raw).ravel()
        # Physics Re-projection: GHI = CSI * ClearSky
        out_df['Pred_LSTM_CSI'] = csi * test_df['GHI_ClearSky'].values

    # 4. PHYSICS GP (Matern Kernel)
    print("   Predicting Physics Matern...")
    model = MaternGP(train_x, train_y_csi, like).to(device)
    ckpt = torch.load(f"saved_models/{dataset_name}_physics_gp_matern.pth")
    model.load_state_dict(ckpt['model']); like.load_state_dict(ckpt['like'])
    model.eval(); like.eval()
    with torch.no_grad(), gpytorch.settings.max_cg_iterations(2000):
        mean = like(model(test_x)).mean
        csi = sc['y_phys'].inverse_transform(mean.cpu().numpy().reshape(-1,1)).ravel()
        out_df['Pred_Physics_Matern'] = csi * test_df['GHI_ClearSky'].values

    # 5. PHYSICS GP (RBF Kernel) - The Champion Model
    # We capture full uncertainty for this model to demonstrate the benefits of the physics approach.
    print("   Predicting Physics RBF (Best)...")
    model = StandardGP(train_x, train_y_csi, like).to(device)
    ckpt = torch.load(f"saved_models/{dataset_name}_physics_gp_rbf.pth")
    model.load_state_dict(ckpt['model']); like.load_state_dict(ckpt['like'])
    model.eval(); like.eval()
    with torch.no_grad(), gpytorch.settings.max_cg_iterations(2000):
        out = like(model(test_x))
        mean, std = out.mean, out.stddev
        
        # Inverse transform CSI predictions
        csi = sc['y_phys'].inverse_transform(mean.cpu().numpy().reshape(-1,1)).ravel()
        low = sc['y_phys'].inverse_transform((mean - 1.96*std).cpu().numpy().reshape(-1,1)).ravel()
        high = sc['y_phys'].inverse_transform((mean + 1.96*std).cpu().numpy().reshape(-1,1)).ravel()
        
        # Physics Re-projection: Map CSI uncertainty back to GHI space
        # Confidence intervals scale dynamically with solar geometry (ClearSky GHI).
        out_df['Pred_Physics_RBF'] = csi * test_df['GHI_ClearSky'].values
        out_df['RBF_Lower_CI'] = low * test_df['GHI_ClearSky'].values
        out_df['RBF_Upper_CI'] = high * test_df['GHI_ClearSky'].values

    # SAVE TO CSV
    # This file serves as the single source of truth for all subsequent analysis tables and plots.
    save_path = f"results/{dataset_name}_all_predictions.csv"
    out_df.to_csv(save_path, index=False)
    print(f"Saved Golden CSV: {save_path}")

# EXECUTE
# Process both datasets to prepare for full comparative analysis.
export_dataset("60mins.csv", "60min")
export_dataset("5mins.csv", "5min")