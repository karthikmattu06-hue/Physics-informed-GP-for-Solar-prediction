# =============================================================================
# SCRIPT 1: ROBUST TRAINING PIPELINE
# =============================================================================
# This script orchestrates the end-to-end training process for solar irradiance forecasting.
# It handles data ingestion, preprocessing (physics-informed feature engineering),
# model instantiation (GPs and LSTMs), and training loops with automated logging and state saving.
# It is designed to be idempotent: re-running it skips already trained models to save compute.

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import gpytorch
import os
import pickle
import logging
import sys
from sklearn.preprocessing import MinMaxScaler
import warnings

# --- 1. SETUP LOGGING & SAFETY ---
# Create directory structure for artifacts to ensure reproducibility and organization.
# - saved_models/: Stores trained model weights (.pth files).
# - scalers/: Stores sklearn scalers for inverse transformation during inference.
# - logs/: Stores execution logs for debugging and auditing.
os.makedirs("saved_models", exist_ok=True)
os.makedirs("scalers", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Configure Logging (Writes to terminal AND file)
# Using a dual-handler setup allows real-time monitoring via stdout while preserving
# a permanent record in 'training_run.log' for post-mortem analysis.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/training_run.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger()

# Hardware Check
# Suppress non-critical warnings to keep logs clean.
# Enforce double precision (float64) for numerical stability, which is crucial for 
# Gaussian Processes to prevent Cholesky decomposition errors.
# Automatically detect CUDA availability to leverage GPU acceleration if present.
warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"STARTING PIPELINE. Hardware detected: {device}")

# ==========================================
# 2. MODEL DEFINITIONS
# ==========================================

class StandardGP(gpytorch.models.ExactGP):
    """
    Standard Gaussian Process model using the Radial Basis Function (RBF) kernel.
    This serves as the baseline model (when trained on raw GHI) and one variation
    of the physics-informed model (when trained on CSI).
    
    Args:
        train_x (Tensor): Training features.
        train_y (Tensor): Training targets.
        like (GaussianLikelihood): The likelihood function handling observation noise.
    """
    def __init__(self, train_x, train_y, like):
        super().__init__(train_x, train_y, like)
        # ConstantMean assumes a constant average value across the domain.
        self.mean_module = gpytorch.means.ConstantMean()
        # ScaleKernel adds an output scale parameter to the RBF kernel, allowing the model
        # to learn the signal variance magnitude.
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))

class MaternGP(gpytorch.models.ExactGP):
    """
    Gaussian Process model using the Matern 3/2 kernel (nu=1.5).
    The Matern kernel is less smooth than RBF, making it often better suited for
    modeling physical processes with sharp transitions or roughness (like rapidly changing cloud cover).
    """
    def __init__(self, train_x, train_y, like):
        super().__init__(train_x, train_y, like)
        self.mean_module = gpytorch.means.ConstantMean()
        # nu=1.5 corresponds to Matern 3/2, offering moderate smoothness.
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5))

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))

class ReferenceLSTM(nn.Module):
    """
    Long Short-Term Memory (LSTM) network serving as the Deep Learning benchmark.
    It processes temporal sequences to predict the next time step, providing a comparison
    against the probabilistic GP approach.
    
    Structure:
        - LSTM Layer: Captures temporal dependencies (hidden size=32).
        - Fully Connected Layers: Maps LSTM output to the final scalar prediction.
        - ReLU Activation: Adds non-linearity.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 32, 3, batch_first=True)
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Pass through LSTM layer
        out, _ = self.lstm(x)
        # Take the output of the last time step for prediction
        x = self.relu(self.fc1(out[:, -1, :]))
        return self.fc2(x)

# ==========================================
# 3. TRAINING CORE
# ==========================================
def run_training_cycle(csv_path, dataset_name):
    """
    Main driver function to process a specific dataset (5-min or 60-min resolution).
    It handles loading, feature engineering, scaling, and triggering training for all model variants.
    """
    logger.info(f"{'='*40}")
    logger.info(f"PROCESSING DATASET: {dataset_name}")
    logger.info(f"{'='*40}")
    
    # A. LOAD DATA
    if not os.path.exists(csv_path):
        logger.error(f"File not found: {csv_path}. Skipping.")
        return

    try:
        df = pd.read_csv(csv_path)
        
        # Standardize column names to ensure consistency across different data sources.
        col_map = {'period_end':'Time','ghi':'GHI','air_temp':'AirTemp','wind_speed_10m':'WindSpeed',
                   'zenith':'Zenith','clearsky_ghi':'GHI_ClearSky','cloud_opacity':'CloudOpacity'}
        for col in df.columns:
            if col in col_map: df.rename(columns={col: col_map[col]}, inplace=True)
        
        df['Time'] = pd.to_datetime(df['Time'])
        df.sort_values('Time', inplace=True)

        # Physics Features Engineering
        # Filter for daylight hours (Zenith < 85 degrees) to remove night-time noise (zeros).
        df['IsDay'] = df['Zenith'] < 85
        
        # Calculate Clearsky Index (CSI) = GHI / Clearsky GHI.
        # This removes the deterministic solar geometry component, leaving the stochastic cloud effects.
        # We clip CSI to [0, 2.0] to handle numerical instability at sunrise/sunset where GHI_ClearSky ~ 0.
        df['CSI'] = df['GHI'] / df['GHI_ClearSky']
        df['CSI'] = df['CSI'].clip(0, 2.0).replace([np.inf, -np.inf], 0)
        
        # Split & Sampling
        # We only train on daylight data to focus the model on relevant physics.
        df_day = df[df['IsDay']].reset_index(drop=True)
        
        # Subsample to 15,000 points if dataset is large to manage GP computational complexity (O(N^3)).
        # Random sampling ensures the training set covers various seasonal and weather conditions.
        np.random.seed(42)
        n_samples = 15000 if len(df_day) > 15000 else len(df_day)
        indices = np.sort(np.random.choice(len(df_day), n_samples, replace=False))
        df_day = df_day.iloc[indices].reset_index(drop=True)
        
        # 80/20 Train-Test Split (Chronological split not strictly enforced due to random sampling above,
        # but index sorting preserves relative temporal order within the sample).
        split_idx = int(len(df_day) * 0.8)
        train_df = df_day.iloc[:split_idx]
        
        logger.info(f"Data Loaded. Train Size: {len(train_df)}")
        
        # B. SCALERS & TENSORS
        # Fit separate scalers for inputs (X), physics target (CSI), and raw target (GHI).
        # Scalers are saved to disk to ensure the inference pipeline uses identical transformations.
        cols_X = ['AirTemp', 'WindSpeed', 'Zenith', 'CloudOpacity']
        scaler_X = MinMaxScaler().fit(train_df[cols_X])
        scaler_y_phys = MinMaxScaler().fit(train_df[['CSI']])
        scaler_y_base = MinMaxScaler().fit(train_df[['GHI']])
        
        with open(f"scalers/{dataset_name}_scalers.pkl", "wb") as f:
            pickle.dump({'X': scaler_X, 'y_phys': scaler_y_phys, 'y_base': scaler_y_base}, f)

        # Convert data to PyTorch tensors and move to the appropriate device (CPU/GPU).
        train_x = torch.tensor(scaler_X.transform(train_df[cols_X])).to(device)
        train_y_csi = torch.tensor(scaler_y_phys.transform(train_df[['CSI']])).view(-1).to(device)
        train_y_ghi = torch.tensor(scaler_y_base.transform(train_df[['GHI']])).view(-1).to(device)
    
    except Exception as e:
        logger.critical(f"Critical Data Error: {str(e)}")
        return

    # C. GP TRAINER (With Skip Logic)
    def train_gp(model_cls, y_target, name):
        """
        Generic training loop for Gaussian Process models.
        """
        save_path = f"saved_models/{dataset_name}_{name}.pth"
        
        # SAFETY CHECK 1: Idempotency
        # If the model file already exists, skip training to save time and compute resources.
        if os.path.exists(save_path):
            logger.info(f"Skipping {name} (Already exists)")
            return

        logger.info(f"Training GP: {name}...")
        try:
            # Initialize likelihood and model
            like = gpytorch.likelihoods.GaussianLikelihood().to(device)
            model = model_cls(train_x, y_target, like).to(device)
            
            # Switch to training mode
            model.train()
            like.train()
            
            # Use Adam optimizer for hyperparameter tuning (lengthscales, noise, etc.)
            opt = torch.optim.Adam(model.parameters(), lr=0.1)
            # Marginal Log Likelihood (MLL) is the loss function for GPs
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(like, model)
            
            # Training Loop (100 iterations is typically sufficient for hyperparameter convergence)
            for i in range(100):
                opt.zero_grad()
                output = model(train_x)
                loss = -mll(output, y_target)
                loss.backward()
                opt.step()
                if i % 25 == 0: logger.info(f"   Step {i}/100 - Loss: {loss.item():.3f}")
                
            # Save both model state and likelihood state (crucial for noise parameters)
            torch.save({'model': model.state_dict(), 'like': like.state_dict()}, save_path)
            logger.info(f"Saved {name}")
        except Exception as e:
            logger.error(f"Failed to train {name}: {str(e)}")

    # D. LSTM TRAINER (With Skip Logic)
    def train_lstm(y_target, name):
        """
        Generic training loop for LSTM models.
        """
        save_path = f"saved_models/{dataset_name}_{name}.pth"
        if os.path.exists(save_path):
            logger.info(f"Skipping {name} (Already exists)")
            return

        logger.info(f"Training LSTM: {name}...")
        try:
            # Initialize model in double precision
            model = ReferenceLSTM(train_x.shape[1]).double().to(device)
            opt = torch.optim.Adam(model.parameters(), lr=0.001)
            crit = nn.MSELoss()
            
            # Create DataLoader for batch processing
            # Input needs to be reshaped to (batch, seq_len, features) -> (N, 1, 4)
            loader = DataLoader(TensorDataset(train_x.unsqueeze(1), y_target.view(-1, 1)), batch_size=32, shuffle=True)
            
            model.train()
            for epoch in range(50):
                epoch_loss = 0
                for bx, by in loader:
                    opt.zero_grad()
                    output = model(bx)
                    loss = crit(output, by)
                    loss.backward()
                    opt.step()
                    epoch_loss += loss.item()
                if epoch % 10 == 0: logger.info(f"   Epoch {epoch}/50 - Loss: {epoch_loss/len(loader):.4f}")
            
            torch.save(model.state_dict(), save_path)
            logger.info(f"Saved {name}")
        except Exception as e:
            logger.error(f"Failed to train {name}: {str(e)}")

    # EXECUTE TRAINERS
    # 1. Baseline GP: Trained on Raw GHI (Standard approach)
    train_gp(StandardGP, train_y_ghi, "baseline_gp_raw")
    # 2. Physics GPs: Trained on CSI (Physics-informed approach)
    train_gp(MaternGP, train_y_csi, "physics_gp_matern") # Matern kernel variant
    train_gp(StandardGP, train_y_csi, "physics_gp_rbf")    # RBF kernel variant
    # 3. LSTMs: Deep Learning benchmarks
    train_lstm(train_y_ghi, "lstm_raw")
    train_lstm(train_y_csi, "lstm_csi")

# --- EXECUTE ---
# Run the pipeline for both the replication dataset (60 min) and the high-frequency novelty dataset (5 min).
# 1. Anchor (60 min)
run_training_cycle("60mins.csv", "60min")
# 2. Novelty (5 min)
run_training_cycle("5mins.csv", "5min")

logger.info("PIPELINE COMPLETE.")