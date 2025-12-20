\# Physics-Informed Gaussian Processes for Solar Irradiance Forecasting

\#\# Overview  
This project implements a \*\*Physics-Informed Gaussian Process (GP)\*\* for probabilistic solar irradiance forecasting. By modeling the \*\*Clear Sky Index ($k^\*$)\*\* rather than raw irradiance, we separate deterministic solar geometry from stochastic atmospheric variance. This approach enforces physical constraints and significantly improves uncertainty quantification compared to standard black-box models.

The repository compares the following architectures:  
1\.  \*\*Smart Persistence:\*\* A baseline assuming the clear sky index remains constant.  
2\.  \*\*Standard LSTM:\*\* A recurrent neural network predicting raw GHI.  
3\.  \*\*Standard GP:\*\* A Gaussian Process predicting raw GHI.  
4\.  \*\*Physics-Informed GP (Ours):\*\* A GP modeling $k^\*$ with custom kernels (RBF & Matern).

\#\# Project Structure  
\`\`\`text  
├── 5mins.csv               \# High-resolution solar dataset (Target)  
├── 60mins.csv              \# Low-resolution solar dataset (Benchmark)  
├── train\_pipeline.py       \# Main script: Trains GPs and LSTMs  
├── save\_predictions.py     \# Inference script: Generates predictions & CIs  
├── term\_project\_plots.ipynb.py     \# Plotting script: Generate plots  
├── requirements.txt        \# Python dependencies  
├── README.md               \# Project documentation  
│  
├── saved\_models/           \# Stores trained model artifacts (.pth)  
├── scalers/                \# Stores scikit-learn scalers (.pkl)  
└── results/                \# Stores final output CSVs

## **Installation**

1. **Clone the repository** (or unzip the project folder).  
2. **Install dependencies:**  
   Bash

```
pip install -r requirements.txt
```

3.   
   *Note: A GPU (CUDA) is recommended for GP training, but the code automatically falls back to CPU if unavailable.*

## **Usage Workflow**

### **Step 1: Train Models**

Executes the training pipeline for the Baseline GP, LSTMs, and Physics-Informed GPs on both datasets.

Bash

```
python train_pipeline.py
```

*   
  **Output:** Saves trained model weights to saved\_models/ and data scalers to scalers/.  
* Notee that the code will skip training if it finds the models are already saved. So in order to run the training again make sure you have deleted the saved\_model/ and scalers/ folders.

### **Step 2: Generate Predictions**

Runs inference on the held-out test set. This script:

1. Reconstructs the Exact GP training state.  
2. Calculates 95% Confidence Intervals (Uncertainty).  
3. Inversely transforms predictions back to physical units ($W/m^2$).

Bash

```
python save_predictions.py
```

*   
  **Output:** Generates results/5min\_all\_predictions.csv and results/60min\_all\_predictions.csv.  
* **Next Step:** These CSV files are used by the plotting scripts to generate the final analysis figures.

### **Step 3: Generate Plots and tables**

Runs the following notebook to get the desired plots and tables.

term\_project\_plots.ipynb

## **Methodology Note**

The core innovation is the transformation:

$$ GHI\_{pred} \= GP(X\_{atm}) \\times GHI\_{clearsky} $$

Where $GP(X\_{atm})$ predicts the Clear Sky Index (0 to 1). This ensures physical constraints (non-negativity) and injects domain knowledge about the diurnal cycle into the probabilistic model.

