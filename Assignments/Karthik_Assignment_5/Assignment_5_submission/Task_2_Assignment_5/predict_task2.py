import pandas as pd
import numpy as np
import pickle
import sys
import os
import argparse
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

# ==============================================================================
# 1. FEATURE ENGINEERING (Must match training exactly)
# ==============================================================================
def add_features(X_input):
    X_feat = X_input.copy()
    # Ensure string column names for XGBoost
    X_feat.columns = X_feat.columns.astype(str)
    
    # Interaction Features
    # Note: These column indices ('30', '66', etc.) refer to the original 
    # integer column indices from the headerless CSV, converted to strings.
    if '30' in X_feat.columns:
        X_feat['Walking_Gate_1'] = X_feat['30'] * X_feat['66']
        X_feat['Walking_Gate_2'] = X_feat['30'] * X_feat['31']
        X_feat['Down_Gate'] = X_feat['30'] * X_feat['123']
        X_feat['Stairs_Dir_Gate'] = X_feat['7'] * X_feat['132']
        X_feat['Sports_Gate'] = X_feat['161'] * X_feat['24']
        X_feat['Jog_Signal'] = X_feat['159'] * X_feat['30']
        # Add epsilon to avoid division by zero
        X_feat['Stairs_Ratio'] = X_feat['24'] / (X_feat['30'].abs() + 0.001)
        X_feat['Sitting_Gravity'] = X_feat['6'].abs()
    
    return X_feat

# ==============================================================================
# 2. MAIN PREDICTION LOOP
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('test_file', type=str, help="Path to test CSV")
    args = parser.parse_args()

    print(f"Processing {args.test_file}...")
    
    # 1. Load Data
    try:
        # HAR data has no header
        df = pd.read_csv(args.test_file, header=None)
    except FileNotFoundError:
        print(f"Error: File {args.test_file} not found.")
        sys.exit(1)

    # 2. Separate Features and Targets
    # Assumption: The last column is the label, just like in training.
    X_raw = df.iloc[:, :-1]
    y_true = df.iloc[:, -1]
    
    # 3. Apply Feature Engineering
    X_eng = add_features(X_raw)
    
    # 4. Load Model and Encoder
    try:
        with open('task2_model.pkl', 'rb') as f:
            pipeline = pickle.load(f)
        with open('task2_le.pkl', 'rb') as f:
            le = pickle.load(f) # We need this for the confusion matrix labels
    except FileNotFoundError:
        print("Error: Model files not found. Please run train_task2.py first.")
        sys.exit(1)

    # 5. Predict
    print("Predicting...")
    y_pred_encoded = pipeline.predict(X_eng)
    y_pred_labels = le.inverse_transform(y_pred_encoded)
    
    # 6. Save Output
    # Matches original format: Features + Prediction Column (No Header)
    output_df = X_raw.copy()
    output_df['prediction'] = y_pred_labels
    
    output_filename = "task2_predictions.csv"
    output_df.to_csv(output_filename, header=False, index=False)
    print(f"Saved: '{output_filename}'")
    
    # 7. Evaluation (Console Output)
    # This block mimics task1_predict.py
    if y_true is not None:
        print("\n" + "="*40)
        print("RESULTS - T2: Human Activity Recognition")
        print("="*40)
        
        # Calculate Metrics
        acc = accuracy_score(y_true, y_pred_labels)
        f1_macro = f1_score(y_true, y_pred_labels, average='macro')
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred_labels, labels=le.classes_)
        
        # Create a readable DataFrame for the Confusion Matrix
        cm_df = pd.DataFrame(cm, 
                             index=[f"Actual: {c}" for c in le.classes_], 
                             columns=[f"Pred: {c}" for c in le.classes_])
        
        print("\n[B] Confusion Matrix:")
        # We allow pandas to print the full matrix without truncation if possible
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
            print(cm_df)
        
        print("\n[C] Metrics:")
        print(f"Accuracy:       {acc:.4f}")
        print(f"Macro-avg F1:   {f1_macro:.4f}")
        print("="*40)

if __name__ == "__main__":
    main()