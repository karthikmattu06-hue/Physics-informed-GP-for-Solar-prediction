import pandas as pd
import numpy as np
import argparse
import sys
import pickle
import warnings
import os
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

# Suppress warnings
warnings.filterwarnings("ignore")

# Attempt imports for the ensemble models
try:
    from catboost import CatBoostClassifier
    import lightgbm as lgb
    import xgboost as xgb
except ImportError as e:
    print(f"Error: Missing library. {e}")
    sys.exit(1)

# ==============================================================================
# 1. HELPER FUNCTIONS (MATH & UTILS)
# ==============================================================================

def haversine(lat1, lon1, lat2, lon2):
    """Computes the great-circle distance in kilometers."""
    R = 6371  # Earth radius (km)
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2)
    return 2 * R * np.arcsin(np.sqrt(a))

def safe_label_transform(le, col_data):
    """
    Safely transforms labels. Unseen labels are mapped to a specific 'unknown' class 
    if possible, or the mode/first class of the encoder to prevent crashes.
    """
    known_classes = set(le.classes_)
    # Map unknown to the most frequent (or first) class
    fill_value = le.classes_[0] 
    return col_data.map(lambda x: x if x in known_classes else fill_value).apply(lambda x: le.transform([x])[0])

# ==============================================================================
# 2. FEATURE ENGINEERING PIPELINE
# ==============================================================================

def feature_engineering_batch(df):
    """
    Replicates the exact feature engineering steps from the training notebook.
    """
    print("... performing complex feature engineering ...")
    data = df.copy()

    # --- 2.1 Date/Time ---
    if 'trans_date' in data.columns and 'trans_time' in data.columns:
        data["trans_datetime"] = pd.to_datetime(
            data["trans_date"].astype(str) + " " + data["trans_time"].astype(str)
        )

    # Sort for rolling features (Critical)
    data = data.sort_values(["cc_num", "trans_datetime"]).reset_index(drop=True)

    # Time Diffs
    data["prev_ts"] = data.groupby("cc_num")["trans_datetime"].shift()
    data["time_diff"] = (data["trans_datetime"] - data["prev_ts"]).dt.total_seconds()
    data["time_diff"] = data["time_diff"].fillna(data["time_diff"].median())

    data["hour"] = data["trans_datetime"].dt.hour
    data["day"] = data["trans_datetime"].dt.day
    data["weekday"] = data["trans_datetime"].dt.weekday

    # --- 2.2 Demographics ---
    if "profile" in data.columns:
        demo = data["profile"].str.split("_", expand=True)
        if demo.shape[1] == 4:
            demo.columns = ["life_stage", "age_band", "profile_gender", "area_type"]
            data = pd.concat([data, demo], axis=1)

    # --- 2.3 Merchant Features ---
    data["merchant_freq"] = data.groupby("merchant")["merchant"].transform("count")
    data["merchant_avg_amt"] = data.groupby("merchant")["amt"].transform("mean")
    data["merchant_std_amt"] = data.groupby("merchant")["amt"].transform("std").fillna(0)
    data["merchant_user_freq"] = data.groupby(["cc_num", "merchant"])["amt"].transform("count")

    # --- 2.4 User Global Stats ---
    grouped_user = data.groupby("cc_num")
    data["user_avg_amt"] = grouped_user["amt"].transform("mean")
    data["user_std_amt"] = grouped_user["amt"].transform("std").fillna(0)
    data["user_max_amt"] = grouped_user["amt"].transform("max")
    data["user_min_amt"] = grouped_user["amt"].transform("min")
    data["user_trans_ct"] = grouped_user["amt"].transform("count")

    # --- 3. Haversine ---
    data["distance_km"] = haversine(
        data["lat"], data["long"],
        data["merch_lat"], data["merch_long"]
    )

    # --- 4. Sparkov Behavioral Features ---
    data["user_home_lat"] = grouped_user["lat"].transform(lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0])
    data["user_home_long"] = grouped_user["long"].transform(lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0])
    
    data["dist_from_home"] = np.sqrt(
        (data["lat"] - data["user_home_lat"])**2 +
        (data["long"] - data["user_home_long"])**2
    )

    data["user_category_freq"] = data.groupby(["cc_num", "category"])["category"].transform("count")
    data["category_rarity"] = 1 / (data["user_category_freq"] + 1)

    data["is_first_time_merchant"] = (
        data.groupby(["cc_num", "merchant"]).cumcount().eq(0).astype(int)
    )

    data["user_mean_hour"] = grouped_user["hour"].transform("mean")
    data["hour_deviation"] = (data["hour"] - data["user_mean_hour"]).abs()

    data["amt_to_user_avg_ratio"] = data["amt"] / (data["user_avg_amt"] + 1e-6)
    data["amt_to_user_std_ratio"] = data["amt"] / (data["user_std_amt"] + 1e-6)
    data["is_high_velocity"] = (data["amt_to_user_avg_ratio"] > 2.5).astype(int)

    # --- 5. Advanced Dynamic Features ---
    data["rolling_amt_mean_3"] = grouped_user["amt"].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    data["rolling_amt_std_3"] = grouped_user["amt"].rolling(3, min_periods=1).std().fillna(0).reset_index(level=0, drop=True)
    data["rolling_amt_max_3"] = grouped_user["amt"].rolling(3, min_periods=1).max().reset_index(level=0, drop=True)

    data["rolling_time_diff_mean_3"] = grouped_user["time_diff"].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    data["rolling_time_diff_std_3"] = grouped_user["time_diff"].rolling(3, min_periods=1).std().fillna(0).reset_index(level=0, drop=True)

    data["user_merchant_ct"] = data.groupby(["cc_num", "merchant"]).cumcount()

    data["user_merchant_avg_amt"] = (
        data.groupby(["cc_num", "merchant"])["amt"]
        .expanding().mean().reset_index(level=[0,1], drop=True)
    )

    data["user_merchant_last_ts"] = data.groupby(["cc_num", "merchant"])["trans_datetime"].shift()
    data["user_merchant_time_since_last"] = (data["trans_datetime"] - data["user_merchant_last_ts"]).dt.total_seconds()
    data["user_merchant_time_since_last"] = data["user_merchant_time_since_last"].fillna(data["user_merchant_time_since_last"].median())

    data["dev_from_user_avg"] = data["amt"] - data["user_avg_amt"]
    data["amt_user_avg_ratio"] = data["amt"] / (data["user_avg_amt"] + 1e-6)
    data["is_spike_3x"] = (data["amt"] > 3 * data["user_avg_amt"]).astype(int)
    data["is_spike_5x"] = (data["amt"] > 5 * data["user_avg_amt"]).astype(int)

    data["cumulative_amt"] = grouped_user["amt"].cumsum()
    data["cumulative_avg_amt"] = data["cumulative_amt"] / (data["user_trans_ct"] + 1e-6)
    data["amt_cumavg_ratio"] = data["amt"] / (data["cumulative_avg_amt"] + 1e-6)

    data["user_category_count"] = data.groupby(["cc_num", "category"])["category"].transform("count")
    data["category_rarity_for_user"] = 1 / (data["user_category_count"] + 1)

    data["is_night"] = ((data["hour"] <= 5) | (data["hour"] >= 23)).astype(int)
    data["night_to_day_jump"] = (
        (data["is_night"].shift(1) == 1) & (data["is_night"] == 0)
    ).astype(int).fillna(0)

    # --- 6. Cleanup (Drop Columns) ---
    drop_cols = [
        "ssn", "first", "last", "street", "city", "state", "zip",
        "job", "dob", "acct_num", "trans_num",
        "trans_date", "trans_time", "unix_time",
        "merchant", "profile", "user_merchant_last_ts", "prev_ts",
        "trans_datetime"
    ]
    
    y = None
    if 'is_fraud' in data.columns:
        y = data['is_fraud']
        X = data.drop(columns=drop_cols + ['is_fraud'], errors='ignore')
    else:
        X = data.drop(columns=drop_cols, errors='ignore')

    return X, y

# ==============================================================================
# 3. APPLY ARTIFACTS
# ==============================================================================

def apply_artifacts(X, artifacts):
    """Applies the global merchant encoding map and label encoders."""
    print("... applying artifacts (merchant map & label encoders) ...")
    X_proc = X.copy()

    # A. Merchant Fraud Rate
    if 'fraud_rate_map' in artifacts:
        fraud_map = artifacts['fraud_rate_map']
        global_mean = artifacts['global_mean_fraud']
        
        def get_rate(freq):
            for interval, rate in fraud_map.items():
                if freq in interval:
                    return rate
            return global_mean

        X_proc['merchant_fraud_rate'] = X_proc['merchant_freq'].apply(get_rate)
    
    # B. Create Model-Specific Copies
    X_cat = X_proc.copy()
    X_lgb = X_proc.copy()
    X_xgb = X_proc.copy()
    
    if 'label_encoders' in artifacts:
        les = artifacts['label_encoders']
        for col, le in les.items():
            if col in X_lgb.columns:
                X_lgb[col] = X_lgb[col].astype(str)
                X_xgb[col] = X_xgb[col].astype(str)
                X_lgb[col] = safe_label_transform(le, X_lgb[col])
                X_xgb[col] = safe_label_transform(le, X_xgb[col])

    return X_cat, X_lgb, X_xgb

# ==============================================================================
# 4. MAIN PREDICTION LOOP
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file', type=str, required=True, help="Path to test CSV")
    args = parser.parse_args()

    # 1. Load All Artifacts
    print("Loading artifacts...")
    required_files = [
        "final_cat.cbm", "final_lgb.txt", "final_xgb.json",
        "fraud_rate_map.pkl", "global_mean_fraud.pkl",
        "w_cat.pkl", "w_lgb.pkl", "w_xgb.pkl",
        "ensemble_best_threshold.pkl", "label_encoders.pkl"
    ]
    
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        print(f"Error: Missing artifact files: {missing}")
        print("Please run Section 15 of your notebook to generate them.")
        sys.exit(1)

    artifacts = {}
    
    # Models
    print("Loading models...")
    artifacts['cat_model'] = CatBoostClassifier()
    artifacts['cat_model'].load_model("final_cat.cbm")
    
    artifacts['lgb_model'] = lgb.Booster(model_file="final_lgb.txt")
    
    artifacts['xgb_model'] = xgb.XGBClassifier()
    artifacts['xgb_model'].load_model("final_xgb.json")

    # Pickles
    artifacts['fraud_rate_map'] = pd.read_pickle("fraud_rate_map.pkl")
    artifacts['global_mean_fraud'] = pd.read_pickle("global_mean_fraud.pkl")
    artifacts['w_cat'] = pd.read_pickle("w_cat.pkl")
    artifacts['w_lgb'] = pd.read_pickle("w_lgb.pkl")
    artifacts['w_xgb'] = pd.read_pickle("w_xgb.pkl")
    artifacts['threshold'] = pd.read_pickle("ensemble_best_threshold.pkl")
    
    with open("label_encoders.pkl", "rb") as f:
        artifacts['label_encoders'] = pickle.load(f)

    # 2. Load Data
    print(f"Loading test data: {args.test_file}")
    df_raw = pd.read_csv(args.test_file)
    
    # 3. Feature Engineering
    X, y_true = feature_engineering_batch(df_raw)
    
    # 4. Apply Artifacts (Encodings)
    X_cat, X_lgb, X_xgb = apply_artifacts(X, artifacts)

    # 5. Predict
    print("Running Ensemble Prediction...")
    p_cat = artifacts['cat_model'].predict_proba(X_cat)[:, 1]
    p_lgb = artifacts['lgb_model'].predict(X_lgb)
    p_xgb = artifacts['xgb_model'].predict_proba(X_xgb)[:, 1]
    
    p_final = (artifacts['w_cat'] * p_cat + artifacts['w_lgb'] * p_lgb + artifacts['w_xgb'] * p_xgb)

    threshold = artifacts['threshold']
    y_pred = (p_final >= threshold).astype(int)
    
    print(f"\nPrediction Complete using threshold: {threshold:.4f}")

    # --- 6. GENERATE SUBMISSION FILE (Requirement: Matches input format) ---
    print("\nGenerating submission file...")
    submission = df_raw.copy()
    
    # Overwrite (or create) the 'is_fraud' column with PREDICTIONS
    submission['is_fraud'] = y_pred
    
    output_filename = "task1_predictions.csv"
    submission.to_csv(output_filename, index=False)
    print(f"Saved: '{output_filename}'") 
    print(f"Details: {submission.shape} rows, includes original columns + predicted 'is_fraud'.")

    # 7. Evaluation (Console Output)
    if y_true is not None:
        print("\n" + "="*40)
        print("RESULTS - T1: Fraud Detection")
        print("="*40)
        
        cm = confusion_matrix(y_true, y_pred)
        
        cm_df = pd.DataFrame(cm, 
                             index=['Actual: No Fraud', 'Actual: Fraud'], 
                             columns=['Pred: No Fraud', 'Pred: Fraud'])
        
        acc = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_fraud = f1_score(y_true, y_pred, pos_label=1)

        print("\n[B] Confusion Matrix:")
        print(cm_df)
        
        print("\n[C] Metrics:")
        print(f"Accuracy:       {acc:.4f}")
        print(f"Macro-avg F1:   {f1_macro:.4f}")
        
        print("\n[D] Key Metric:")
        print(f"F1 (Fraud):     {f1_fraud:.4f}")
        print("="*40)

if __name__ == "__main__":
    main()