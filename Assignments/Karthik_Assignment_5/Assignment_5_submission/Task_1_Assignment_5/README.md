# **Task 1: Credit Card Fraud Detection System**

This project implements a deployment-ready fraud detection pipeline using a weighted ensemble of **CatBoost**, **LightGBM**, and **XGBoost**. The system uses a standalone prediction script to process synthetic credit card transaction data and identify fraudulent activities with high recall.

## **System Overview**

The core of the deployment is `task1_predict.py`, which:

1. **Loads Pre-trained Models:** Imports the frozen ensemble models and feature mapping artifacts.  
2. **Performs Feature Engineering:** Replicates the complex geospatial and temporal feature extraction used during training (e.g., Haversine distance, rolling window aggregations).  
3. **Generates Predictions:** Outputs class labels (0/1) and fraud probabilities.

## **Prerequisites**

* Python 3.8+  
* Required Python libraries (see `requirements.txt`)

## **Installation**

1. Ensure you have Python installed.  
2. Install the required dependencies:

```
pip install -r requirements.txt
```

## **Required Artifacts**

For the prediction script to function, the following pre-trained artifact files must be present in the same directory as `task1_predict.py`. These represent the "frozen" state of the system:

* **Model Files:**  
  * `final_cat.cbm` (CatBoost Model)  
  * `final_lgb.txt` (LightGBM Model)  
  * `final_xgb.json` (XGBoost Model)  
* **Encoders & Mappings:**  
  * `fraud_rate_map.pkl` (Merchant risk mapping)  
  * `global_mean_fraud.pkl` (Fallback statistics)  
  * `label_encoders.pkl` (Categorical encoders)  
* **Ensemble Configuration:**  
  * `w_cat.pkl`, `w_lgb.pkl`, `w_xgb.pkl` (Model weights)  
  * `ensemble_best_threshold.pkl` (Optimized decision threshold)

## **Usage**

To evaluate the model on a new dataset, run the script from the command line.

### **Command Syntax**

```
python task1_predict.py --test_file <path_to_csv>
```

### **Example**

```
python task1_predict.py --test_file test_set_realistic_1pct.csv
```

### **Input Data Format**

The input CSV must follow the Sparkov synthetic dataset schema, including columns such as:

* **IDs:** `cc_num`, `merchant`  
* **Time:** `trans_date`, `trans_time`  
* **Details:** `amt`, `category`, `profile`  
* **Location:** `lat`, `long`, `merch_lat`, `merch_long`  
* *(Optional)* `is_fraud`: If present, the script will calculate and print evaluation metrics.

### **Output**

1. **Submission File :**  
   * The script **always** generates a file named `task1_predictions.csv` in the current directory.  
   * This file mirrors the **exact format** of the input CSV (retaining all original columns).  
   * The `is_fraud` column is overwritten (or added) containing the model's **predictions** (0 or 1), replacing any ground truth labels.  
   * This is the file required for the assignment submission.  
2. **Console Metrics:**  
   * If the input file contains ground truth labels (`is_fraud`), the script will also print:  
     * **Confusion Matrix** (Counts of TP, TN, FP, FN).  
     * **Accuracy** and **Macro-averaged F1 Score**.  
     * **F1 Score for the Fraud Class** (Key Metric).

