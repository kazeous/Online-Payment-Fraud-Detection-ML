# Online Payment Fraud Detection (ML)

This repository contains an end-to-end machine learning workflow for detecting fraudulent online payment transactions using a Kaggle dataset.

## Project contents

- `Online Payment Fraud Detection` — the main notebook (JSON/Jupyter format) containing:
  - data loading and preprocessing
  - exploratory data analysis (EDA)
  - feature preparation
  - model training and evaluation

## Problem statement

The goal is to classify transactions as fraudulent (`isFraud = 1`) or non-fraudulent (`isFraud = 0`) from historical transaction data.

## Dataset

The notebook uses the Kaggle **Online Payment Fraud Detection** dataset (`onlinefraud.csv`) and analyzes features such as:

- `step`
- `type`
- `amount`
- `nameOrig`
- `oldbalanceOrg`
- `newbalanceOrig`
- `nameDest`
- `oldbalanceDest`
- `newbalanceDest`
- `isFraud`

## Workflow summary

1. **EDA**
   - checks for missing/duplicate values
   - studies transaction distributions and fraud imbalance
   - explores univariate, bivariate, and multivariate relationships

2. **Preprocessing**
   - encodes transaction type
   - scales numeric features
   - applies undersampling (`RandomUnderSampler`) to address class imbalance

3. **Modeling**
   - compares:
     - Random Forest Classifier
     - Logistic Regression
   - uses stratified 5-fold cross-validation with metrics:
     - accuracy
     - precision
     - recall
     - F1
     - ROC-AUC

4. **Evaluation**
   - confusion matrix
   - ROC curve and AUC

## Result

According to the notebook conclusion, **Random Forest** is the best-performing model, achieving an AUC close to **0.999**.

## Run locally

1. Ensure Python 3 is installed.
2. Install dependencies:

```bash
pip install numpy pandas matplotlib seaborn scipy tabulate tensorflow scikit-learn imbalanced-learn
```

3. Open the notebook file in Jupyter/Colab/Kaggle and update the dataset path if needed.

## Notes

- The notebook file is stored without a `.ipynb` extension.
- The original notebook references a Kaggle input path (`/kaggle/input/...`). You may need to adjust paths for local execution.
