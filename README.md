# Credit Card Fraud Detection

Machine-learning capstone that detects fraudulent card transactions in a highly imbalanced dataset, compares seven model families, and weighs the business cost of errors.

## Problem

A fraudulent transaction is a direct loss to the bank and hurts customer trust. Fraud is rare, so a model can score high accuracy by predicting "genuine" every time. This project therefore judges models on ROC-AUC, precision and recall rather than accuracy.

## Data

[Credit Card Fraud Detection (Kaggle, ULB)](https://www.kaggle.com/mlg-ulb/creditcardfraud): 284,807 transactions, of which 492 are fraudulent (about 0.17%).

## Approach

1. Exploratory analysis and skew correction (PowerTransformer where distributions were skewed)
2. Stratified train/test split, with the test set held out for final evaluation
3. Models: Logistic Regression (L1 and L2), KNN, Decision Tree (Gini and entropy), Random Forest, XGBoost; SVM was tried but was too slow on the oversampled data
4. Cross-validation with RepeatedKFold and StratifiedKFold
5. Class balancing with random oversampling
6. Hyperparameter tuning (scikit-optimize in the notebook's imports) and threshold selection
7. Cost-benefit analysis in `Cost_Benefit_Analysis.xlsx`

## Results

- Best model on the original data: XGBoost with RepeatedKFold, ROC-AUC 0.984
- Selected model: XGBoost with random oversampling and StratifiedKFold, ROC-AUC 0.985, with a tuned decision threshold of about 0.005
- Logistic Regression with L1 regularisation reached ROC-AUC 0.985 in cross-validation
- On the held-out test set (96 fraud cases), the Logistic Regression runs reached fraud-class recall of 0.49 to 0.56 at the default threshold, which is why threshold tuning and cost-benefit analysis matter here

## Files

| File | Purpose |
| --- | --- |
| `Credit_card_fraud detection_Anand_Sukanya_Vijaya.ipynb` | Full analysis and modelling |
| `Cost_Benefit_Analysis.xlsx` | Business cost of false positives and false negatives |

## Run it

```bash
pip install -r requirements.txt
jupyter notebook "Credit_card_fraud detection_Anand_Sukanya_Vijaya.ipynb"
```

Download `creditcard.csv` from the Kaggle link above and place it beside the notebook.

## Team

Anand Pratap, Sukanya Sarkar, Vijaya
