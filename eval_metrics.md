
# Evaluation Metrics Report

## Model Overview

This report provides a comprehensive evaluation of the models developed as part of this machine learning exercise. The key models assessed include **Random Forest Regressor** and **Decision Tree Regressor**. Their performance metrics are computed on both training and test datasets to evaluate their predictive accuracy, generalizability, and overfitting tendencies.

---

## Model Details & Instantiation

Below are the instantiated models with their hyperparameter settings:

```python
# Random Forest Regressor
model_randomforest = RandomForestRegressor(
    n_estimators=100,          # Number of decision trees in the ensemble.
    max_depth=None,            # Allow trees to grow until they are perfectly fit.
    random_state=42,           # Ensures reproducibility.
    n_jobs=-1                  # Utilize all available processors for parallel training.
)

# Decision Tree Regressor
model_decisiontree = DecisionTreeRegressor(
    max_depth=None,            # Allow the tree to grow until it fits the data perfectly.
    random_state=42           # Ensures reproducibility.
)
```

---

## Evaluation Metrics

The following key performance metrics were computed on both training and test datasets:

- **MAE (Mean Absolute Error)**: Measures the average magnitude of prediction errors.
- **RMSE (Root Mean Squared Error)**: Penalizes larger errors more heavily than MAE.
- **R² (R-squared)**: Indicates the proportion of variance in the dependent variable explained by the model.

### Random Forest Metrics

#### Training
- **MAE**: ~27,030.55  
- **RMSE**: ~47,463.45  
- **R²**: 0.98  

#### Test
- **MAE**: ~67,508.12  
- **RMSE**: ~110,693.10  
- **R²**: 0.87  

---

### Decision Tree Metrics

#### Training
- **MAE**: ~4,047.36  
- **RMSE**: ~26,677.42  
- **R²**: 0.99  

#### Test
- **MAE**: ~66,726.84  
- **RMSE**: ~139,522.98  
- **R²**: 0.79  

---

## Features Used & Feature Engineering

To ensure that data leakage was minimized and feature quality was appropriate, careful feature engineering was implemented:

### Features

1. Columns used from the dataset: 
        bedrooms', 'kitchen', 'facades', 'price', 'terrace', 'gardensurface',
       'livingarea', 'surfaceoftheplot', 'as_new', 'good', 'just_renovated',
       'to_be_done_up', 'to_renovate', 'to_restore', 'is_apartment',
       'is_house', 'Average_Income_Per_Citizen', 'province_Antwerpen',
       'province_Brussel', 'province_Henegouwen', 'province_Limburg',
       'province_Luik', 'province_Luxemburg', 'province_Namen',
       'province_Oost-Vlaanderen', 'province_Vlaams-Brabant',
       'province_Waals-Brabant', 'province_West-Vlaanderen'

2. **Added Feature**:
   - **Average_Income_Per_Citizen**: An aggregated economic indicator derived from external data merged with the training data. Downloaded from [StatBel](https://statbel.fgov.be/fr/themes/datalab/revenu-disponible-administratif#:~:text=Le%20revenu%20m%C3%A9dian%20par%20commune,%2DL%C3%A9ger%20(%E2%82%AC%2034.924).)


---

### Feature Engineering Steps

1. Dropped irrelevant features or columns potentially contributing noise.
2. Z-score normalization was applied post-cleaning to standardize features.

---

## Accuracy Computation

### Split Percentage
80/20 (training/test split).

### Validation
Validation relied on a simple train-test split and cross validation.
All metrics were computed on unseen test splits.

---

## Model Efficiency

Both models are runned at the same time in less than 1 minute

These efficiency metrics are crucial for evaluating model feasibility in production deployment.

---

## Dataset Overview

Below is a summary of data preparation steps and dataset insights:

1. **Initial Records**: 39,264 entries with 43 columns. 3 extra rescraped have been done to add consistency to the dataset.
2. **Filtered Records**: 37,783 (rows filtered post initial feature removal).
3. **Data Cleaning**:
   - Z-score filtering reduced the number of records to **35,401** after outlier handling.
4. **Preprocessing Steps**:
   - Duplicates removed.
   - Missing values handled using row-based filtering with `dropna`.
   - Redundant columns dropped to streamline feature engineering efforts.

---

## Conclusions

1. The **Random Forest Regressor** achieved better generalization (**R² = 0.87** on test set) with less overfitting compared to the **Decision Tree model**.
2. The **Decision Tree Regressor** exhibited signs of overfitting with **R² = 0.99** on the training set but only **R² = 0.79** on unseen test data.
3. Random Forest's ensemble mechanism provides robustness against variance issues compared to Decision Trees' tendency to overfit.

---

## Summary

This report serves as a blueprint to evaluate current model performance and provides insights into the steps taken during preprocessing, feature engineering, model training, and evaluation.
```
