It seems like you're facing an issue with the formatting in the markdown. I'll correct the indentation and ensure that the bash commands for installation are clear and properly formatted.

Here's the corrected version:

```markdown
# Real Estate Price Prediction Workflow

This project is a machine learning pipeline for predicting real estate prices using a variety of preprocessing steps, models, and visualizations. The implementation emphasizes modularity and readability, employing object-oriented programming principles.

---

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Pipeline Components](#pipeline-components)
    - [Data Loading](#data-loading)
    - [Data Preprocessing](#data-preprocessing)
    - [Model Training and Evaluation](#model-training-and-evaluation)
    - [Feature Importance Visualization](#feature-importance-visualization)
4. [Usage](#usage)
5. [Dependencies](#dependencies)
6. [Directory Structure](#directory-structure)
7. [Logging](#logging)
8. [Future Improvements](#future-improvements)

---

## Overview
This workflow processes raw real estate data, cleans and preprocesses it, trains machine learning models (Random Forest and Decision Tree regressors), evaluates their performance, and visualizes feature importances. The workflow is designed to be modular, allowing for easy integration of additional steps or models.
``` 
---

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. Create a virtual environment using `venv`:
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:
   - On **Windows**:
     ```bash
     venv\Scripts\activate
     ```
   - On **macOS/Linux**:
     ```bash
     source venv/bin/activate
     ```

4. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

5. Verify the installation by listing the installed packages:
   ```bash
   pip list
   ```

6. When you're done working, deactivate the virtual environment:
   ```bash
   deactivate
   ```

This setup ensures a clean, isolated environment for your project dependencies.

---

## Pipeline Components

### Data Loading
The `DataLoader` class loads raw datasets:
- **Inputs**:
  - `dataset_path`: Path to the real estate dataset.
  - `income_mun_path`: Path to the supplementary dataset with income data per municipality.
- **Outputs**:
  - `df`: Main dataset.
  - `df_income`: Income data.

```python
loader = DataLoader(dataset_path, income_mun_path)
df, df_income = loader.load()
```

### Data Preprocessing
The `DataPreProcessor` class cleans and enriches the dataset with several steps:
1. Adds additional features using average income per municipality data.
2. Filters by postal codes.
3. Applies one-hot encoding for categorical features.
4. Cleans the dataset using custom rules.
5. Removes outliers based on Z-scores.

```python
preprocessor = DataPreProcessor(df, df_income)
df = preprocessor.preprocess()
```

### Model Training and Evaluation
The `ModelTrainer` class handles:
1. Training models.
2. Evaluating performance using metrics like MAE, RMSE, and R².

Models included:
- **RandomForestRegressor**
- **DecisionTreeRegressor**

```python
trainer = ModelTrainer(X_train, X_test, y_train, y_test)
model_randomforest = trainer.train_and_evaluate(model_randomforest, "RandomForest")
model_decisiontree = trainer.train_and_evaluate(model_decisiontree, "DecisionTree")
```

### Feature Importance Visualization
The `FeatureImportanceVisualizer` class displays the importance of features for the trained Random Forest model.

```python
visualizer = FeatureImportanceVisualizer(model_randomforest, X_train.columns)
visualizer.plot()
```

---

## Usage
1. Set the paths to your datasets in the `MainWorkflow`:
   ```python
   dataset_path = "./data/raw/dataset_province_municipality_code_large.csv"
   income_mun_path = "./data/raw/income_municipality.csv"
   ```

2. Run the script:
   ```bash
   python main_oop.py
   ```

---

## Dependencies
- Python 3.8+
- Required libraries:
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn
  - logging

Install dependencies with:
```bash
pip install -r requirements.txt
```

---

## Directory Structure
```
project/
│
├── data/
│   ├── raw/
│   │   ├── dataset_province_municipality_code_large.csv
│   │   └── income_municipality.csv
│   └── processed/
│
├── src/
│   ├── data_cleaning.py
│   ├── extra_features.py
│   ├── filtering.py
│   ├── outliers.py
│   └── evaluation.py
│
├── main_oop.py
├── requirements.txt
└── README.md
```

---

## Logging
Logging is used throughout the pipeline to track progress and errors:
- INFO logs indicate the successful completion of each step.
- ERROR logs capture data loading or processing issues.

Example:
```
2024-12-06 10:00:00 - INFO - Data preprocessing completed
2024-12-06 10:05:00 - ERROR - Error loading data: File not found
```

---

## Future Improvements
1. **Hyperparameter Tuning**:
   Integrate tools like GridSearchCV for optimized model parameters everytime the model is launched.
2. **Model Comparison**:
   Add more regressors for comparison.
3. **Pipeline Automation**:
   Implement a configuration file for setting paths and parameters.
4. **Visualization Enhancements**:
   Add more detailed plots for analysis.

---

Feel free to contribute or report issues! 😊
```
