
# Real Estate Price Prediction Pipeline

This project for BeCode during the Data and Ai course implements a machine learning pipeline for predicting real estate prices based on given features. It uses preprocessing, model training (Random Forest and Decision Tree), model evaluation, and visualization tools to analyze performance.

---

## 🛠️ Overview

The pipeline performs the following steps:

1. **Data Loading**: Loads the main dataset and income municipality data.
2. **Data Preprocessing**: Cleans and preprocesses the data using outlier removal, encoding, and filtering.
3. **Model Training**: Trains two models, **Random Forest** and **Decision Tree**, for price prediction.
4. **Model Evaluation**: Evaluates the performance of both models.
5. **Visualization**: Displays insights from learning curves, feature importance, residual plots, predictions vs. actuals, and predicted price distributions.

---

## 📂 Directory Structure

The following directory structure is implemented:

```
project_root/
│
├── main.py                     # Main script that orchestrates the pipeline
│
├── src/
│   ├── data_cleaning.py       # Data cleaning logic
│   ├── extra_features.py      # Logic for adding additional features
│   ├── filtering.py           # Data filtering operations
│   ├── outliers.py            # Outlier detection logic
│   ├── evaluation.py          # Model evaluation logic
│   └── visualizer.py          # Visualization tools
│
├── data/
│   └── raw/
│       ├── dataset_province_municipality_code_large.csv
│       └── income_municipality.csv
│
├── requirements.txt           # List of dependencies
├── eval_metrics.md          # A full evaluation report of the model
└── README.md                  # This documentation file
```

---

## 📥 Installation Instructions

To run this project, follow the steps below:

### 1. Clone the Repository

Clone the project repository:

```bash
git clone <repository-url>
cd project_root
```

### 2. Set up a Virtual Environment

Set up a Python virtual environment to avoid conflicts:

```bash
python -m venv venv
source venv/bin/activate  # For Unix-based systems
venv\Scripts\activate  # For Windows
```

### 3. Install Dependencies

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## ⚙️ Requirements

This project depends on the following libraries:

- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `logging`

Add these dependencies in the `requirements.txt` file:

```
pandas
scikit-learn
matplotlib
seaborn
scipy
```

---

## 🏗️ How to Run

After installation, execute the main pipeline script:

```bash
python main.py
```

---

## 📊 Output & Visualizations

When you run the script, it will:

1. Train **Random Forest** and **Decision Tree** models on the dataset.
2. Log evaluation results.
3. Visualize:
   - Learning curves.
   - Feature importance plots.
   - Predictions vs. actual values plots.
   - Residual plots.
   - Predicted price distributions.

You will see visualizations and logs generated sequentially during execution.

---

## 🔄 Pipeline Workflow Steps

### 1. **Load Data**
The `DataLoader` class reads in the dataset and income municipality information.

### 2. **Preprocess Data**
The pipeline preprocesses the data using these steps:
- **Add Income Feature**: Combines income data with the main dataset.
- **Filter Based on Postal Codes**: Filters properties by postal codes.
- **Filter Based on Bedrooms**: Filters out unrealistic bedroom counts.
- **Apply One-Hot Encoding**: Encodes categorical variables like province names.
- **Remove Outliers**: Uses the Z-score method to eliminate extreme data points.

### 3. **Split Data**
Splits data into training and testing sets with an 80-20 split.

### 4. **Train Models**
The models used are:
- **Random Forest Regressor**
- **Decision Tree Regressor**

### 5. **Evaluate Models**
Both models are evaluated, and metrics are logged.

### 6. **Visualizations**
After training and evaluation, several visualizations are generated:
- Learning curves for both Random Forest and Decision Tree.
- Feature importance plots for both models.
- Comparison of actual prices vs. predictions.
- Residual plots.
- Predicted price distributions for a better understanding of variance.

---

## 🖹 Visualizations

The pipeline generates the following visualizations:

1. **Learning Curves**: Displays how well each model learns during training.
2. **Feature Importance**: Highlights which features were most influential in the Random Forest and Decision Tree models.
3. **Prediction vs. Actuals**: A scatter plot comparing predicted house prices against actual values.
4. **Residuals**: Displays the residuals of model predictions to identify bias or systematic errors.
5. **Predicted Price Distributions**: Displays the distribution of predicted house prices for both models.

---

## 📚 Dependencies (for developers)

You can use the following versions (ensure they're compatible with your Python version):

```
pandas>=2.0
scikit-learn>=1.2
matplotlib>=3.6
```

Add these in `requirements.txt`.

---

## 🛑 Troubleshooting

### Common errors:
1. **FileNotFoundError**:  
   Ensure that `dataset_province_municipality_code_large.csv` and `income_municipality.csv` exist in `./data/raw/`. Double-check the path.

2. **Model Training Errors**:  
   If the models fail to train, check the preprocessing logic or the cleaned data to ensure no invalid data points exist.

3. **Visualization Not Showing**:  
   Ensure you are running this in a compatible environment with GUI capabilities for visualization (e.g., Jupyter Notebook or local Python environment).

---

## 🎉 Acknowledgments

This project uses preprocessing, filtering, and visualization components from the following:
- Scikit-learn's machine learning models and evaluation tools.
- Pandas for data manipulation.
- Custom visualization tools (`src.visualizer`) to inspect data insights.

---

## 🧑‍💻 Author Information

Developed by **Kevin Jamotte**

If you have questions, feature requests, or bugs to report, please contact me through GitHub :)
