import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from src.data_cleaning import DataFrameCleaner
from src.extra_features import Income_Municipality
from src.filtering import Postal_Filtering, One_Hot
from src.outliers import ZScoreFilter
from src.evaluation import ModelEvaluation
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Setting up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Function to load data
def load_data(dataset_path, income_mun_path):
    try:
        df = pd.read_csv(dataset_path)
        df_income = pd.read_csv(income_mun_path)
        logging.info(
            f"Data loaded successfully from {dataset_path} and {income_mun_path}"
        )
        return df, df_income
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise


# Function for data cleaning and preprocessing
def preprocess_data(df, df_income):
    # Apply all the preprocessing steps
    extra_feature = Income_Municipality(df, df_income)
    df = extra_feature.add_feature()

    postal_filter = Postal_Filtering(df, 20)
    df = postal_filter.postal_filtering()

    one_hot = One_Hot(df, "province")
    df = one_hot.one_hot_encoder()

    cleaner = DataFrameCleaner(df)
    df = cleaner.clean()

    zscore_filter = ZScoreFilter(df, columns=["price", "livingarea"], threshold=3)
    df = zscore_filter.filter()

    logging.info("Data preprocessing completed")
    return df


# Function to split data into training and testing sets
def split_data(df):
    X = df.drop(columns=["price"])
    y = df["price"]
    return train_test_split(X, y, test_size=0.2, random_state=42)


# Function to train and evaluate a model
def train_and_evaluate_model(X_train, X_test, y_train, y_test, model, model_name):
    # Train the model
    model.fit(X_train, y_train)

    # Evaluate the model performance
    evaluator = ModelEvaluation(model, X_train, X_test, y_train, y_test)
    evaluator.print_metrics()

    return model


# Function to visualize feature importances
def plot_feature_importances(model, X_train):
    feature_importances = model.feature_importances_
    important_features = sorted(
        zip(X_train.columns, feature_importances), key=lambda x: x[1], reverse=True
    )

    # Visualizing Feature Importances
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=[importance for _, importance in important_features],
        y=[feature for feature, _ in important_features],
    )
    plt.title("Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.show()


# Main function to execute the workflow
def main():
    # File paths (could be passed as arguments or loaded from a config file)
    dataset_path = "./data/raw/dataset_province_municipality_code_large4.csv"
    income_mun_path = "./data/raw/income_municipality.csv"

    # Step 1: Load data
    df, df_income = load_data(dataset_path, income_mun_path)

    # Step 2: Preprocess data
    df = preprocess_data(df, df_income)

    # Step 3: Split data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(df)

    # Step 4: Initialize models with the best parameters
    best_params = {
        "bootstrap": False,
        "max_depth": 30,
        "max_features": "sqrt",
        "min_samples_leaf": 1,
        "min_samples_split": 2,
        "n_estimators": 300,
    }
    model_randomforest = RandomForestRegressor(**best_params, random_state=42)
    model_decisiontree = DecisionTreeRegressor(random_state=42)

    # Step 5: Train and evaluate RandomForest model
    logging.info("Training RandomForest model...")
    model_randomforest = train_and_evaluate_model(
        X_train, X_test, y_train, y_test, model_randomforest, "RandomForest"
    )

    # Step 6: Train and evaluate DecisionTree model
    logging.info("Training DecisionTree model...")
    model_decisiontree = train_and_evaluate_model(
        X_train, X_test, y_train, y_test, model_decisiontree, "DecisionTree"
    )

    # Step 7: Feature importance visualization
    logging.info("Visualizing feature importances...")

    plot_feature_importances(model_randomforest, X_train)


if __name__ == "__main__":
    main()
