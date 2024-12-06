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

class DataLoader:
    def __init__(self, dataset_path, income_mun_path):
        self.dataset_path = dataset_path
        self.income_mun_path = income_mun_path
        
    def load(self):
        """Loads datasets from the specified paths."""
        try:
            df = pd.read_csv(self.dataset_path)
            df_income = pd.read_csv(self.income_mun_path)
            logging.info(f"Data loaded successfully from {self.dataset_path} and {self.income_mun_path}")
            return df, df_income
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise

class DataPreProcessor:
    def __init__(self, df, df_income):
        self.df = df
        self.df_income = df_income
        
    def preprocess(self):
        """Applies all preprocessing steps."""
        logging.info("Starting preprocessing...")

        # Add extra feature
        extra_feature = Income_Municipality(self.df, self.df_income)
        self.df = extra_feature.add_feature()

        # Apply postal filtering
        postal_filter = Postal_Filtering(self.df, 20)
        self.df = postal_filter.postal_filtering()

        # One-hot encoding for 'province'
        one_hot = One_Hot(self.df, "province")
        self.df = one_hot.one_hot_encoder()

        # Data cleaning
        cleaner = DataFrameCleaner(self.df)
        self.df = cleaner.clean()

        # Remove outliers using Z-score
        zscore_filter = ZScoreFilter(self.df, columns=["price", "livingarea"], threshold=3)
        self.df = zscore_filter.filter()

        logging.info(f"Data preprocessing completed: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        return self.df

class ModelTrainer:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
    def train_and_evaluate(self, model, model_name):
        """Trains and evaluates the model."""
        logging.info(f"Training {model_name} model...")
        model.fit(self.X_train, self.y_train)

        # Evaluate model performance
        evaluator = ModelEvaluation(
            model=model,  # Ensure 'model' is passed first
            X_train=self.X_train,
            X_test=self.X_test,
            y_train=self.y_train,
            y_test=self.y_test
        )
        evaluator.print_metrics()

        return model


class FeatureImportanceVisualizer:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        
    def plot(self):
        """Visualizes feature importances."""
        feature_importances = self.model.feature_importances_
        important_features = sorted(
            zip(self.feature_names, feature_importances), key=lambda x: x[1], reverse=True
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

class MainWorkflow:
    def __init__(self, dataset_path, income_mun_path):
        self.dataset_path = dataset_path
        self.income_mun_path = income_mun_path
        
    def execute(self):
        """Executes the entire workflow."""
        # Step 1: Load data
        loader = DataLoader(self.dataset_path, self.income_mun_path)
        df, df_income = loader.load()

        # Step 2: Preprocess data
        preprocessor = DataPreProcessor(df, df_income)
        df = preprocessor.preprocess()

        # Step 3: Split data into training and testing sets
        X = df.drop(columns=["price"])
        y = df["price"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Step 4: Initialize models with best parameters
        best_params_randomforest = {
            "bootstrap": False,
            "max_depth": 30,
            "max_features": "sqrt",
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "n_estimators": 300,
        }
        best_params_decisiontree = {
            'criterion': 'friedman_mse', 
            'max_depth': 30, 
            'max_features': None, 
            'min_samples_leaf': 1, 
            'min_samples_split': 2, 
            'splitter': 'best'
        }

        model_randomforest = RandomForestRegressor(**best_params_randomforest, random_state=42)
        model_decisiontree = DecisionTreeRegressor(**best_params_decisiontree, random_state=42)

        # Step 5: Train and evaluate models
        trainer = ModelTrainer(X_train, X_test, y_train, y_test)
        
        model_randomforest = trainer.train_and_evaluate(model_randomforest, "RandomForest")
        model_decisiontree = trainer.train_and_evaluate(model_decisiontree, "DecisionTree")

        # Step 6: Visualize feature importances
        visualizer = FeatureImportanceVisualizer(model_randomforest, X_train.columns)
        visualizer.plot()

if __name__ == "__main__":
    # File paths
    dataset_path = "./data/raw/dataset_province_municipality_code_large.csv"
    income_mun_path = "./data/raw/income_municipality.csv"

    # Execute the workflow
    workflow = MainWorkflow(dataset_path, income_mun_path)
    workflow.execute()