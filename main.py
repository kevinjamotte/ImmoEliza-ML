import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from src.data_cleaning import DataFrameCleaner
from src.extra_features import Income_Municipality
from src.filtering import Postal_Filtering, One_Hot, BedroomsFiltering
from src.outliers import ZScoreFilter
from src.outliers import OutlierRemover
from src.evaluation import ModelEvaluation

# Setting up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class DataLoader:
    def __init__(self, dataset_path, income_mun_path):
        self.dataset_path = dataset_path
        self.income_mun_path = income_mun_path

    def load(self):
        try:
            df = pd.read_csv(self.dataset_path)
            df_income = pd.read_csv(self.income_mun_path)
            return df, df_income
        except Exception as e:
            logging.error(f"Error loading data: {e}")


class DataPreProcessor:
    def __init__(self, df, df_income):
        self.df = df
        self.df_income = df_income

    def preprocess(self):
        # Apply all the preprocessing steps
        extra_feature = Income_Municipality(self.df, self.df_income)
        self.df = extra_feature.add_feature()

        postal_filter = Postal_Filtering(self.df, 10)
        self.df = postal_filter.postal_filtering()

        bedrooms_filter = BedroomsFiltering(self.df, 8)
        self.df = bedrooms_filter.bedrooms_filtering()

        one_hot = One_Hot(self.df, "province")
        self.df = one_hot.one_hot_encoder()

        cleaner = DataFrameCleaner(self.df)
        self.df = cleaner.clean()

        zscore_filter = ZScoreFilter(
            self.df, columns=["price", "livingarea"], threshold=3
        )
        self.df = zscore_filter.filter()

        outlier_remover = OutlierRemover(
            self.df, columns=["price", "livingarea"], threshold=1.5
        )
        # self.df = outlier_remover.remove_outliers()
        
        logging.info("Data preprocessing completed")
        return self.df


class ModelTrainer:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def train_and_evaluate(self, model, model_name):
        # Train the model
        model.fit(self.X_train, self.y_train)

        # Evaluate the model
        evaluator = ModelEvaluation(
            model, self.X_train, self.X_test, self.y_train, self.y_test
        )
        print("Calling evaluator.print_metrics()...")
        evaluator.print_metrics()
        print("Finished calling evaluator.print_metrics()...")

        return model
    

class FeatureImportanceVisualizer:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        
    def plot(self):
        feature_importances = self.model.feature_importances_
        important_features = sorted(
            zip(self.feature_names, feature_importances),
            key=lambda x: x[1],
            reverse=True,
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


class OverfittingAnalyzer:
    """Analyzes overfitting using learning curves."""

    def __init__(self, model, X_train, y_train, X_test, y_test):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def plot_learning_curve(self):
        """Generate and plot the learning curve."""
        train_sizes, train_scores, test_scores = learning_curve(
            self.model,
            self.X_train,
            self.y_train,
            cv=5,
            scoring="r2",
            train_sizes=np.linspace(0.1, 1.0, 10),
            n_jobs=-1,  # Use multiple processors for speed
        )

        # Calculate mean and standard deviation
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        # Plot learning curves
        plt.figure(figsize=(12, 6))
        plt.plot(train_sizes, train_mean, label="Train R^2", color="blue", marker="o")
        plt.fill_between(
            train_sizes,
            train_mean - train_std,
            train_mean + train_std,
            alpha=0.1,
            color="blue",
        )
        plt.plot(train_sizes, test_mean, label="Test R^2", color="green", marker="o")
        plt.fill_between(
            train_sizes,
            test_mean - test_std,
            test_mean + test_std,
            alpha=0.1,
            color="green",
        )
        plt.xlabel("Training Set Size")
        plt.ylabel("R^2 Score")
        plt.title("Learning Curve Analysis")
        plt.legend()
        plt.grid()
        plt.show()
        logging.info("Learning curve plotted.")


class DecisionTreeVisualizer:
    """
    Class for visualizing a decision tree model using sklearn's plot_tree method.
    """

    def __init__(self, model, feature_names):
        """
        Initializes the DecisionTreeVisualizer.

        Args:
            decision_tree_model: A trained DecisionTreeClassifier or DecisionTreeRegressor model.
            feature_names (list): List of feature names corresponding to the model's features.
        """
        self.model = model
        self.feature_names = feature_names

    def plot(self, figsize=(20, 10), fontsize=12):  # Set default fontsize here
        """
        Plots the decision tree using sklearn's plot_tree method.

        Args:
            figsize (tuple): The size of the figure to plot (default is (20, 10)).
            fontsize (int): The font size for the visualization.
        """
        plt.figure(figsize=figsize)
        plot_tree(
            self.model,
            feature_names=self.feature_names,
            filled=True,
            rounded=True,
            fontsize=fontsize,  # Explicitly setting fontsize
        )
        plt.title("Decision Tree Visualization")
        plt.show()

class MainWorkflow:
    def __init__(self, dataset_path, income_mun_path):
        self.dataset_path = dataset_path
        self.income_mun_path = income_mun_path

    def execute(self):
        # Step 1: Load data
        loader = DataLoader(self.dataset_path, self.income_mun_path)
        df, df_income = loader.load()

        # Step 2: Preprocess data
        preprocessor = DataPreProcessor(df, df_income)
        df = preprocessor.preprocess()

        # Step 3: Split data
        X = df.drop(columns=["price"])
        y = df["price"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Step 4: Initialize models
        best_params_randomtreeforest = {
            "bootstrap": True,
            "max_depth": 10,
            "max_features": "sqrt",
            "min_samples_leaf": 5,
            "min_samples_split": 10,
            "n_estimators": 300
        }

        best_params_decisiontree = {
            "criterion": "friedman_mse",
            "max_depth": 10,
            "max_features": None,
            "min_samples_leaf": 5,
            "min_samples_split": 10,
            "splitter": "best",
        }

        model_randomforest = RandomForestRegressor(random_state=42)
        model_decisiontree = DecisionTreeRegressor(random_state=42)

        # Step 5: Train and evaluate models
        trainer = ModelTrainer(X_train, X_test, y_train, y_test)
        logging.info("Training RandomForest model...")
        model_randomforest = trainer.train_and_evaluate(
            model_randomforest, "RandomForest"
        )

        logging.info("Training DecisionTree model...")
        model_decisiontree = trainer.train_and_evaluate(
            model_decisiontree, "DecisionTree"
        )
        # Analyze overfitting
        """        
        analyzer_rf = OverfittingAnalyzer(model_randomforest, X_train, y_train, X_test, y_test)
        analyzer_rf.plot_learning_curve()
        analyzer_dt = OverfittingAnalyzer(model_decisiontree, X_train, y_train, X_test, y_test)
        analyzer_dt.plot_learning_curve()
        """
        # Step 6: Visualize feature importances
        logging.info("Visualizing feature importances...")
        featureofimportancevisualizer = FeatureImportanceVisualizer(model_randomforest, X_train.columns)
        #featureofimportancevisualizer.plot()
        decisiontreevisualizer = DecisionTreeVisualizer(model_decisiontree, feature_names=X_train.columns)
        decisiontreevisualizer.plot(fontsize=16)


if __name__ == "__main__":
    # File paths
    dataset_path = "./data/raw/dataset_province_municipality_code_large.csv"
    income_mun_path = "./data/raw/income_municipality.csv"

    # Execute the workflow
    workflow = MainWorkflow(dataset_path, income_mun_path)
    workflow.execute()
