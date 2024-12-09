import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import pandas as pd


class LearningCurve:
    """
    A class for analyzing learning curves.
    """

    def __init__(
        self,
        model: BaseEstimator,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_test: pd.DataFrame,
    ) -> None:
        """
        Initializes the LearningCurve.

        Args:
            model: Trained regression model (DecisionTree or RandomForest).
            X_train: Training feature data.
            y_train: True target values for training.
            X_test: Testing features.
            y_test: True target values for testing.
        """

        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def plot_learning_curve(self) -> None:
        """
        Generate and plot the learning curve.
        """
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
        plt.title(f"Learning Curve Analysis for {self.model}")
        plt.legend()
        plt.grid()
        plt.savefig(f"LearningCurve_{type(self.model).__name__}.png")
        plt.show()
        logging.info("Learning curve plotted.")


class PredictionVsActualPlotter:
    """
    A class for visualizing predicted vs actual prices for regression models.
    """

    def __init__(
        self, model: BaseEstimator, X_test: pd.DataFrame, y_test: pd.DataFrame
    ) -> None:
        """
        Initializes the PredictionVsActualPlotter.

        Args:
            model: Trained regression model (DecisionTree or RandomForest).
            X_test: Testing features.
            y_test: True target values for testing.
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test

    def plot(self) -> None:
        """
        Generates the scatter plot of predicted vs actual prices.
        """
        # Generate predictions
        y_pred = self.model.predict(self.X_test)

        # Plot the scatter plot
        plt.figure(figsize=(8, 6))
        plt.scatter(self.y_test, y_pred, color="blue", alpha=0.5)
        plt.plot(
            [self.y_test.min(), self.y_test.max()],
            [self.y_test.min(), self.y_test.max()],
            "r--",
            lw=2,
        )  # Reference line
        plt.xlabel("Actual Price")
        plt.ylabel("Predicted Price")
        plt.title(f"Actual vs Predicted Prices for {self.model} with test set")
        plt.savefig(f"PredictionVsActual{type(self.model).__name__}.png")
        plt.show()


class FeatureImportanceVisualizer:
    def __init__(self, model: BaseEstimator, feature_names: list) -> None:
        """
        Initializes the FeatureOfImportanceVisualizer.

        Args:
            model: Trained regression model (DecisionTree or RandomForest).
            feature_name : name of the
        """
        self.model = model
        self.feature_names = feature_names

    def plot(self) -> None:
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
        plt.title(f"Feature Importances for {self.model}")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.savefig(f"FeatureOfImportance{type(self.model).__name__}.png")
        plt.show()


class ResidualsPlotter:
    """
    A class for visualizing residuals for regression models.
    """

    def __init__(
        self, model: BaseEstimator, X_test: pd.DataFrame, y_test: pd.DataFrame
    ) -> None:
        """
        Initializes the ResidualsPlotter.

        Args:
            model: Trained regression model (DecisionTree or RandomForest).
            X_test: Testing features.
            y_test: True target values for testing.
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test

    def plot(self) -> None:
        """
        Generates a residuals plot comparing actual prices and residuals.
        """
        # Generate predictions
        y_pred = self.model.predict(self.X_test)
        residuals = self.y_test - y_pred

        # Plot residuals
        plt.figure(figsize=(8, 6))
        plt.scatter(self.y_test, residuals, color="green", alpha=0.5)
        plt.axhline(0, linestyle="--", color="gray")
        plt.xlabel("Actual Price")
        plt.ylabel("Residuals")
        plt.title(f"Residuals Plot for {self.model}")
        plt.savefig(f"ResidualsPlot{type(self.model).__name__}.png")
        plt.show()


class PredictedPriceDistributionPlotter:
    """
    A class for visualizing the distribution of predicted prices for regression models.
    """

    def __init__(self, model: BaseEstimator, X_test: pd.DataFrame) -> None:
        """
        Initializes the PredictedPriceDistributionPlotter.

        Args:
            model: Trained regression model.
            X_test: Testing features.
        """
        self.model = model
        self.X_test = X_test

    def plot(self) -> None:
        """
        Generates a histogram of predicted prices to analyze their distribution.
        """
        # Generate predictions
        y_pred = self.model.predict(self.X_test)

        # Plot histogram
        plt.figure(figsize=(8, 6))
        plt.hist(y_pred, bins=30, color="purple", alpha=0.7, label="Predicted Price")
        plt.xlabel("Predicted Price")
        plt.ylabel("Frequency")
        plt.title(f"Histogram of Predicted Prices for {self.model}")
        plt.legend()
        plt.savefig(f"PredictorPriceDistribution{type(self.model).__name__}.png")
        plt.show()
