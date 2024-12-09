# model_evaluation.py (module)

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np


class ModelEvaluation:
    def __init__(self, model, X_train, X_test, y_train, y_test):
        """
        Initializes the ModelEvaluation object with model, training data, and test data.
        """
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        # Make predictions for both train and test data
        self.y_train_pred = model.predict(X_train)
        self.y_test_pred = model.predict(X_test)

    def calculate_metrics(self):
        """
        Calculates MSE, RMSE, R2, MAE, MAPE, and sMAPE for the model on both training and test data.
        """
        # Test set metrics
        mse_test = mean_squared_error(self.y_test, self.y_test_pred)
        rmse_test = np.sqrt(mse_test)
        r2_test = r2_score(self.y_test, self.y_test_pred)
        mae_test = mean_absolute_error(self.y_test, self.y_test_pred)
        mape_test = self.mape(self.y_test, self.y_test_pred)
        smape_test = self.smape(self.y_test, self.y_test_pred)

        # Training set metrics
        mse_train = mean_squared_error(self.y_train, self.y_train_pred)
        rmse_train = np.sqrt(mse_train)
        r2_train = r2_score(self.y_train, self.y_train_pred)
        mae_train = mean_absolute_error(self.y_train, self.y_train_pred)
        mape_train = self.mape(self.y_train, self.y_train_pred)
        smape_train = self.smape(self.y_train, self.y_train_pred)

        return (
            mse_test,
            rmse_test,
            r2_test,
            mae_test,
            mape_test,
            smape_test,
            mse_train,
            rmse_train,
            r2_train,
            mae_train,
            smape_train,
            mape_train,
        )

    def mape(self, y_true, y_pred):
        """
        Calculates Mean Absolute Percentage Error (MAPE).
        """
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    def smape(self, y_true, y_pred):
        """
        Calculates Symmetric Mean Absolute Percentage Error (sMAPE).
        """
        return (
            np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))
            * 100
        )

    def print_metrics(self):
        """
        Prints out the metrics for MSE, RMSE, R2, MAE, MAPE, and sMAPE for both test and training datasets.
        """
        (
            mse_test,
            rmse_test,
            r2_test,
            mae_test,
            mape_test,
            smape_test,
            mse_train,
            rmse_train,
            r2_train,
            mae_train,
            smape_train,
            mape_train,
        ) = self.calculate_metrics()

        metrics_message = (
            f"Metrics for {self.model}:\n\n"
            f"Test Metrics:\n"
            f"- Mean Squared Error (MSE): {mse_test:.2f}\n"
            f"- Root Mean Squared Error (RMSE): {rmse_test:.2f}\n"
            f"- R-squared: {r2_test:.2f}\n"
            f"- Mean Absolute Error (MAE): {mae_test:.2f}\n"
            f"- Mean Absolute Percentage Error (MAPE): {mape_test:.2f}%\n"
            f"- Symmetric Mean Absolute Percentage Error (sMAPE): {smape_test:.2f}%\n\n"
            f"Training Metrics:\n"
            f"- Mean Squared Error (MSE): {mse_train:.2f}\n"
            f"- Root Mean Squared Error (RMSE): {rmse_train:.2f}\n"
            f"- R-squared: {r2_train:.2f}\n"
            f"- Mean Absolute Error (MAE): {mae_train:.2f}\n"
            f"- Mean Absolute Percentage Error (MAPE): {mape_train:.2f}%\n"
            f"- Symmetric Mean Absolute Percentage Error (sMAPE): {smape_train:.2f}%"
        )

        print(metrics_message)
