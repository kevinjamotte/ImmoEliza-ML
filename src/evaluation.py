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
        Calculates MSE, RMSE, R2, and MAE for the model on the training and test data.
        """
        # Calculate MSE RMSE and R2
        mse = mean_squared_error(self.y_test, self.y_test_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.y_test, self.y_test_pred)

        # Calculate MAE
        mae_train = mean_absolute_error(self.y_train, self.y_train_pred)
        mae_test = mean_absolute_error(self.y_test, self.y_test_pred)

        # Calculate MAPE and sMAPE
        mape_value = self.mape(self.y_test, self.y_test_pred)
        smape_value = self.smape(self.y_test, self.y_test_pred)

        return mse, rmse, r2, mae_train, mae_test, mape_value, smape_value

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
        Prints out the metrics for MSE, RMSE, R2, MAE, MAPE, and sMAPE.
        """
        mse, rmse, r2, mae_train, mae_test, mape_value, smape_value = (
            self.calculate_metrics()
        )

        print(f"Mean Squared Error {self.model}: {mse:.2f}")
        print(f"Root Mean Squared Error {self.model}: {rmse:.2f}")
        print(f"R-squared {self.model}: {r2:.2f}")
        print(f"MAE for Train Data {self.model}: {mae_train:.2f}")
        print(f"MAE for Test Data {self.model}: {mae_test:.2f}")
        print(f"MAPE {self.model}: {mape_value:.2f}%")
        print(f"sMAPE {self.model}: {smape_value:.2f}%")
