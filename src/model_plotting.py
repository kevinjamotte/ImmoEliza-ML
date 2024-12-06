import matplotlib.pyplot as plt


def plot_actual_vs_predicted(y_actual, y_pred, model_name="Model"):
    """
    Plots a scatter plot of actual vs. predicted values for a regression model.

    Parameters:
    - y_actual: array-like, the actual target values.
    - y_pred: array-like, the predicted target values.
    - model_name: str, name of the model (used in the plot title).

    Returns:
    - None
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_actual, y_pred, alpha=0.6, color="blue", label="Predicted vs Actual")
    plt.plot(
        [y_actual.min(), y_actual.max()],
        [y_actual.min(), y_actual.max()],
        "--r",
        linewidth=2,
        label="Perfect Prediction",
    )
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"{model_name}: Actual vs. Predicted Values")
    plt.legend()
    plt.grid(True)
    plt.show()
