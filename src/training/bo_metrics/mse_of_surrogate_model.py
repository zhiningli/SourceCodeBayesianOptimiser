from sklearn.metrics import mean_squared_error

def mse_of_surrogate_model(predictions, true_values):
    """
    Calculate the Mean Squared Error (MSE) for a surrogate model's predictions.

    Parameters:
    predictions (list or numpy array): The predicted values from the surrogate model.
    true_values (list or numpy array): The actual values of the objective function.

    Returns:
    float: The mean squared error.
    """
    predictions = np.array(predictions)
    true_values = np.array(true_values)

    if len(predictions) != len(true_values):
        raise ValueError("Predictions and true values must have the same length.")

    mse = mean_squared_error(true_values, predictions)
    return mse

