from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from typing import Tuple

def train_linear_regression_model(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame) -> Tuple[LinearRegression, float]:
    """
    Trains a Linear Regression model, makes predictions on the test set, and evaluates the model using RMSE.

    Parameters:
    - X_train (pd.DataFrame): Features for training.
    - y_train (pd.Series): Target variable for training.
    - X_test (pd.DataFrame): Features for testing.

    Returns:
    - model (LinearRegression): Trained Linear Regression model.
    - rmse (float): Root Mean Squared Error on the test set.
    """
    # Features and target variable
    X = X_train
    y = y_train

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model using RMSE
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    return model, rmse