from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

def train_lasso_model(X_train, y_train, X_test):
    """
    Train Lasso Regression model, optimize hyperparameters, and evaluate its performance.

    Parameters:
    - X_train (DataFrame): Training features.
    - y_train (Series): Target variable for training.
    - X_test (DataFrame): Testing features.

    Returns:
    - Tuple: Tuple containing predicted values and root mean squared error (RMSE).

    Examples:
    ```
    # Features and target variable
    X = dataset[['Likes', 'Dislikes', 'Followers']]
    y = dataset['Price']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and evaluate Lasso model
    predicted_values, rmse = train_lasso_model(X_train, y_train, X_test)
    ```
    """
    lasso_model = Lasso()
    param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}  # Adjust alpha values as needed
    lasso_grid = GridSearchCV(lasso_model, param_grid, cv=5, scoring='neg_mean_squared_error')
    lasso_grid.fit(X_train, y_train)

    # Make predictions
    y_pred_lasso = lasso_grid.predict(X_test)

    # Evaluate the model
    rmse_lasso = mean_squared_error(y_test, y_pred_lasso, squared=False)
    print(f'Lasso Regression RMSE: {rmse_lasso}')

    # Print best hyperparameter
    print(f'Best Lasso Hyperparameter: {lasso_grid.best_params_}')

    return y_pred_lasso, rmse_lasso

def train_random_forest_model(X_train, y_train, X_test):
    """
    Train Random Forest Regression model and evaluate its performance.

    Parameters:
    - X_train (DataFrame): Training features.
    - y_train (Series): Target variable for training.
    - X_test (DataFrame): Testing features.

    Returns:
    - Tuple: Tuple containing predicted values and root mean squared error (RMSE).

    """
    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X_train, y_train)

    # Make predictions
    y_pred_rf = rf_model.predict(X_test)

    # Evaluate the model
    rmse_rf = mean_squared_error(y_test, y_pred_rf, squared=False)
    print(f'Random Forest Regression RMSE: {rmse_rf}')

    return y_pred_rf, rmse_rf