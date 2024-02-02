import math
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,  mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import MultiTaskLassoCV
import logging
from cleaning_preprocessing_func import *
logging.basicConfig(level=logging.INFO)

random_seed = np.random.RandomState(42)

def performance_metrics(y_pred, y_test, decimals):
    """Calculates metrics for the ML model

    Parameters:
        y_pred (Dataframe): The predictions
        y_test (Dataframe): The true values
        decimals (int): The number of decimals
    """
    # Calculate Root Mean Squared Error
    print('RMSE:', round(math.sqrt(mean_squared_error(y_test, y_pred)), decimals))
    # Calculate Mean Absolute Error
    print('MAE:', mean_absolute_error(y_test, y_pred).round(decimals))
    # Calculate R-squared
    print('R-Squared:', r2_score(y_test, y_pred).round(decimals))
    
# Cristian Path
PATH_REPO= r'C:/Users/ASUS/OneDrive/Documents/GitHub/2022-23d-1fcmgt-reg-ai-01-group-team4'

# Nick path
#PATH_REPO = r'C:\Users\nickb\OneDrive\Documenten\GitHub\2022-23d-1fcmgt-reg-ai-01-group-team4'

PATHS_MODELLING = [
    (f'{PATH_REPO}/data/modelling_X_y/X.csv', ','),
    (f'{PATH_REPO}/data/modelling_X_y/y.csv', ','),
]

# Load datasets
X, y = load_csvs(PATHS_MODELLING)

y_columns = y.columns.to_list()

# Split into test/train datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state = random_seed)

# Define the model, train it and predict
model = RandomForestRegressor(max_depth=10, min_samples_leaf=4, min_samples_split=20, n_estimators=100, random_state = random_seed)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# SHow performance metrics
performance_metrics(y_pred, y_test, 3)

# Calculate the percentage of correct predictions for each column
for i in range(0, len(y_columns)):
    # Number of correct predictions
    correct_predictions = (y_pred[:,i] == y_test[y_columns[i]]).sum()
    # Number of predictions
    total_predictions = len(y_pred)
    # Calculate accuracy
    accuracy = correct_predictions / total_predictions * 100
    print(f"Accuracy for {y_columns[i]}: {accuracy:.2f}%")