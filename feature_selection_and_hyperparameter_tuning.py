import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
random_seed = np.random.RandomState(42)

# STINGA
REPO_PATH = r'C:/Users/ASUS/OneDrive/Documents/GitHub/2022-23d-1fcmgt-reg-ai-01-group-team4'

# Nick
REPO_PATH = r'C:\Users\nickb\OneDrive\Documenten\GitHub\2022-23d-1fcmgt-reg-ai-01-group-team4'

X = pd.read_csv(f'{REPO_PATH}/data/modelling_X_y/X.csv')
y = pd.read_csv(f'{REPO_PATH}/data/modelling_X_y/y.csv')
Xy = pd.read_csv(f'{REPO_PATH}/data/modelling_X_y/Xy_full.csv')

X_selected = Xy.iloc[:, :-4]
y_selected = Xy.iloc[:, -4:]
# Define model for Feature Selection and fit it
lasso = MultiTaskLassoCV(cv=5)
lasso.fit(X_selected, y_selected)
# Get the importance of features
importance = lasso.coef_
features = X.columns.to_list()
# Save the important columns/features to a new dataframe
X_selected = X[np.array(features)[importance[0] != 0]]
X_selected.to_csv(f'{REPO_PATH}/data/modelling_X_y/X_selected_full.csv', index=False)

# Create a pipeline for hyperparameter tuning
pipeline = Pipeline([
    ('std_slc', StandardScaler()),
    ('forest', RandomForestRegressor())
    ])

# Dictionary of parameters
search_param = dict(forest__criterion=['squared_error', 'poisson'], 
                  forest__max_depth = [None, 2, 5, 10, 25, 100],
                  forest__min_samples_split = [2, 3, 5, 10, 20],
                  forest__min_samples_leaf = [2, 4, 8, 10, 50],
                  forest__bootstrap=[True, False], 
                  forest__n_estimators=[50, 100, 200])

# Search for a combination of good prameters
search = RandomizedSearchCV(pipeline, search_param, verbose=2, n_iter=23)
search.fit(X_selected, y_selected)

# Show the best parameters
print("Best parameters:")
print(search.best_estimator_.get_params()['forest'])