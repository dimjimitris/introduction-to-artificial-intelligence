import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

from datasets import load_dataset

DATASET_NAME = 'diabetes'

def set_seed(seed):
    """Set the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)

def print_correlation_matrix(df):
    """
    Generate and display a correlation matrix for the DataFrame.
    Helps identify closely related features that can cause multicollinearity.
    """
    corr_matrix = df.corr(numeric_only=True)
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, vmin=-1, vmax=1, cmap='Blues', annot=True)
    plt.xticks(rotation=45)
    plt.show()

if __name__ == "__main__":
    # Set seed for reproducibility
    seed = 0
    set_seed(seed)

    # Load and preprocess dataset
    X, y = load_dataset(DATASET_NAME)

    # Combine features and target
    df = pd.concat([X, y], axis=1)

    # Uncomment to display the first few rows of the dataset
    # print(df.head())

    # Uncomment to display dataset structure and datatype information
    # df.info()    

    # Display correlation matrix
    print_correlation_matrix(df)

    '''
    Strong correlations between features:
    Positive:
    - bmi, target
    - s1, s2
    - s1, s4
    - s1, s5
    - s2, s4
    - s4, s5
    - s5, target

    Negative:
    - s3, s4

    We can consider removing some of these features to reduce multicollinearity.
    We keep bmi, target, s1, s5 and s3. We remove s2, s4.
    
    '''

    # Drop features with strong correlations
    X = X.drop(columns=['s2', 's4'])

    # Scale features to standard normalization
    scaling_factor = np.sqrt(len(X))
    X = X * scaling_factor

    print("\nFeature statistics after scaling:\n", X.describe())

    # Split data into train and test partitions with 80% train and 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    # Define the models
    # Linear Regression and Random Forest
    linear_model = LinearRegression(
        copy_X=True,
    )
    random_forest = RandomForestRegressor(
        n_estimators=80,
        random_state=seed
    )

    # Evaluate models using 4-fold cross-validation and RMSE metric
    linear_cv_rmse = -cross_val_score(linear_model, X_train, y_train, cv=4, scoring='neg_root_mean_squared_error').mean()
    rf_cv_rmse = -cross_val_score(random_forest, X_train, y_train, cv=4, scoring='neg_root_mean_squared_error').mean()

    print("\nRMSE using 4-fold cross-validation:")
    print(f"- Linear Regression: {linear_cv_rmse:.3f}")
    print(f"- Random Forest Regressor: {rf_cv_rmse:.3f}")

    # Fit models on the entire training set
    final_linear_model = linear_model.fit(X_train, y_train)
    final_random_forest = random_forest.fit(X_train, y_train)

    # Calculate predictions on both training set and test set
    predictions_linear_train = final_linear_model.predict(X_train)
    predictions_rf_train = final_random_forest.predict(X_train)

    predictions_linear_test = final_linear_model.predict(X_test)
    predictions_rf_test = final_random_forest.predict(X_test)

    # Evaluate the final predictions and compare model performance on both
    # Training and Test sets to detect potential overfitting
    rmse_linear_train = root_mean_squared_error(y_train, predictions_linear_train)
    rmse_rf_train = root_mean_squared_error(y_train, predictions_rf_train)

    rmse_linear_test = root_mean_squared_error(y_test, predictions_linear_test)
    rmse_rf_test = root_mean_squared_error(y_test, predictions_rf_test)

    print("\nModel RMSE evaluation on Training and Test Sets:")
    print(f"- Linear Regression - Training RMSE: {rmse_linear_train:.3f}")
    print(f"- Linear Regression - Test RMSE: {rmse_linear_test:.3f}")
    print(f"- Random Forest Regressor - Training RMSE: {rmse_rf_train:.3f}")
    print(f"- Random Forest Regressor - Test RMSE: {rmse_rf_test:.3f}")

    #  Hyperparameter Tuning Ridge Regression
    print("\n--- Hyperparameter Tuning (Ridge Regression) ---\n")

    # Ridge Regression
    ridge_model = Ridge(copy_X=True)
    
    # Set of parameters to test
    param_grid_ridge = {
        'alpha' : [0.001, 0.01, 0.1, 1, 10, 50, 100, 500]
    }

    grid_search_ridge = GridSearchCV(estimator=ridge_model, param_grid=param_grid_ridge, cv=4, scoring='neg_root_mean_squared_error')

    grid_search_ridge.fit(X_train, y_train)

    # Display the mean test scores for all the parameter combinations
    print("RMSE computed:")
    for mean, params in zip(-grid_search_ridge.cv_results_['mean_test_score'], grid_search_ridge.cv_results_['params']):
        print(f"{mean:.3f} for {params}")

    best_params_ridge = grid_search_ridge.best_params_
    best_score_ridge = -grid_search_ridge.best_score_

    print(f"\nBest parameters for Ridge Regression: {best_params_ridge}")
    print(f"Best RMSE for Ridge Regression: {best_score_ridge:.3f}")

    #  Hyperparameter tuning Random Forest Regressor
    print("\n--- Hyperparameter Tuning (Random Forest Regressor) ---\n")
    param_grid_rf = {
        'n_estimators': [50, 100],
        'max_depth': [10, None],
        'min_samples_split': [6, 10],
        'min_samples_leaf': [3, 4],
        'max_features': ['sqrt']
    }
    grid_search_rf = GridSearchCV(estimator=random_forest, param_grid=param_grid_rf, cv=4, scoring='neg_root_mean_squared_error')

    grid_search_rf.fit(X_train, y_train)

    # Print the mean test scores for all the parameter combinations
    print("RMSE computed:")
    for mean, params in zip(-grid_search_rf.cv_results_['mean_test_score'], grid_search_rf.cv_results_['params']):
        print(f"{mean:.3f} for {params}")

    best_params_rf = grid_search_rf.best_params_
    best_score_rf = -grid_search_rf.best_score_

    print(f"\nBest parameters for Random Forest Regressor: {best_params_rf}")
    print(f"Best RMSE for Random Forest Regressor: {best_score_rf:.3f}\n")

    # Fit the best models on the entire training set and get the predictions
    final_ridge = grid_search_ridge.best_estimator_.fit(X_train, y_train)
    final_random_forest = grid_search_rf.best_estimator_.fit(X_train, y_train)

    predictions_final_ridge_train = final_ridge.predict(X_train)
    predictions_final_rf_train = final_random_forest.predict(X_train)
    predictions_final_ridge_test = final_ridge.predict(X_test)
    predictions_final_rf_test = final_random_forest.predict(X_test)

    # Evaluate the final predictions and compare model performance on both
    # Training and Tests set to detect potential overfitting
    rmse_final_ridge_train = root_mean_squared_error(y_train, predictions_final_ridge_train)
    rmse_final_rf_train = root_mean_squared_error(y_train, predictions_final_rf_train)
    rmse_final_ridge_test = root_mean_squared_error(y_test, predictions_final_ridge_test)
    rmse_final_rf_test = root_mean_squared_error(y_test, predictions_final_rf_test)

    print("Final RMSE evaluation on Training and Test sets with Best Models:")
    print(f"- Ridge Regression Best Model - Train RMSE: {rmse_final_ridge_train:.3f}")
    print(f"- Random Forest Best Model - Train RMSE: {rmse_final_rf_train:.3f}")
    print(f"- Ridge Regression Best Model - Test RMSE: {rmse_final_ridge_test:.3f}")
    print(f"- Random Forest Best Model - Test RMSE: {rmse_final_rf_test:.3f}")
