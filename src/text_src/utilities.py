from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

# Model evaluation
def model_evaluation(estimator, X, y):
    predicted_values = estimator.predict(X)
    MSELoss = mean_squared_error(predicted_values, y)
    RMSELoss = np.sqrt(MSELoss)
    r2score = r2_score(predicted_values, y)
    print("RMSE loss: ", RMSELoss)
    print("R2 score: ", r2score)
    return predicted_values

# Cross validation
def cross_validation(estimator, X, y, score = 'r2', n = 5):
    validate = cross_val_score(estimator, X, y, scoring = score, cv = n)
    print("Mean valiation R2 score: ", validate.mean())

# Histplot of the residuals
def histplot(residuals):
    sns.histplot(
        data=residuals,
        kde=True,
        color='red'
    )

# Plot the residuals
def plot(residuals):
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(residuals)), residuals, color='blue') 
    plt.xlabel('Index')
    plt.ylabel('Residuals')
    plt.show()

# Hypertunning
def hyper_tune(estimator, X, y, param_grid, score = 'r2', n = 5):
    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=score,
        cv=n
    )
    grid_search.fit(X, y)
    best_score = grid_search.best_score_
    print("Best R2 score: ", best_score)
    # Add this line if we wish to return the best_degree for polynomial features
    # And remember to return best_degree
    # best_degree = grid_search.best_params_['poly_feat__degree']
    return grid_search.best_estimator_