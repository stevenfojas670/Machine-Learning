import numpy as np
import pandas as pd
import matplotlib as plt
from prettyTables import Table

data = pd.read_csv('auto-mpg.data.csv')

def linear_regression(X, y):
    """
    Computes the coefficients for linear regression using the Normal Equation.
    """
    b = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return b

def rmse(predictions, targets):
    """
    Computes the Root Mean Square Error.
    """
    return np.sqrt(((predictions - targets) ** 2).mean())

# Using the scheme of Xb = y
X = data.drop(['mpg', 'carname'], axis = 1).to_numpy()
X = np.insert(X, 0, np.ones(X.shape[0]), axis = 1) # inserting 1s to account for the slope intercept
y = data.iloc[:, 0].to_numpy() # dependent set

# Need to create K-Folds
K = 10

# Evenly partition the both y and X into 10 partitions
np.random.seed(X.shape[0])
indices = np.random.permutation(X.shape[0]) # Randomizing indices to shuffle rows

indices = np.array_split(indices, K)

# Placeholder for storing RMSE of each fold
RMSE_values = []

table = Table()

table.field_names = ["mpg", "Cylinder", "Displacement", "Horsepower", "Weight", "Acceleration", "Model_year", "RMSE"]

for index in range(K):

    test_indices = indices[index]

    train_indices = np.concatenate([indices[i] for i in range(K) if i != index])

    test_X = X[test_indices]
    test_y = y[test_indices]
    train_X = X[train_indices]
    train_y = y[train_indices]

    # Implement Linear Regression
    b_optimized = linear_regression(train_X, train_y)

    # Make predictions on the testing set
    predictions = test_X.dot(b_optimized)

    # Calculate RMSE for the current fold
    current_rmse = rmse(predictions, test_y)

    row = [index + 1]
    row.extend(b_optimized)
    row.append(current_rmse)

    table.add_row(row)

print(table)