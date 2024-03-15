import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prettytable import PrettyTable as pt

data = pd.read_csv('auto-mpg.data.csv')

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
RMSE = []
predictions = []
b_optimized = []

myTable = pt(["", "Prediction", "Cylinders", "Displacement", "Horsepower", "Weight", "Acceleration", "Model Year", "RMSE"])

# Linear Regression Using RMSE

for index in range(K):

    test_indices = indices[index]

    train_indices = np.concatenate([indices[i] for i in range(K) if i != index])

    test_X = X[test_indices]
    test_y = y[test_indices]
    train_X = X[train_indices]
    train_y = y[train_indices]

    # Implement Linear Regression
    b_optimized = np.dot(np.dot(np.linalg.inv(np.dot(train_X.transpose(), train_X)), train_X.transpose()), train_y)

    # Calculating predictions
    predictions = np.dot(test_X, b_optimized)

    # Calculating RMSE for the current fold
    RMSE = np.sqrt((predictions - test_y)**2).mean()

    myTable.add_row(np.array(["Fold " + str(index+1), *b_optimized, RMSE], dtype=object))

print("Table")
print(myTable)