import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

# Implement SVM with MNIST data using linear, poly, and rbf, then compute the accuracy with 5-fold CV
# Compare performance between each
# Should output 5 accuracies

df = pd.read_csv('MNIST_HW4.csv')

# Setting X and Y
y = df['label']
X = df.drop('label', axis=1)

# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Test size for a small dataset is recommended to be 20% testing and 80%
# We typically want to train with as much data as possible

# SVM models with different kernels
kernels = ['linear', 'poly', 'rbf']
results = {}

# Implementing 5-Fold CV
for kernel in kernels:
    model = SVC(kernel=kernel)
    cv_scores = cross_val_score(model, X_scaled, y, cv=5)
    results[kernel] = cv_scores

sum=0 # Sum of all kernel scores

# Outputting the accuracy results
print("Testing Results:")
for kernel, scores in results.items():
    print(f"Kernel: {kernel}, Accuracy: {scores.mean():.2f}")
    sum += scores.mean()

average = sum / len(kernels)
print(f"Average: {round(average, 2)}")
