import numpy as np
import pandas as pd

# 10 classes, 95 samples each
train_data = pd.read_csv('MNIST_training.csv', header=None)
# 10 classes, 5 samples each
test_data = pd.read_csv('MNIST_test.csv', header=None)

# Display the first few rows of each dataset to understand their structure
train_data_head = train_data.head()
test_data_head = test_data.head()

# print(train_data_head, test_data_head)

# Remove the first row (header) and convert data to integers
train_data = train_data.iloc[1:].reset_index(drop=True).astype(int)
test_data = test_data.iloc[1:].reset_index(drop=True).astype(int)

# Display the first row for verification
# print(train_data.head(), test_data.head())

# Separate labels and pixel values for both datasets
train_labels = train_data.iloc[:, 0]
train_pixels = train_data.iloc[:, 1:]

test_labels = test_data.iloc[:, 0]
test_pixels = test_data.iloc[:, 1:]

# Display the first few rows of labels and pixels for verification
# print(train_labels.head(), train_pixels.head(), test_labels.head(), test_pixels.head())

def knn_predict(test_pixels, train_pixels, train_labels, k=3):
    predictions = []
    for i in range(len(test_pixels)):
        # Calculate Euclidean distances between the test sample and all training samples
        dists = np.sqrt(((train_pixels - test_pixels.iloc[i]) ** 2).sum(axis=1))

        # Find the indices of the k smallest distances
        nearest_indices = dists.nsmallest(k).index

        # Find the labels of the nearest neighbors
        nearest_labels = train_labels.loc[nearest_indices]

        # Determine the majority class among the nearest neighbors
        majority_label = nearest_labels.mode()[0]

        predictions.append(majority_label)

    return predictions


# Predict labels for the test set
k = 7
predicted_labels = knn_predict(test_pixels, train_pixels, train_labels, k)

# Calculate accuracy
correct_predictions = np.sum(predicted_labels == test_labels.values)
incorrect_predictions = np.sum(predicted_labels != test_labels.values)
accuracy = correct_predictions / len(test_labels)

print(f'Predicted Labels: {predicted_labels}')
print(f'Correctly Identified Predictions: {correct_predictions}')
print(f'Incorrectly Identified Predictions: {incorrect_predictions}')
print(f'Accuracy for K={k}: {accuracy}')