import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


df = pd.read_csv('MNIST_HW4.csv')
X = df.iloc[:, 1:].values.reshape(-1, 28, 28, 1)  # Reshape for CNN
Y = df.iloc[:, 0].values

# Normalize the input to [0,1] to increase convergence
X = X / 255.0


def create_cnn_model():
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def cross_validate(model_func):
    accuracies = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold = 1
    for train_index, test_index in kf.split(X):
        model = model_func()
        print(f"Training fold {fold}...")
        history = model.fit(X[train_index], Y[train_index], epochs=10, verbose=0,
                            validation_data=(X[test_index], Y[test_index]))

        # Evaluating the model
        _, accuracy = model.evaluate(X[test_index], Y[test_index], verbose=0)
        accuracies.append(accuracy)
        print(f"Accuracy for fold {fold}: {accuracy:.5f}")
        fold += 1

        # Plotting learning curve for the last fold
        if fold == 5:
            plt.figure()
            plt.plot(history.history['accuracy'], label='Training Accuracy')
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
            plt.title('Learning Curve')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.show()

    print(f"Average Accuracy: {np.mean(accuracies):.5f}")
    return np.mean(accuracies)


print("Evaluating CNN Model")
cnn_accuracy = cross_validate(create_cnn_model)
