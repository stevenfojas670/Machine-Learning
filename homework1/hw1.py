import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

data = pd.read_csv('MNIST_100.csv')
y = data.iloc[:, 0]
x = data.drop('label', axis=1)

pca = PCA(n_components=2)
pca.fit(x)
PCAX = pca.transform(x)

print(x.shape)
print(y.shape)

colors = {0: 'red', 1: 'green', 2: 'blue', 3: 'cyan', 4: 'magenta',
          5: 'yellow', 6: 'black', 7: 'pink', 8: 'orange', 9: 'purple'}

plt.figure(figsize=(8, 6))
plt.plot(PCAX[:, 0], PCAX[:, 1], 'wo', )

for i in range(len(y)):
    plt.text(PCAX[i:i + 1, 0], PCAX[i:i + 1, 1], str(y.iloc[i]), color=colors[y.iloc[i]])

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('MNIST Data Reduced to 2D using PCA')
plt.show()

data = pd.read_csv('housing_training.csv')

col_K = data.iloc[:, 10]
col_M = data.iloc[: , 12]
col_N = data.iloc[:, 13]

plt.figure(figsize=(8,6))
plt.boxplot([col_K, col_M, col_N])
plt.title("Housing Data")
plt.ylabel("Value")
plt.xticks([1, 2, 3], ['Column K', 'Column M', 'Column N'])
plt.show()

col_A = data.iloc[:, 0]
plt.figure(figsize=(8,6))
plt.hist(col_A, bins=10, edgecolor='black')
plt.title('Column A Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()