# Hierarchical Clustering

## Overview
This project demonstrates the use of the **Hierarchical Clustering** algorithm to segment customers based on their annual income and spending score. By creating a dendrogram and clustering customers into distinct groups, businesses can better understand customer behavior and target them more effectively.

## Dataset
The dataset used for this project is `"Mall_Customers.csv"`, which contains the following features:

- **CustomerID**: Unique identifier for each customer.
- **Gender**: Gender of the customer (not used in clustering).
- **Age**: Age of the customer (not used in clustering).
- **Annual Income (k$)**: Customer's annual income in thousands of dollars.
- **Spending Score (1-100)**: Score assigned to customers based on their spending behavior.

### Sample Data:

| CustomerID | Annual Income (k$) | Spending Score (1-100) |
|------------|--------------------|------------------------|
| 1000025    | 5                  | 1                      |
| 1002945    | 5                  | 4                      |
| 1015425    | 3                  | 1                      |
| 1016277    | 6                  | 8                      |
| 1017023    | 4                  | 1                      |

## Steps in the Code

### 1. Importing Necessary Libraries
```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```
- **Numpy**: Used for numerical computations.
- **Matplotlib**: Used for data visualization.
- **Pandas**: Used for data manipulation and analysis.

### 2. Loading the Dataset
```python
dataset = pd.read_csv('Mall_Customers.csv')
```
- The dataset is loaded into a Pandas DataFrame.

### 3. Extracting Features for Clustering
```python
X = dataset.iloc[:, [3, 4]].values
```
- The features **Annual Income** and **Spending Score** are selected from the dataset for clustering.

### 4. Creating a Dendrogram
```python
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()
```
- A dendrogram is plotted using the **Ward's method**, which helps to identify the optimal number of clusters by visualizing how data points merge together at different distances.

### 5. Applying Hierarchical Clustering
```python
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, metric='euclidean', linkage='ward')
y_hc = hc.fit_predict(X)
```
- **Agglomerative Clustering** is used with **Euclidean distance** as the metric and **Ward's method** as the linkage criterion.
- Based on the dendrogram, 5 clusters are selected for the final clustering.

### 6. Visualizing the Clusters
```python
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s=100, c='cyan', label='Cluster 4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s=100, c='magenta', label='Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
```
- The clusters of customers are visualized on a 2D plot, with different colors representing different clusters based on their **Annual Income** and **Spending Score**.

## Installation

1. Install the required libraries:
```bash
pip install numpy pandas matplotlib scikit-learn scipy
```

2. Download the dataset and place it in the same directory as the script.

## Running the Project

1. Ensure that the dataset (`Mall_Customers.csv`) is available.
2. Run the Python script to generate a dendrogram and apply hierarchical clustering to visualize the customer segments.

## Key Points
- **Dendrogram**: Helps in identifying the optimal number of clusters by showing the hierarchy of data point merging.
- **Agglomerative Clustering**: Groups similar customers based on **Annual Income** and **Spending Score**.
- **Visualization**: Provides insights into customer segmentation, helping businesses target specific customer groups effectively.
