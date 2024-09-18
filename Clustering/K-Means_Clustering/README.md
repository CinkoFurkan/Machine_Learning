# K-Means Clustering

## Overview
This project demonstrates the use of the K-Means clustering algorithm to segment customers based on their annual income and spending score. The data is visualized to identify clusters of customers with similar purchasing behavior, helping businesses understand their customer base better.

## Dataset
The dataset used for this project is `"Mall_Customers.csv"`, which contains the following features:

- **CustomerID**: Unique identifier for each customer.
- **Gender**: Gender of the customer (not used in clustering).
- **Age**: Age of the customer (not used in clustering).
- **Annual Income (k$)**: Customer's annual income in thousands of dollars.
- **Spending Score (1-100)**: Score assigned to customers based on their behavior and spending.

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
- **Matplotlib**: Used for creating visualizations.
- **Pandas**: Used for data manipulation.

### 2. Loading the Dataset
```python
dataset = pd.read_csv("Mall_Customers.csv")
```
- The dataset is loaded into a Pandas DataFrame.

### 3. Extracting Features for Clustering
```python
X = dataset.iloc[:, [3, 4]].values
```
- Columns for **Annual Income** and **Spending Score** are selected for clustering.

### 4. Applying K-Means Clustering
```python
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
```
- K-Means clustering is applied with a range of cluster numbers (1 to 10) to compute the **Within-Cluster-Sum of Squares** (WCSS) for each.
- **`k-means++`**: Used for smart centroid initialization, which helps with faster convergence.

### 5. Plotting the Elbow Method
```python
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
```
- The **Elbow Method** is used to find the optimal number of clusters by visualizing the WCSS values.

### 6. Applying K-Means with Optimal Clusters
```python
kmeans = KMeans(n_clusters=5, init="k-means++", random_state=42)
y_kmeans = kmeans.fit_predict(X)
```
- Based on the Elbow Method, 5 clusters are chosen for K-Means clustering.

### 7. Visualizing the Clusters and Centroids
```python
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=100, c='cyan', label='Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=100, c='magenta', label='Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
```
- The clusters and their respective centroids are visualized using a scatter plot, with different colors representing different clusters and the centroids highlighted in yellow.

## Installation

1. Install required libraries:
```bash
pip install numpy pandas matplotlib scikit-learn
```

2. Download the dataset and place it in the same directory as the script.

## Running the Project

1. Ensure that the dataset (`Mall_Customers.csv`) is available.
2. Run the Python script to perform K-Means clustering and visualize the results.
