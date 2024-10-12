# Eclat Algorithm Project

## Overview
This project demonstrates the use of the **Eclat algorithm** for market basket analysis to find frequent itemsets in a dataset of customer transactions. The Eclat algorithm focuses on identifying co-occurring items, helping businesses understand which products are often bought together, and potentially aiding in inventory management or cross-selling strategies.

## Dataset
The dataset used for this project is `"Market_Basket_Optimisation.csv"`, which contains customer transactions with various purchased products. Each row represents a transaction, and each column corresponds to a product purchased within that transaction.

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
dataset = pd.read_csv("Market_Basket_Optimisation.csv", header=None)
```
- The dataset is loaded into a Pandas DataFrame, and since there are no headers in the dataset, `header=None` is specified.

### 3. Data Preprocessing
```python
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])
```
- Each transaction is appended to the `transactions` list as a list of products purchased in that specific transaction.

### 4. Applying the Eclat Algorithm
```python
from apyori import apriori

rules = apriori(transactions=transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2, max_length=2)
results = list(rules)
```
- The **`apyori`** package is used to implement the Eclat algorithm, though it is typically used for Apriori. Here, we apply it with constraints focusing on item pairs, making it function similarly to Eclat. The minimum support, confidence, and lift thresholds are set to filter meaningful itemsets.

### 5. Extracting Results
```python
def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    return list(zip(lhs, rhs, supports))

resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Product 1', 'Product 2', 'Support'])
resultsinDataFrame.nlargest(n = 10, columns = 'Support')
```
- A custom `inspect` function is used to extract the **left-hand side** (Product 1), **right-hand side** (Product 2), and **support** of each rule. This data is converted into a DataFrame for easy inspection of the top 10 results.

### 6. Sample Output
```plaintext
    Product 1            Product 2            Support
4   herb & pepper        ground beef          0.015998
7   whole wheat pasta    olive oil            0.007999
2   pasta                escalope             0.005866
1   mushroom cream sauce escalope             0.005733
5   tomato sauce         ground beef          0.005333
```
- The table displays the top product pairs with the highest support, indicating how frequently these pairs are bought together.

## Installation

1. Install required libraries:
```bash
pip install numpy pandas matplotlib apyori
```

2. Download the dataset and place it in the same directory as the script.

## Running the Project

1. Ensure that the dataset (`Market_Basket_Optimisation.csv`) is available.
2. Run the Python script to apply the Eclat algorithm and inspect the frequent product pairs.
