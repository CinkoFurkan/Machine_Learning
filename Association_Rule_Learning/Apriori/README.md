# Market Basket Analysis using Apriori Algorithm

## Overview

This project implements the **Apriori algorithm** to identify associations between items purchased together in a retail store. The analysis is based on the **Market Basket Optimization** dataset, which contains transactions of various products bought together by customers. The Apriori algorithm helps discover frequent itemsets and generate rules that can help retailers optimize product placement and promotions.

## Dataset

The dataset used for this project is `"Market_Basket_Optimisation.csv"`, which consists of **7,501 transactions**. Each transaction contains items purchased by a customer during a shopping trip. The dataset is structured with each row representing a single transaction and each column containing an item bought in that transaction.

## Steps in the Code

### 1. Importing Necessary Libraries
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from apyori import apriori
```
- **Numpy**: Used for numerical computations.
- **Pandas**: Used for data manipulation.
- **Matplotlib**: Used for creating visualizations (if needed).
- **Apyori**: Library that implements the Apriori algorithm for association rule mining.

### 2. Loading the Dataset
```python
dataset = pd.read_csv("Market_Basket_Optimisation.csv", header=None)
```
- The dataset is loaded into a Pandas DataFrame. Since there is no header, we specify `header=None`.

### 3. Preprocessing the Data
```python
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])
```
- A list of transactions is created where each transaction is a list of items bought by a customer. 
- Non-purchased items are represented as `nan`, but will be ignored in the Apriori analysis.

### 4. Applying Apriori Algorithm
```python
rules = apriori(transactions = transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2, max_length=2)
results = list(rules)
```
- **`min_support=0.003`**: An item must appear in at least 0.3% of transactions to be considered frequent.
- **`min_confidence=0.2`**: The rule must hold true at least 20% of the time.
- **`min_lift=3`**: Rules with a lift greater than 3 are considered strong.
- **`min_length=2`**: Only consider rules with at least 2 items.
- **`max_length=2`**: Consider rules with at most 2 items.

### 5. Inspecting the Results
```python
def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))

resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])
```
- This function organizes the results into a readable format, displaying the left-hand side (LHS) and right-hand side (RHS) of the rule, along with support, confidence, and lift values.

### 6. Displaying the Top 10 Rules by Lift
```python
top_rules = resultsinDataFrame.nlargest(n = 10, columns = 'Lift')
top_rules
```
- The top 10 association rules are selected based on the highest lift values, which indicate the strongest associations between items.

### Sample Output:

| Left Hand Side     | Right Hand Side | Support | Confidence | Lift    |
|--------------------|-----------------|---------|------------|---------|
| frozen vegetables  | milk            | 0.00307 | 0.38333    | 7.98718 |
| olive oil          | milk            | 0.00333 | 0.29412    | 6.12827 |
| whole wheat pasta  | olive oil       | 0.00387 | 0.40278    | 6.11586 |
| herb & pepper      | ground beef     | 0.01600 | 0.32345    | 3.29199 |
| light cream        | chicken         | 0.00453 | 0.29060    | 4.84395 |

## Installation

1. Install required libraries:
```bash
pip install numpy pandas matplotlib apyori
```

2. Download the dataset and place it in the same directory as the script.

## Running the Project

1. Ensure that the dataset (`Market_Basket_Optimisation.csv`) is available.
2. Run the Python script to generate association rules and inspect the results. The output will display the top association rules in a tabular format.

## Conclusion

The Apriori algorithm helps identify strong associations between products, which can assist in strategies like product bundling and store layout optimization. The discovered rules can be used to better understand customer buying patterns and increase sales through cross-promotions.
