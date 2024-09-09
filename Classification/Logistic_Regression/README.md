# Logistic Regression for Social Network Ads

This project demonstrates the implementation of a Logistic Regression model to predict user responses to social network ads based on their age and estimated salary. The dataset is used to train and evaluate the model, which aims to classify users into categories based on their likelihood of responding to an ad.

## Overview

Logistic Regression is a classification algorithm used to predict the probability of a binary outcome based on one or more predictor variables. In this project, we use Logistic Regression to classify users as either responding to the ad or not based on their age and estimated salary.

## Dataset

The dataset `Social_Network_Ads.csv` contains the following columns:

- **Age**: The age of the user.
- **EstimatedSalary**: The estimated salary of the user.
- **Purchased**: Binary outcome indicating whether the user responded to the ad (1) or not (0).

Example data from `Social_Network_Ads.csv`:

| Age | EstimatedSalary | Purchased |
|-----|-----------------|-----------|
| 19  | 19000           | 0         |
| 35  | 20000           | 0         |
| 26  | 43000           | 0         |
| 27  | 57000           | 0         |
| 32  | 150000          | 1         |

## Project Structure

- `Social_Network_Ads.csv`: The dataset containing user information and ad response.
- `logistic_regression.ipynb`: Jupyter Notebook containing the code for Logistic Regression.
- `README.md`: Documentation file explaining the project.

## Requirements

To run the notebook in this project, you'll need:

- Python 3.x
- Jupyter Notebook or JupyterLab
- Required libraries: `numpy`, `pandas`, `matplotlib`, `scikit-learn`

You can install the required libraries using pip:

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

## Code Explanation

1. **Importing Libraries**: The script starts by importing necessary libraries such as `numpy`, `pandas`, `matplotlib`, and `scikit-learn` for data processing, visualization, and classification.

2. **Loading the Dataset**: The dataset is loaded using `pandas`. The independent variables (`X`) are age and estimated salary, while the dependent variable (`y`) is the ad response.

3. **Splitting the Dataset**: The dataset is split into training and testing sets using `train_test_split` from `scikit-learn`. 25% of the data is used for testing, and 75% is used for training.

4. **Feature Scaling**: The feature values are standardized using `StandardScaler` to improve the performance of the Logistic Regression model.

5. **Training the Model**:
   - A Logistic Regression classifier is created using `LogisticRegression` from `scikit-learn`.
   - The model is trained on the scaled training data.

6. **Prediction**:
   - The model predicts the response for a new user with age 30 and estimated salary 87000.
   - Predictions on the test set are compared with the actual test labels.

7. **Evaluation**:
   - A confusion matrix is computed to evaluate the performance of the model.
   - The accuracy score is calculated to measure the proportion of correctly classified instances.

## Usage

To run the notebook, execute the following command to open Jupyter Notebook:

```bash
jupyter notebook
```

Open the `logistic_regression.ipynb` file from the Jupyter interface and run the cells sequentially. The notebook will:

- Load and preprocess the dataset.
- Train the Logistic Regression model.
- Make predictions and evaluate the model using a confusion matrix and accuracy score.

## Conclusion

This project demonstrates the application of Logistic Regression for binary classification tasks. The model successfully predicts user responses to social network ads based on age and estimated salary. Evaluation metrics such as confusion matrix and accuracy score provide insights into the model's performance.

By using Logistic Regression, businesses can make data-driven decisions on targeting users for ads, optimizing marketing strategies, and improving campaign effectiveness.

