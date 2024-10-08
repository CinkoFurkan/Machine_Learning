# Multiple Linear Regression on Startup Data

This project demonstrates the implementation of Multiple Linear Regression to predict the profit of startups based on their spending on various business activities and the state they are operating in. The project utilizes data from `50_Startups.csv` and follows a typical machine learning workflow including data preprocessing, model training, and evaluation.

## Overview

Multiple Linear Regression is a statistical technique that models the relationship between one dependent variable and two or more independent variables. This project uses multiple linear regression to predict the profit of startups based on features like R&D Spend, Administration Spend, Marketing Spend, and State.

## Dataset

The dataset used in this project is `50_Startups.csv`, which includes the following columns:

- **R&D Spend**: The amount of money spent on research and development.
- **Administration**: The amount of money spent on administration.
- **Marketing Spend**: The amount of money spent on marketing.
- **State**: The state where the startup operates (categorical variable).
- **Profit**: The profit generated by the startup (target variable).

Example data from `50_Startups.csv`:

| R&D Spend | Administration | Marketing Spend | State     | Profit   |
|-----------|----------------|-----------------|-----------|----------|
| 165349.20 | 136897.80      | 471784.10       | New York  | 192261.83 |
| 162597.70 | 151377.59      | 443898.53       | California| 191792.06 |
| 153441.51 | 101145.55      | 407934.54       | Florida   | 191050.39 |
| 144372.41 | 118671.85      | 383199.62       | New York  | 182901.99 |
| 142107.34 | 91391.77       | 366168.42       | Florida   | 166187.94 |

## Project Structure

- `50_Startups.csv`: The dataset containing numerical and categorical features.
- `multiple_linear_regression.py`: Python script with the code for training and evaluating the multiple linear regression model.
- `README.md`: Documentation file describing the project.

## Requirements

To run this project, you'll need:

- Python 3.x
- Required libraries: `numpy`, `pandas`, `matplotlib`, `scikit-learn`

You can install the required libraries using pip:

```bash
pip install numpy pandas matplotlib scikit-learn
```

## Code Explanation

1. **Importing Libraries**: The script begins by importing necessary libraries such as `numpy`, `pandas`, `matplotlib`, and `scikit-learn` for data manipulation, visualization, and machine learning.

2. **Loading the Dataset**: The dataset is loaded using `pandas`. The independent variables (`X`) and dependent variable (`y`) are extracted from the dataset.

3. **Encoding Categorical Data**: The categorical variable `State` is encoded into numerical format using `OneHotEncoder`. This step converts the categorical states into binary columns.

4. **Splitting the Dataset**: The dataset is split into training and test sets using an 80-20 split ratio with `train_test_split` from `scikit-learn`.

5. **Training the Model**: A Multiple Linear Regression model is created using `LinearRegression` from `scikit-learn` and is trained on the training data.

6. **Predicting Test Results**: The model predicts the profit for the test set and the results are compared against the actual values.

7. **Model Evaluation**:
   - The predicted and actual profits for the test set are printed side by side for comparison.
   - A single prediction is made for a hypothetical startup with specified feature values.
   - The model's coefficients and intercept are printed to understand the impact of each feature on the target variable.

## Usage

To run the project, execute the Python script:

```bash
python multiple_linear_regression.py
```

The script will output:

- The encoded and preprocessed dataset.
- The predicted vs. actual values for the test set.
- A specific prediction for a new set of features.
- The coefficients and intercept of the trained model.

## Conclusion

This project provides an example of using multiple linear regression to predict business outcomes based on various inputs. Understanding the relationships between different business expenditures and profit can help in strategic decision-making and optimizing resource allocation.
