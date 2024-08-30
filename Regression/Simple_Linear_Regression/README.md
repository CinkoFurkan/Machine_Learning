# Salary Prediction Using Linear Regression

This project demonstrates a simple linear regression model to predict employee salaries based on their years of experience. The model is trained and tested using the `Salary_Data.csv` dataset.

## Overview

The goal of this project is to create a linear regression model that can predict the salary of an employee given their years of experience. The dataset consists of two columns: `YearsExperience` and `Salary`. We use a supervised machine learning approach, training the model on a subset of the data and testing it on the remaining data to evaluate its performance.

## Dataset

The dataset used in this project is `Salary_Data.csv`, which includes the following columns:

- **YearsExperience**: The number of years of experience an employee has.
- **Salary**: The corresponding salary of the employee.

Example data from `Salary_Data.csv`:

| YearsExperience | Salary   |
|-----------------|----------|
| 1.1             | 39343.00 |
| 1.3             | 46205.00 |
| 1.5             | 37731.00 |
| 2.0             | 43525.00 |
| 2.2             | 39891.00 |
| 2.9             | 56642.00 |

## Project Structure

- `Salary_Data.csv`: The dataset containing years of experience and corresponding salaries.
- `linear_regression.py`: Python script containing the code to train the linear regression model and make predictions.
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

1. **Importing Libraries**: The project starts by importing necessary libraries such as `numpy`, `pandas`, `matplotlib`, and `scikit-learn`.

2. **Loading the Dataset**: The dataset is loaded using `pandas` from the `Salary_Data.csv` file.

3. **Splitting the Data**: The data is split into training and test sets using an 80-20 split ratio.

4. **Training the Model**: A linear regression model is created and trained using the training data (`X_train`, `y_train`).

5. **Making Predictions**: The model makes predictions on the test set (`X_test`) and new data points.

6. **Visualization**:
   - The first plot shows the training data and the fitted regression line.
   - The second plot shows the test data and the regression line based on the training data, illustrating the model's performance on unseen data.

7. **Prediction for a Specific Value**: The code predicts the salary for an employee with 12 years of experience.

8. **Model Coefficients**: The slope (`coef_`) and intercept (`intercept_`) of the regression line are printed.

## Usage

To run the project, simply execute the Python script:

```bash
python linear_regression.py
```

The script will output:

- Two plots: One showing the regression line fit on the training set, and another on the test set.
- A predicted salary for 12 years of experience.
- The coefficients (slope) and intercept of the regression model.

## Results

- The model fits a line to the data points, illustrating the relationship between years of experience and salary.
- Predictions are made on the test set to evaluate the model's accuracy.

## Conclusion

This project provides a basic example of using linear regression for predictive modeling. It demonstrates how to train a model, visualize its performance, and make predictions using a simple dataset.
