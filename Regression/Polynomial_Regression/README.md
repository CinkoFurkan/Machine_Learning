# Polynomial Regression on Position Salaries

This project demonstrates the implementation of Polynomial Regression to predict salaries based on the position level in a company. The dataset contains information on various job positions and their corresponding salary levels. This project compares the performance of a simple linear regression model with a polynomial regression model to highlight the effectiveness of polynomial regression in capturing non-linear relationships.

## Overview

Polynomial Regression is an extension of Linear Regression that models the relationship between the independent variable and the dependent variable as an nth degree polynomial. This project compares Linear Regression and Polynomial Regression models to predict salaries based on position levels in a company.

## Dataset

The dataset used in this project is `Position_Salaries.csv`, which includes the following columns:

- **Position**: The job title within the company.
- **Level**: A numerical representation of the job level (independent variable).
- **Salary**: The salary corresponding to the job position (dependent variable).

Example data from `Position_Salaries.csv`:

| Position          | Level | Salary |
|-------------------|-------|--------|
| Business Analyst  | 1     | 45000  |
| Junior Consultant | 2     | 50000  |
| Senior Consultant | 3     | 60000  |
| Manager           | 4     | 80000  |

## Project Structure

- `Position_Salaries.csv`: The dataset containing position levels and salaries.
- `polynomial_regression.py`: Python script containing the code for training and evaluating both linear and polynomial regression models.
- `README.md`: Documentation file explaining the project.

## Requirements

To run this project, you'll need:

- Python 3.x
- Required libraries: `numpy`, `pandas`, `matplotlib`, `scikit-learn`

You can install the required libraries using pip:

```bash
pip install numpy pandas matplotlib scikit-learn
```

## Code Explanation

1. **Importing Libraries**: The script starts by importing necessary libraries such as `numpy`, `pandas`, `matplotlib`, and `scikit-learn` for data processing, visualization, and regression modeling.

2. **Loading the Dataset**: The dataset is loaded using `pandas`. The independent variable (`X`) representing position levels and the dependent variable (`y`) representing salaries are extracted.

3. **Linear Regression**:
   - A simple linear regression model is created using `LinearRegression` from `scikit-learn`.
   - The model is trained on the entire dataset, and a linear prediction is made.
   - A scatter plot of actual salaries versus position levels is generated with the linear regression line to visualize the fit.

4. **Polynomial Regression**:
   - A polynomial feature transformer is created using `PolynomialFeatures` with a specified degree (4 in this case) to capture non-linear relationships.
   - The transformed features are used to train another `LinearRegression` model.
   - The polynomial regression line is plotted over the scatter plot of actual salaries to compare its performance against the simple linear regression model.

5. **Model Predictions**:
   - Predictions are made using both the linear and polynomial regression models for a given level (e.g., 6.5).
   - The output demonstrates how polynomial regression better captures the non-linear relationship between position levels and salaries.

## Usage

To run the project, execute the Python script:

```bash
python polynomial_regression.py
```

The script will:

- Plot the actual salaries and the linear regression prediction line.
- Plot the actual salaries and the polynomial regression prediction line.
- Output predictions for a specific position level using both regression models.

## Conclusion

This project highlights the use of polynomial regression to capture non-linear relationships between position levels and salaries in a company. While linear regression can provide a basic approximation, polynomial regression demonstrates superior performance in this context by fitting the data more accurately.

By understanding the advantages of polynomial regression, businesses can make more informed decisions about salary structuring and career progression based on complex, non-linear patterns in the data.

