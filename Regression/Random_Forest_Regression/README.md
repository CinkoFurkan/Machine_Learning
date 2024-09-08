# Random Forest Regression on Position Salaries

This project demonstrates the implementation of Random Forest Regression to predict salaries based on position levels within a company. The Random Forest model, which is an ensemble of multiple decision trees, is used to capture non-linear relationships between the job levels and salaries, resulting in robust and accurate predictions.

## Overview

Random Forest Regression is an ensemble learning technique that constructs multiple decision trees during training and outputs the mean prediction of the individual trees for regression tasks. This approach reduces overfitting and improves the model's generalization. In this project, Random Forest Regression is applied to model the relationship between job levels and salaries within a company.

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
| Country Manager   | 5     | 110000 |

## Project Structure

- `Position_Salaries.csv`: The dataset containing position levels and salaries.
- `random_forest_regression.py`: Python script containing the code for training and evaluating the Random Forest Regression model.
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

1. **Importing Libraries**: The script begins by importing essential libraries such as `numpy`, `pandas`, `matplotlib`, and `scikit-learn` for data processing, visualization, and regression modeling.

2. **Loading the Dataset**: The dataset is loaded using `pandas`. The independent variable (`X`) representing position levels and the dependent variable (`y`) representing salaries are extracted.

3. **Random Forest Regression**:
   - A Random Forest Regressor is created using `RandomForestRegressor` from `scikit-learn` with 10 estimators (trees).
   - The model is trained on the entire dataset, allowing it to capture complex patterns without overfitting.

4. **Prediction**:
   - The model is used to predict the salary for a specific position level (e.g., 6.5).
   - This prediction demonstrates how Random Forest Regression leverages multiple decision trees to capture non-linear patterns in the data.

5. **Visualization**:
   - The regression results are visualized using a high-resolution plot.
   - A grid of values for the independent variable (`X`) is created with small steps (0.01) to provide a smooth curve that illustrates the model's predictions over the data range.
   - A scatter plot of the original data points (`X`, `y`) is shown in red, while the Random Forest predictions are plotted in blue.

## Usage

To run the project, execute the Python script:

```bash
python random_forest_regression.py
```

The script will:

- Print the prediction for a specific position level (e.g., 6.5).
- Display a plot showing the Random Forest regression predictions against the actual data points.

## Conclusion

This project illustrates the application of Random Forest Regression for predicting salaries based on job levels within a company. By averaging the results of multiple decision trees, Random Forests provide a powerful tool for capturing non-linear relationships and complex patterns in the data, leading to more accurate and reliable predictions.

The use of Random Forest Regression allows companies to better understand how different job levels correspond to salaries, aiding in decision-making related to compensation and career development.

