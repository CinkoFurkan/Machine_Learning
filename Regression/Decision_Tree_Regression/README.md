# Decision Tree Regression on Position Salaries

This project demonstrates the use of Decision Tree Regression to predict salaries based on position levels within a company. The dataset contains information on various job positions and their corresponding salary levels. The Decision Tree Regressor model is used here due to its ability to capture non-linear relationships and create piecewise constant predictions.

## Overview

Decision Tree Regression is a non-linear regression technique that partitions the data into regions with similar target values. Each leaf node in the tree represents a predicted value, which is the average target value of the samples in that region. This project employs a Decision Tree Regressor to model and predict salaries based on job levels.

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
- `decision_tree_regression.ipynb`: Python script containing the code for training and evaluating the Decision Tree Regressor.
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

3. **Training the Decision Tree Regressor**:
   - A Decision Tree Regressor is created using `DecisionTreeRegressor` from `scikit-learn`.
   - The model is trained on the position levels and corresponding salaries.

4. **Making Predictions**:
   - The trained model is used to predict the salary for a specific position level (e.g., 6.5).

5. **Visualization**:
   - A visualization is created showing the decision tree's predictions. The scatter plot represents actual salaries, while the line plot shows the predicted values from the Decision Tree Regressor.

## Usage

To run the project, execute the Python script:

```bash
python decision_tree_regression.py
```

The script will:

- Train a Decision Tree Regressor using the dataset.
- Predict the salary for a specific position level (e.g., 6.5).
- Display a plot with actual salaries and model predictions.

## Conclusion

This project illustrates how Decision Tree Regression can be used to model and predict salaries based on job levels. The Decision Tree Regressor is particularly effective for capturing non-linear relationships and making piecewise constant predictions.

By leveraging Decision Tree Regression, organizations can gain insights into salary predictions based on position levels, aiding in better compensation planning and decision-making.
