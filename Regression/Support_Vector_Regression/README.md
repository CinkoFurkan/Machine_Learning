# Support Vector Regression (SVR) on Position Salaries

This project demonstrates the implementation of Support Vector Regression (SVR) to predict salaries based on position levels in a company. The dataset contains information on various job positions and their corresponding salary levels. The SVR model is used here because it can effectively handle non-linear relationships by using kernel functions.

## Overview

Support Vector Regression (SVR) is a type of Support Vector Machine (SVM) that supports both linear and non-linear regression tasks. It uses kernel functions to project data into higher dimensions, making it capable of capturing complex patterns. This project employs SVR with an RBF (Radial Basis Function) kernel to predict salaries based on job levels in a company.

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
- `svr_regression.py`: Python script containing the code for training and evaluating the SVR model.
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

3. **Data Reshaping**:
   - The dependent variable `y` is reshaped into a column vector to fit the SVR model.

4. **Feature Scaling**:
   - Feature scaling is crucial for SVR as it is sensitive to the scale of input features. 
   - Both the independent variable (`X`) and the dependent variable (`y`) are scaled using `StandardScaler`.

5. **Support Vector Regression**:
   - An SVR model with an RBF (Radial Basis Function) kernel is created using `SVR` from `scikit-learn`.
   - The model is trained on the scaled features and target variables.
   - A prediction is made for a specific level (e.g., 6.5) by transforming the input using the same scaler and then inversely transforming the prediction to get the original scale.

6. **Model Predictions**:
   - The prediction for a specific position level (e.g., 6.5) is performed using the trained SVR model.
   - The result is scaled back to the original salary range using the inverse transform method.

## Usage

To run the project, execute the Python script:

```bash
python svr_regression.py
```

The script will:

- Print the scaled input features and target values.
- Train an SVR model using the scaled data.
- Output a salary prediction for a specific position level (e.g., 6.5) using the SVR model.

## Conclusion

This project demonstrates how Support Vector Regression can be used to predict salaries based on job levels in a company. SVR is particularly useful for capturing complex, non-linear relationships that might not be easily modeled by traditional linear regression techniques.

By using SVR, businesses can make more accurate predictions about salaries based on job level data, enabling better decision-making around compensation and career progression.

