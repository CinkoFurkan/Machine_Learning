# Regression Models on Energy Output Prediction

This project involves building and evaluating various regression models to predict the energy output (`PE`) of a system based on multiple environmental and operational parameters. The regression models implemented include Multiple Linear Regression, Polynomial Regression, Support Vector Regression, Decision Tree Regression, and Random Forest Regression. The goal is to determine which model best captures the relationship between the independent variables and the target variable, energy output.

## Overview

Regression analysis is used to model the relationship between the dependent variable (energy output) and multiple independent variables (temperature, exhaust vacuum, ambient pressure, and relative humidity). By comparing various regression models, we aim to find the most accurate model for predicting energy output, which can be crucial for optimizing performance and efficiency in industrial applications.

## Dataset

The dataset `Data.csv` includes the following columns:

- **AT (Temperature)**: Ambient temperature in degrees Celsius.
- **V (Exhaust Vacuum)**: Exhaust vacuum in cm Hg.
- **AP (Ambient Pressure)**: Ambient pressure in millibars.
- **RH (Relative Humidity)**: Relative humidity in percentage.
- **PE (Energy Output)**: Energy output in megawatts.

Example data from `Data.csv`:

| AT   | V    | AP      | RH    | PE     |
|------|------|---------|-------|--------|
| 14.96| 41.76| 1024.07 | 73.17 | 463.26 |
| 25.18| 62.96| 1020.04 | 59.08 | 444.37 |
| 5.11 | 39.40| 1012.16 | 92.14 | 488.56 |
| 20.86| 57.32| 1010.24 | 76.64 | 446.48 |
| 10.82| 37.50| 1009.23 | 96.62 | 473.90 |

The dataset consists of 9,568 entries, providing a comprehensive set of data points for training and evaluating the models.

## Models Implemented

The following regression models are applied:

1. **Multiple Linear Regression**: Models the relationship between multiple independent variables and the dependent variable using a linear approach.
2. **Polynomial Regression**: Extends linear regression by considering polynomial terms, allowing it to capture non-linear relationships.
3. **Support Vector Regression (SVR)**: Uses kernel functions to map input features into higher-dimensional spaces for capturing non-linear patterns.
4. **Decision Tree Regression**: Uses a tree-like structure to split the data based on feature values, capturing complex patterns.
5. **Random Forest Regression**: An ensemble of decision trees that averages their predictions, enhancing accuracy and reducing overfitting.

## R² Scores

The R² scores for each regression model, indicating the proportion of variance explained by the model, are as follows:

- **Multiple Linear Regression**: 0.932
- **Polynomial Regression**: 0.945
- **Support Vector Regression**: 0.948
- **Decision Tree Regression**: 0.922
- **Random Forest Regression**: 0.961

## Project Structure

- `multiple_linear_regression.ipynb`: Jupyter Notebook implementing Multiple Linear Regression.
- `polynomial_regression.ipynb`: Jupyter Notebook implementing Polynomial Regression.
- `support_vector_regression.ipynb`: Jupyter Notebook implementing Support Vector Regression.
- `decision_tree_regression.ipynb`: Jupyter Notebook implementing Decision Tree Regression.
- `random_forest_regression.ipynb`: Jupyter Notebook implementing Random Forest Regression.
- `Data.csv`: The dataset containing environmental and operational parameters and energy output.
- `README.md`: Documentation file explaining the project.

## Requirements

To run the notebooks in this project, you'll need:

- Python 3.x
- Jupyter Notebook or JupyterLab
- Required libraries: `numpy`, `pandas`, `matplotlib`, `scikit-learn`

Install the required libraries using pip:

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

## Usage

1. Clone the repository and navigate to the project directory.

2. Open Jupyter Notebook or JupyterLab:

    ```bash
    jupyter notebook
    ```

3. Open the desired notebook (`.ipynb` file) from the Jupyter interface.

4. Run the cells in the notebook sequentially to:

   - Load and preprocess the dataset.
   - Train the regression model.
   - Evaluate the model using the R² score.
   - Visualize the predictions against the actual data points.

## Conclusion

This project explores multiple regression techniques to predict energy output based on various environmental factors. Random Forest Regression achieved the highest R² score of 0.961, indicating the best performance among the models tested. These results underscore the importance of selecting the appropriate model to capture the underlying patterns in the data, especially for non-linear relationships in complex systems.
