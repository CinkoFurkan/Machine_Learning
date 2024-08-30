# Data Preprocessing Tools

This project demonstrates various data preprocessing techniques used in machine learning pipelines. It covers steps such as handling missing data, encoding categorical variables, splitting the dataset into training and test sets, and feature scaling. These preprocessing steps are essential for preparing data before training machine learning models.

## Overview

Data preprocessing is a crucial step in the machine learning workflow, as it ensures that the data is clean, formatted, and ready for modeling. This project illustrates the common data preprocessing tasks using the `Data.csv` dataset.

## Dataset

The dataset used in this project is `Data.csv`, which includes the following columns:

- **Country**: Categorical data representing the country names.
- **Age**: Numerical data representing the age of individuals.
- **Salary**: Numerical data representing the salary of individuals.
- **Purchased**: Categorical data indicating whether a purchase was made (Yes/No).

Example data from `Data.csv`:

| Country | Age | Salary | Purchased |
|---------|-----|--------|-----------|
| France  | 44  | 72000  | No        |
| Spain   | 27  | 48000  | Yes       |
| Germany | 30  | 54000  | No        |
| Spain   | 38  | 61000  | No        |
| Germany | 40  | NaN    | Yes       |

## Project Structure

- `Data.csv`: The dataset containing categorical and numerical data.
- `data_preprocessing.py`: Python script containing the code for data preprocessing steps.
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

1. **Importing Libraries**: The necessary libraries (`numpy`, `pandas`, `scikit-learn`, etc.) are imported for data manipulation and preprocessing.

2. **Loading the Dataset**: The dataset is loaded using `pandas` from the `Data.csv` file. `X` contains the independent variables, and `y` contains the dependent variable.

3. **Handling Missing Data**: The missing values in the `Age` and `Salary` columns are replaced with the mean of the respective columns using `SimpleImputer` from `scikit-learn`.

4. **Encoding Categorical Data**:
   - **Independent Variable Encoding**: The categorical variable `Country` is encoded using `OneHotEncoder`, converting it into multiple binary columns.
   - **Dependent Variable Encoding**: The `Purchased` column is encoded into numerical format (0 and 1) using `LabelEncoder`.

5. **Splitting the Dataset**: The data is split into training and test sets using an 80-20 split ratio with `train_test_split` from `scikit-learn`.

6. **Feature Scaling**: To standardize the features, the numerical columns (`Age`, `Salary`) are scaled using `StandardScaler`. Feature scaling ensures that the model does not get biased by the varying ranges of the features.

## Usage

To run the project, execute the Python script:

```bash
python data_preprocessing.py
```

The script will output:

- The processed dataset after handling missing data.
- The encoded independent and dependent variables.
- The training and test sets.
- The scaled features for the training and test sets.

## Conclusion

This project provides a comprehensive overview of data preprocessing steps in machine learning, preparing the data for further modeling tasks. Proper preprocessing ensures that the data is in an optimal format for training machine learning models.
