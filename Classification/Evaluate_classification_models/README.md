# Evaluating Classification Models

This project demonstrates the implementation and evaluation of various classification models on a dataset to predict outcomes. The models compared in this project include Decision Tree, K-Nearest Neighbors (K-NN), Kernel SVM, Logistic Regression, Naive Bayes, Random Forest, and Support Vector Machine (SVM). The evaluation metrics used to compare these models include the confusion matrix and accuracy scores.

## Overview

Classification is a supervised learning technique used to predict the categorical labels of given input data. This project evaluates the performance of multiple classifiers and compares their ability to accurately classify the data. 

## Dataset

The dataset used in this project is a medical dataset for predicting the class of tumors (benign or malignant). It contains various features describing the characteristics of cell samples, including clump thickness, uniformity of cell size, and uniformity of cell shape. The last column represents the class label, where `2` indicates benign tumors and `4` indicates malignant tumors.

Example data from the dataset:

| Sample Code Number | Clump Thickness | Uniformity of Cell Size | Uniformity of Cell Shape | Marginal Adhesion | Single Epithelial Cell Size | Bare Nuclei | Bland Chromatin | Normal Nucleoli | Mitoses | Class |
|-------------------|-----------------|-------------------------|--------------------------|------------------|----------------------------|-------------|-----------------|-----------------|---------|-------|
| 1000025           | 5               | 1                       | 1                        | 1                | 2                          | 1           | 3               | 1               | 1       | 2     |
| 1002945           | 5               | 4                       | 4                        | 5                | 7                          | 10          | 3               | 2               | 1       | 2     |
| 1015425           | 3               | 1                       | 1                        | 1                | 2                          | 2           | 3               | 1               | 1       | 2     |
| 1016277           | 6               | 8                       | 8                        | 1                | 3                          | 4           | 3               | 7               | 1       | 2     |
| 1017023           | 4               | 1                       | 1                        | 3                | 2                          | 1           | 3               | 1               | 1       | 2     |

The features include:

- **Clump Thickness**
- **Uniformity of Cell Size**
- **Uniformity of Cell Shape**
- **Marginal Adhesion**
- **Single Epithelial Cell Size**
- **Bare Nuclei**
- **Bland Chromatin**
- **Normal Nucleoli**
- **Mitoses**

The target column, **Class**, is a binary variable where:
- `2`: Benign
- `4`: Malignant

## Project Structure

- `classification_evaluation.py`: Python script that trains multiple classification models and evaluates them using confusion matrices and accuracy scores.
- `Position_Salaries.csv`: The dataset containing sample data.
- `README.md`: Documentation explaining the project.

## Requirements

To run this project, you'll need:

- Python 3.x
- Required libraries: `numpy`, `pandas`, `scikit-learn`, `matplotlib`

You can install the required libraries using pip:

```bash
pip install numpy pandas scikit-learn matplotlib
```

## Code Explanation

1. **Importing Libraries**: The script begins by importing necessary libraries such as `numpy`, `pandas`, and `scikit-learn` for data processing, model training, and evaluation.

2. **Loading the Dataset**: The dataset is loaded using `pandas`. Features (`X`) and the target variable (`y`) are extracted for model training.

3. **Model Training**:
   - Several classification models are created and trained on the dataset. These models include:
     - Decision Tree
     - K-Nearest Neighbors (K-NN)
     - Kernel SVM
     - Logistic Regression
     - Naive Bayes
     - Random Forest
     - Support Vector Machine (SVM)

4. **Model Evaluation**:
   - For each model, predictions are made on the test data.
   - The confusion matrix for each model is calculated using `confusion_matrix` from `scikit-learn`. The confusion matrix provides insights into the number of true positives, true negatives, false positives, and false negatives.
   - The accuracy score is computed using `accuracy_score`, which measures the proportion of correctly classified instances out of all instances.

5. **Confusion Matrices and Accuracy Scores**:
   Below are the confusion matrices and accuracy scores for each model:

   - **Decision Tree**:
     - Confusion Matrix: `[[103, 4], [3, 61]]`
     - Accuracy: `0.9590`
   
   - **K-Nearest Neighbors (K-NN)**:
     - Confusion Matrix: `[[103, 4], [5, 59]]`
     - Accuracy: `0.9474`

   - **Kernel SVM**:
     - Confusion Matrix: `[[102, 5], [3, 61]]`
     - Accuracy: `0.9532`
   
   - **Logistic Regression**:
     - Confusion Matrix: `[[103, 4], [5, 59]]`
     - Accuracy: `0.9474`
   
   - **Naive Bayes**:
     - Confusion Matrix: `[[99, 8], [2, 62]]`
     - Accuracy: `0.9415`

   - **Random Forest**:
     - Confusion Matrix: `[[102, 5], [6, 58]]`
     - Accuracy: `0.9357`
   
   - **Support Vector Machine (SVM)**:
     - Confusion Matrix: `[[102, 5], [5, 59]]`
     - Accuracy: `0.9415`

6. **Comparison of Models**:
   - **Best Performing Model**: The Decision Tree model has the highest accuracy (0.9590), closely followed by the Kernel SVM model (0.9532).
   - **Other Models**: K-NN, Logistic Regression, and SVM have similar accuracies, slightly below that of the best models. Naive Bayes and Random Forest performed the lowest in terms of accuracy, although the difference is relatively small.

## Usage

To run the project and evaluate the models, execute the Python script:

```bash
python classification_evaluation.py
```

The script will:

- Train each of the classification models on the dataset.
- Output the confusion matrix and accuracy score for each model.
- Provide a comparison of the performance of each classifier.

## Conclusion

This project demonstrates how different classification models can be evaluated based on their confusion matrix and accuracy score. While all models perform reasonably well, the Decision Tree model exhibits the highest accuracy in this case, making it the best choice for this particular dataset.

Understanding the strengths and weaknesses of each model helps in selecting the most appropriate classifier for a given problem, enabling more informed decision-making in real-world applications.
