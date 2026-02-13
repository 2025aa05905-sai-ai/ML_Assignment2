<<<<<<< HEAD
# ML_Assignment2
=======
Machine Learning Classification Models Comparison
## Problem Statement
The objective of this project is to implement and compare multiple machine learning classification models on a medical dataset for predicting diabetes risk. The project also includes building and deploying an interactive Streamlit web application to evaluate model performance using test data.

The aim is to understand how different machine learning models perform on the same dataset and evaluate them using multiple performance metrics.

## Dataset Description

**Dataset:** Diabetes Health Indicators Dataset

**Source:** Kaggle

**Description:**
This is a medical dataset containing health-related indicators used to predict whether a person has diabetes or not, making this a binary classification problem.

Dataset Characteristics:

Type: Binary Classification

Number of features: 21 medical features

Number of instances: ~67000 records

Target variable: Diabetes_binary (0 = No diabetes, 1 = Diabetes)

Missing Values: None


## Models Used 

The following classification models were implemented:

Logistic Regression

Decision Tree Classifier

K-Nearest Neighbors (KNN)

Naive Bayes (Gaussian)

Random Forest (Ensemble)

XGBoost (Ensemble)

## Evaluation Metrics Used

Accuracy

Precision

Recall

F1 Score

AUC Score

Matthews Correlation Coefficient (MCC)

### Comparison Table with Evaluation Metrics

| ML Model            | Accuracy | AUC   | Precision | Recall | F1 Score | MCC   |
| ------------------- | -------- | ----- | --------- | ------ | -------- | ----- |
| Logistic Regression | 0.737    | 0.737 | 0.733     | 0.758  | 0.746    | 0.475 |
| Decision Tree       | 0.640    | 0.640 | 0.646     | 0.641  | 0.644    | 0.280 |
| KNN                 | 0.689    | 0.689 | 0.686     | 0.713  | 0.699    | 0.378 |
| Naive Bayes         | 0.714    | 0.714 | 0.716     | 0.724  | 0.720    | 0.428 |
| Random Forest       | 0.732    | 0.731 | 0.719     | 0.773  | 0.745    | 0.464 |
| XGBoost             | 0.744    | 0.743 | 0.729     | 0.787  | 0.757    | 0.489 |

## Observations about Model Performance

| ML Model            | Observation about model performance                                                     |
| ------------------- | --------------------------------------------------------------------------------------- |
| Logistic Regression | Logistic Regression achieved 73.7% accuracy with 75.8% Precall. Provided good baseline performance and stable results across metrics.                   |
| Decision Tree       | Decision Trees achieved 64% accuracy, 64.1% Recall and 28% MCC. Showed lower accuracy and MCC due to overfitting and limited generalization.            |
| KNN                 | KNN achieved 68.9% accuracy, 71.3% Recall. Moderate performance but computationally slower with larger dataset.                    |
| Naive Bayes         | Naive Bayes achieved 71.4% accuracy, 72.4% Recall. Fast training and decent performance but slightly lower accuracy than ensemble models.  |
| Random Forest       | Random Forest achieved 73.2% accuracy and 77.3% Recall. High accuracy and recall with strong generalization. Performs well on medical datasets like the current dataset.  |
| XGBoost             | XGBoost achieved 74.4% accuracy and 78.7% Recall. Best overall performance across most metrics with highest accuracy, recall, F1 and MCC. |

## Usage

### Streamlit Application Features

The deployed Streamlit application includes:

1. Test dataset download and upload option (CSV)

2. Model selection dropdown

3. Display of all evaluation metrics

4. Confusion matrix visualization

>>>>>>> 9799970 (ML Assignment 2)
