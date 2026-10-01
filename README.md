# Bank Marketing Classification

A Machine Learning classification project for predicting whether a
customer will respond positively to a bank marketing campaign.

## Project Overview

The goal of this project is to compare several classification algorithms
and improve the detection of the positive class in an imbalanced
dataset.

The project covers:

-   Data exploration and preprocessing
-   Checking class distribution
-   Encoding categorical features
-   Train/test splitting with stratification
-   Feature scaling
-   Training multiple classification models
-   Evaluation using classification metrics
-   Handling class imbalance with SMOTE
-   Comparing model performance before and after SMOTE
-   Hyperparameter tuning with `GridSearchCV`

## Target

The target variable is binary:

-   `0` --- negative class
-   `1` --- positive class

The dataset is imbalanced, with the positive class representing a much
smaller portion of the observations.

Because of this imbalance, accuracy alone is not enough to judge model
performance.

## Data Preparation

The workflow includes:

1.  Loading the dataset
2.  Inspecting the data
3.  Separating features and target
4.  Splitting the data using `stratify=y`
5.  Standardizing the features where appropriate
6.  Applying SMOTE only to the training data for the imbalance
    experiment

### Why Stratification?

`stratify=y` keeps approximately the same class distribution in both the
training and test sets.

### Why SMOTE?

SMOTE (Synthetic Minority Over-sampling Technique) creates synthetic
samples for the minority class.

Importantly, SMOTE is applied to the training data only. The test set
remains untouched so that evaluation represents performance on the
original data distribution.

## Models

The project compares several classification algorithms:

-   Logistic Regression
-   K-Nearest Neighbors (KNN)
-   Decision Tree
-   Random Forest
-   Gradient Boosting
-   AdaBoost
-   XGBoost

Tree-based models do not require feature scaling, while models such as
Logistic Regression and KNN benefit from standardized features.

## Evaluation Metrics

The main metrics are:

-   **Accuracy** --- proportion of all predictions that are correct.
-   **Precision** --- among predicted positive cases, how many are
    actually positive.
-   **Recall** --- among actual positive cases, how many are detected.
-   **F1 Score** --- harmonic mean of precision and recall.

Because the dataset is imbalanced, **Recall and F1 Score are
particularly important** for evaluating the positive class.

## Baseline Results

Before applying SMOTE, the models show high overall accuracy, but
performance on the minority class is more limited.

For example, the initial XGBoost model achieved approximately:

-   Accuracy: `0.904`
-   Precision: `0.630`
-   Recall: `0.489`
-   F1 Score: `0.551`

This demonstrates why accuracy by itself can be misleading for an
imbalanced classification problem.

## SMOTE Results

After applying SMOTE to the training data, some tree-based models
achieved substantially higher recall for the positive class.

The strongest model in the current experiment was **Gradient Boosting**
when prioritizing minority-class recall and F1:

-   Accuracy: approximately `0.880`
-   Recall: approximately `0.747`
-   Precision: approximately `0.502`
-   F1 Score: approximately `0.601`

This represents a trade-off: recall increased considerably, while
precision and overall accuracy decreased.

## Hyperparameter Tuning

Gradient Boosting was selected for further tuning because it provided
the strongest balance for the project's main objective.

`GridSearchCV` is used to search combinations of parameters such as:

-   `n_estimators`
-   `learning_rate`
-   `max_depth`
-   `subsample`

The search uses cross-validation to compare different parameter
combinations.

The final model is selected using the chosen scoring metric and then
evaluated on the untouched test set.

## Important Note About SMOTE and Grid Search

For the learning version of this project, the workflow is:

``` text
Training data
     ↓
Scaling
     ↓
SMOTE
     ↓
GridSearchCV
     ↓
Best Gradient Boosting model
     ↓
Original test data
```

For a more production-ready implementation, SMOTE should be placed
inside an `imblearn` Pipeline so that oversampling happens separately
inside each cross-validation training fold. This avoids synthetic-data
leakage between CV folds.

## Project Structure

``` text
Bank-Marketing-Classification/
├── bank_marketing.ipynb
├── dataset.csv
├── README.md
├── requirements.txt
└── results/
    └── confusion_matrix.png
```
## Best Params
Best Parameters: Learning_rate:0.05 ,Max_depth:3 ,n_estimators:200, subsample:1
Accuracy: 88%
Recall: 74%
Precision: 52%
F1: 61%

## Model Results

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.899 | 0.654 | 0.342 | 0.449 |
| KNN | 0.893 | 0.604 | 0.335 | 0.431 |
| Decision Tree | 0.861 | 0.439 | 0.555 | 0.490 |
| Random Forest | 0.894 | 0.559 | 0.588 | 0.573 |
| Gradient Boosting | 0.880 | 0.502 | 0.747 | 0.601 |
| AdaBoost | 0.844 | 0.416 | 0.727 | 0.530 |
| XGBoost | 0.902 | 0.600 | 0.558 | 0.578 |

## Libraries

Main Python libraries used:

``` text
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
xgboost
```

## What I Learned

This project was used to practice a complete classification workflow:

1.  Understanding class imbalance
2.  Preparing categorical and numerical data
3.  Using stratified train/test splitting
4.  Understanding feature scaling
5.  Comparing several classification algorithms
6.  Evaluating precision, recall and F1
7.  Understanding why accuracy can be misleading
8.  Using SMOTE for minority-class oversampling
9.  Comparing models before and after balancing
10. Using GridSearchCV for hyperparameter tuning

## Author

This project was implemented as part of my Machine Learning portfolio
and learning journey.
