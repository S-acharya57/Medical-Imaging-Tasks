# Machine Learning TASK

High-Dimensional Feature Classification: A machine learning pipeline for classifying high-dimensional data using feature selection and ensemble models.

## Features

- Preprocessing & Feature Selection
  - infinite values `(inf)` were reduced to NaN, and then changed to 0 for easier computation
- Multiple Classifiers (Logistic Regression, SVM)
- Soft Voting Ensemble
- Performance Evaluation (with necessary metrics)

## Project Structure

- `predictions`: Has all prediction probabilities for each dataset (train, test, blind), summary of metrics for each model used.
- `log_reg.py`: Logistic Regression Model
- `random_forest_features.txt`: All important features
- `rf_feature_selection.py`: RandomForest to get Important Features
- `svm_model.py`: SVM Model
- `last_model.py`: Ensemble model
- `Methodology-Sajjan.pdf`: Description of methodologies used in this task, in report format.
