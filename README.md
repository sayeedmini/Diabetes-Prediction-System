# Diabetes Prediction System

## Project Overview
This project builds a Machine Learning pipeline to predict whether a patient has diabetes based on diagnostic measures (e.g., Glucose, BMI, Age). It meets all requirements for the ML Final Exam, including data preprocessing, pipeline creation, cross-validation, hyperparameter tuning, and deployment.

- **Dataset:** Pima Indians Diabetes Dataset
- **Model:** Random Forest Classifier
- **Deployment:** Hugging Face Spaces + Gradio

## Repository Structure
- `train.py`: Main script for training, evaluation, and saving the pipeline.
- `app.py`: Gradio web application script for inference.
- `requirements.txt`: List of dependencies.
- `diabetes_pipeline.pkl`: The saved trained model pipeline.

## 1. Data Preprocessing
As per Task 2, the following 5 distinct preprocessing steps were applied:
1.  **Duplicate Removal:** Dropped duplicate rows to ensure data quality.
2.  **Zero Handling:** Replaced invalid `0` values in columns like `Glucose`, `BloodPressure`, and `BMI` with `NaN`.
3.  **Outlier Clipping:** Capped features at the 1st and 99th percentiles to reduce noise.
4.  **Imputation:** Used `SimpleImputer(strategy='median')` within the pipeline to fill missing values.
5.  **Scaling:** Applied `StandardScaler` to normalize feature distributions.

## 2. Model Selection & Tuning
- **Primary Model:** Random Forest Classifier was selected for its robustness to outliers and ability to model non-linear relationships better than logistic regression.
- **Cross-Validation:** 5-Fold Stratified CV was used to report the average ROC-AUC score.
- **Hyperparameter Tuning:** `RandomizedSearchCV` was used to optimize `n_estimators`, `max_depth`, and `min_samples_split`.

## 3. Performance
The final model achieved the following metrics on the test set:
- **Accuracy:** ~76% (varies slightly by run)
- **ROC-AUC:** ~0.84

## 4. How to Run Locally
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
