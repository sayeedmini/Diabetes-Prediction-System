import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


print("Data Loading")

data = pd.read_csv('diabetes.csv')
print(f"Shape: {data.shape}")
print(data.head())


data = data.drop_duplicates()


cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_with_zeros:
    data[col] = data[col].replace(0, np.nan)


num_cols = data.select_dtypes(include=[np.number]).columns.tolist()
for col in num_cols:
    if col != "Outcome":
        lower = data[col].quantile(0.01)
        upper = data[col].quantile(0.99)
        data[col] = data[col].clip(lower, upper)


X = data.drop('Outcome', axis=1)
y = data['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(random_state=42, class_weight='balanced'))
])



print("\n")
print("Selected Algorithm: Random Forest Classifier.")
print("Justification: Random Forest selected because it is an ensemble method that handles non-linear relationships well, is robust to outliers, and reduces overfitting compared to a single Decision Tree.")

print("\n")
print("Cross-Validation Robustness")


cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='roc_auc')
print(f"CV ROC-AUC Scores: {cv_scores}")
print(f"Average ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")


print("\n")
print("Hyperparameter Tuning")

param_dist = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [None, 5, 10],
    'model__min_samples_split': [2, 5]
}

search = RandomizedSearchCV(pipeline, param_dist, n_iter=5, cv=5, scoring='roc_auc', random_state=42, n_jobs=-1)
search.fit(X_train, y_train)

print(f"Best Parameters Found: {search.best_params_}")
print(f"Best Score: {search.best_score_:.4f}")


best_model = search.best_estimator_


print("\n")
print("Evaluation on Test Set")

y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

joblib.dump(best_model, 'diabetes_pipeline.pkl')
print("\nModel saved to 'diabetes_pipeline.pkl'")
