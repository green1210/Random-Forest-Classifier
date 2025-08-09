import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

df = pd.read_csv("churn-bigml-20.csv")
print("Initial shape:", df.shape)

df = df.drop(columns=[col for col in ["State", "Area code", "Phone"] if col in df.columns])

df['Churn'] = df['Churn'].astype(str).str.strip().str.lower()
df = df[df['Churn'].isin(['true.', 'false.', 'true', 'false'])]
df['Churn'] = df['Churn'].map({'true.': 1, 'false.': 0, 'true': 1, 'false': 0})

df['International plan'] = df['International plan'].map({'yes': 1, 'no': 0})
df['Voice mail plan'] = df['Voice mail plan'].map({'yes': 1, 'no': 0})

X = df.drop("Churn", axis=1)
y = df["Churn"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)

best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

cv_scores = cross_val_score(best_rf, X_scaled, y, cv=5, scoring='f1')
print("\nCross-validated F1-score:", np.mean(cv_scores))

importances = best_rf.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show()
