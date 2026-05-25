import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import joblib

# =====================================
# LOAD DATASET
# =====================================

print("\nLoading Dataset...\n")

data = pd.read_csv("../dataset/creditcard.csv")

print("Dataset Loaded Successfully!")

# =====================================
# FEATURES & TARGET
# =====================================

X = data.drop("Class", axis=1)

y = data["Class"]

# =====================================
# TRAIN TEST SPLIT
# =====================================

print("\nSplitting Dataset...\n")

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42
)

# =====================================
# RANDOM FOREST MODEL
# =====================================

print("Training Random Forest Model...\n")

model = RandomForestClassifier(

    n_estimators=50,

    random_state=42
)

# Train Model
model.fit(X_train, y_train)

print("Model Training Completed!")

# =====================================
# PREDICTIONS
# =====================================

y_pred = model.predict(X_test)

# =====================================
# MODEL ACCURACY
# =====================================

accuracy = accuracy_score(y_test, y_pred)

print("\n=====================================")
print("MODEL ACCURACY")
print("=====================================")

print(f"\nAccuracy: {accuracy * 100:.2f}%")

# =====================================
# CLASSIFICATION REPORT
# =====================================

print("\n=====================================")
print("CLASSIFICATION REPORT")
print("=====================================\n")

print(classification_report(y_test, y_pred))

# =====================================
# CONFUSION MATRIX
# =====================================

print("\n=====================================")
print("CONFUSION MATRIX")
print("=====================================\n")

print(confusion_matrix(y_test, y_pred))

# =====================================
# SAVE TRAINED MODEL
# =====================================

joblib.dump(

    model,

    "../model/fraud_model.pkl"
)

print("\n=====================================")
print("MODEL SAVED SUCCESSFULLY!")
print("=====================================\n")