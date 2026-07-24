import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder

print("=" * 60)
print("      CREDIT CARD FRAUD DETECTION")
print("=" * 60)

# ----------------------------
# Load Dataset
# ----------------------------
print("\nLoading datasets...")

train_df = pd.read_csv("fraudTrain.csv")
test_df = pd.read_csv("fraudTest.csv")

print("Datasets Loaded Successfully!")

# ----------------------------
# Use only a sample for faster training
# ----------------------------
train_df = train_df.sample(n=100000, random_state=42)
test_df = test_df.sample(n=30000, random_state=42)

print("\nSampled Dataset")
print("Training Shape :", train_df.shape)
print("Testing Shape  :", test_df.shape)

# ----------------------------
# Remove unnecessary columns
# ----------------------------
drop_columns = [
    "Unnamed: 0",
    "trans_date_trans_time",
    "cc_num",
    "first",
    "last",
    "street",
    "city",
    "state",
    "zip",
    "dob",
    "trans_num",
]

train_df.drop(columns=drop_columns, inplace=True, errors="ignore")
test_df.drop(columns=drop_columns, inplace=True, errors="ignore")

# ----------------------------
# Encode categorical columns
# ----------------------------
combined = pd.concat([train_df, test_df], ignore_index=True)

categorical_columns = combined.select_dtypes(include=["object", "string"]).columns

encoder = LabelEncoder()

for column in categorical_columns:
    combined[column] = encoder.fit_transform(combined[column].astype(str))

train_processed = combined.iloc[: len(train_df)]
test_processed = combined.iloc[len(train_df):]

# ----------------------------
# Split Features and Target
# ----------------------------
X_train = train_processed.drop("is_fraud", axis=1)
y_train = train_processed["is_fraud"]

X_test = test_processed.drop("is_fraud", axis=1)
y_test = test_processed["is_fraud"]

# ----------------------------
# Train Model
# ----------------------------
print("\nTraining Random Forest Model...")

model = RandomForestClassifier(
    n_estimators=50,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)

print("Model Trained Successfully!")

# ----------------------------
# Prediction
# ----------------------------
print("\nMaking Predictions...")

y_pred = model.predict(X_test)

# ----------------------------
# Results
# ----------------------------
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

print(f"\nAccuracy : {accuracy:.4f}")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:\n")
print(cm)

# ----------------------------
# Plot Confusion Matrix
# ----------------------------
plt.figure(figsize=(6, 5))
plt.imshow(cm)

plt.title("Confusion Matrix")
plt.colorbar()

plt.xticks([0, 1], ["Not Fraud", "Fraud"])
plt.yticks([0, 1], ["Not Fraud", "Fraud"])

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.show()

# ----------------------------
# Interactive Prediction
# ----------------------------
print("\n" + "=" * 60)
print("Prediction Summary")
print("=" * 60)

fraud = (y_pred == 1).sum()
normal = (y_pred == 0).sum()

print(f"Fraud Transactions Detected     : {fraud}")
print(f"Normal Transactions Detected    : {normal}")

print("\nProject Completed Successfully!")