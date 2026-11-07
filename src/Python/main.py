from pathlib import Path
import pandas as pd

data_dir = Path("/app/data")

print("Loading Titanic datasets...", flush=True)
# Q13: Load and preview the Titanic datasets
train = pd.read_csv(data_dir / "train.csv")
test = pd.read_csv(data_dir / "test.csv")

print(f"Training data: {train.shape[0]} rows, {train.shape[1]} columns")
print(f"Test data: {test.shape[0]} rows, {test.shape[1]} columns")
print("\nTraining preview:")
print(train.head())

# Q14: Inspect missing values and survival distribution
print("\nQ14: Missing values in training data:")
print(train.isna().sum())

print("\nQ14: Missing values in test data:")
print(test.isna().sum())

print("\nQ14: Survival counts (0 = died, 1 = survived):")
print(train["Survived"].value_counts())

# Q14: Prepare features
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]

X_train = train[features].copy()
X_test = test[features].copy()
y_train = train["Survived"]

for column in ["Age", "Fare"]:
    median_value = X_train[column].median()
    X_train[column] = X_train[column].fillna(median_value)
    X_test[column] = X_test[column].fillna(median_value)
    print(f"\nQ14: Filled missing {column} with training median: {median_value}")

X_train["Sex"] = X_train["Sex"].map({"male": 0, "female": 1})
X_test["Sex"] = X_test["Sex"].map({"male": 0, "female": 1})

print("\nQ14: Encoded Sex as male=0, female=1.")
print("Q14: Selected features:", features)
print("Q14: Excluded PassengerId, Name, Ticket, Cabin, and Embarked from model features.")
print("Q14: Remaining missing feature values:")
print("Training:", X_train.isna().sum().sum())
print("Test:", X_test.isna().sum().sum())

# Q15: Train a logistic regression model
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, random_state=42)
)

print("\nQ15: Standardizing features and training logistic regression.")
model.fit(X_train, y_train)
print("Q15: Model training completed.")

# Q16: Measure training accuracy
from sklearn.metrics import accuracy_score

train_predictions = model.predict(X_train)
train_accuracy = accuracy_score(y_train, train_predictions)

print(f"\nQ16: Training accuracy: {train_accuracy:.2%}")

# Q17: Predict survival on the test dataset
test_predictions = model.predict(X_test)

prediction_results = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": test_predictions
})

print(f"\nQ17: Generated predictions for {len(prediction_results)} passengers.")
print(prediction_results.to_string(index=False))

# Q18: Explain why test accuracy cannot be calculated locally
print("\nQ18: Test accuracy is unavailable because test.csv has no Survived labels.")