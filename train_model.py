from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

# ==========================================
# 1. Load extracted audio features
# ==========================================
PROJECT_PATH = Path(__file__).resolve().parent
CSV_PATH = PROJECT_PATH / "data" / "audio_features.csv"
MODEL_PATH = PROJECT_PATH / "models" / "music_genre_model.joblib"

df = pd.read_csv(CSV_PATH)
print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# ==========================================
# 2. Prepare features and target
# ==========================================
X = df.drop(columns=["filename", "genre"])
y = df["genre"]

X = X.apply(pd.to_numeric, errors="coerce")
X = X.fillna(X.median())

# ==========================================
# 3. Split into training and testing sets
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ==========================================
# 4. Train Random Forest classifier
# ==========================================
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)
print("\nTraining model...")
model.fit(X_train, y_train)

# ==========================================
# 5. Evaluate the model & Extract Metrics
# ==========================================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
labels = sorted(y.unique())

# Get terminal report
print(f"\nTest Accuracy: {accuracy:.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Save report as dictionary for Streamlit
report_dict = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
cm = confusion_matrix(y_test, y_pred, labels=labels)

# ==========================================
# 6. Save the trained model AND metrics
# ==========================================
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(
    {
        "model": model,
        "feature_columns": list(X.columns),
        "accuracy": accuracy,
        "classification_report": report_dict,
        "confusion_matrix": cm,
        "classes": labels
    },
    MODEL_PATH
)
print(f"\nModel and evaluation metrics saved successfully at: {MODEL_PATH}")