import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# ── Load Dataset ─────────────────────────────────────────────────────────────
print("Loading dataset...") 
df = pd.read_csv("creditcard.csv")
print(f"Dataset shape: {df.shape}")
print(f"Fraud cases: {df['Class'].sum()} / {len(df)}")

# ── Preprocessing ─────────────────────────────────────────────────────────────
scaler = StandardScaler()
df["Amount_scaled"] = scaler.fit_transform(df["Amount"].values.reshape(-1, 1))
df["Time_scaled"]   = scaler.fit_transform(df["Time"].values.reshape(-1, 1))

feature_cols = [c for c in df.columns if c not in ["Class", "Amount", "Time"]]
feature_cols += ["Amount_scaled", "Time_scaled"]

X = df[feature_cols]
y = df["Class"]

# ── Train/Test Split ──────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Train Model ───────────────────────────────────────────────────────────────
print("\nTraining Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n── Classification Report ──")
print(classification_report(y_test, y_pred, target_names=["Genuine", "Fraud"]))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

# ── Save Model & Scaler ───────────────────────────────────────────────────────
pickle.dump(model,        open("fraud_model.pkl", "wb"))
pickle.dump(scaler,       open("scaler.pkl", "wb"))
pickle.dump(feature_cols, open("feature_cols.pkl", "wb"))

print("\nModel saved: fraud_model.pkl")
print("Scaler saved: scaler.pkl")
print("Features saved: feature_cols.pkl")
