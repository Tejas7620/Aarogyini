"""
CareMaa - Maternal Risk Assessment Model Training Script
Dataset: Maternal Health Risk (Kaggle) or synthetic generation below
Features: Age, SystolicBP, DiastolicBP, BloodSugar, BodyTemp, HeartRate
Target: RiskLevel (0=Low, 1=Medium, 2=High)
"""

import numpy as np
import pandas as pd
import pickle
import os
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# ─── 1. Generate Synthetic Training Data ────────────────────────────────────
np.random.seed(42)
N = 3000

def generate_dataset(n):
    rows = []
    for _ in range(n):
        risk = np.random.choice(["Low", "Medium", "High"], p=[0.5, 0.3, 0.2])
        if risk == "Low":
            age = np.random.randint(18, 33)
            sbp = np.random.randint(90, 125)
            dbp = np.random.randint(60, 82)
            bs = np.random.uniform(70, 115)
            temp = np.random.uniform(97.0, 99.0)
            hr = np.random.randint(60, 90)
        elif risk == "Medium":
            age = np.random.randint(25, 38)
            sbp = np.random.randint(120, 142)
            dbp = np.random.randint(78, 92)
            bs = np.random.uniform(100, 145)
            temp = np.random.uniform(98.5, 100.5)
            hr = np.random.randint(78, 105)
        else:  # High
            age = np.random.randint(30, 50)
            sbp = np.random.randint(138, 180)
            dbp = np.random.randint(88, 120)
            bs = np.random.uniform(130, 200)
            temp = np.random.uniform(99.0, 103.0)
            hr = np.random.randint(95, 130)

        rows.append({
            "Age": age, "SystolicBP": sbp, "DiastolicBP": dbp,
            "BS": round(bs, 1), "BodyTemp": round(temp, 1),
            "HeartRate": hr, "RiskLevel": risk
        })
    return pd.DataFrame(rows)

print("🔄 Generating training dataset...")
df = generate_dataset(N)
print(f"✅ Dataset: {df.shape[0]} samples")
print(df["RiskLevel"].value_counts())

# ─── 2. Preprocess ──────────────────────────────────────────────────────────
le = LabelEncoder()
df["RiskLabel"] = le.fit_transform(df["RiskLevel"])  # High=0, Low=1, Medium=2
# Remap to 0=Low, 1=Medium, 2=High
risk_order = {"Low": 0, "Medium": 1, "High": 2}
df["RiskLabel"] = df["RiskLevel"].map(risk_order)

feature_cols = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]
X = df[feature_cols].values
y = df["RiskLabel"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ─── 3. Train Model ─────────────────────────────────────────────────────────
print("\n🔄 Training RandomForest model...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_split=4,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1,
)
model.fit(X_train, y_train)

# ─── 4. Evaluate ────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Test Accuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Low", "Medium", "High"]))

cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
print(f"Cross-Validation: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ─── 5. Feature Importance ──────────────────────────────────────────────────
feat_imp = sorted(zip(feature_cols, model.feature_importances_), key=lambda x: -x[1])
print("\n📊 Feature Importances:")
for feat, imp in feat_imp:
    print(f"  {feat}: {imp:.4f}")

# ─── 6. Save Model ──────────────────────────────────────────────────────────
model_dir = Path(__file__).parent / "model"
model_dir.mkdir(exist_ok=True)
model_path = model_dir / "model.pkl"

with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"\n✅ Model saved to {model_path}")

# ─── 7. Quick Inference Test ─────────────────────────────────────────────────
sample = np.array([[28, 145, 92, 160, 100.2, 98]])  # Should be High
pred = model.predict(sample)[0]
proba = model.predict_proba(sample)[0]
labels = ["Low", "Medium", "High"]
print(f"\n🧪 Test prediction (High risk input): {labels[pred]} (confidence: {max(proba):.2f})")
