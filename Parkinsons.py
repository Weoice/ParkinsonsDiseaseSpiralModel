import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    ConfusionMatrixDisplay,
)

FOLDER_PATH = r"C:\Users\cvmpr\Downloads\parkinson+disease+spiral+drawings+using+digitized+graphics+tablet\hw_dataset"
LABELS = {"control": 0, "parkinson": 1}
SPIRAL_TEST_IDS = {0, 1}
records = []
for folder_name, label in LABELS.items():
    search = os.path.join(FOLDER_PATH, folder_name, "*.txt")
    for file_path in glob.glob(search):
        temp = pd.read_csv(file_path, sep=";", engine="python", header=None)
        temp["Patient_ID"] = os.path.basename(file_path).replace(".txt", "")
        temp["Label"] = label
        records.append(temp)

df = pd.concat(records, ignore_index=True)
df = df.rename(
    columns={
        0: "X",
        1: "Y",
        2: "Z",
        3: "Pressure",
        4: "GripAngle",
        5: "Timestamp",
        6: "Test_ID",
    }
)

print("--- First 5 Rows ---")
print(df.head())
print("\n--- DataFrame Info ---")
df.info()

print(f"\nRaw rows: {len(df):,}")
df = df.dropna(subset=["Timestamp"])
df = df[df["Pressure"] > 0]  # drop pen-lift samples
df = df[df["Test_ID"].isin(SPIRAL_TEST_IDS)]
print(f"Rows after cleaning: {len(df):,}")


def engineer_kinematics(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("Timestamp").copy()
    dt = group["Timestamp"].diff()

    dx = group["X"].diff()
    dy = group["Y"].diff()
    distance = np.sqrt(dx**2 + dy**2)

    velocity = distance / dt
    acceleration = velocity.diff() / dt
    jerk = acceleration.diff() / dt

    group["Velocity"] = velocity
    group["Acceleration"] = acceleration
    group["Jerk"] = jerk
    median_dt = dt.median()
    window = max(3, int(round(50 / median_dt))) if median_dt and median_dt > 0 else 5
    group["Pressure_RollStd"] = group["Pressure"].rolling(window=window, min_periods=2).std()

    return group


df = df.groupby(["Patient_ID", "Test_ID"], group_keys=False).apply(engineer_kinematics)

# diff()-of-diff()-of-diff() leaves NaNs on the first few rows of every
# group; segment boundaries with dt<=0 produce inf. Drop both.
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=["Velocity", "Acceleration", "Jerk", "Pressure_RollStd"])
agg_funcs = {
    "Velocity": ["mean", "std", "max"],
    "Acceleration": ["mean", "std", "max"],
    "Jerk": ["mean", "std", "max"],
    "Pressure": ["mean", "std"],
    "Pressure_RollStd": ["mean", "max"],
    "GripAngle": ["mean", "std"],
}

features = df.groupby(["Patient_ID", "Test_ID"]).agg(agg_funcs)
features.columns = ["_".join(c) for c in features.columns]
features = features.reset_index()

labels = df.groupby(["Patient_ID", "Test_ID"])["Label"].first().reset_index()
features = features.merge(labels, on=["Patient_ID", "Test_ID"])

print(f"\nFeature table: {features.shape[0]} samples (patients x tests), "
      f"{features.shape[1] - 3} engineered features")
print(features["Label"].value_counts().rename({0: "Control", 1: "Parkinson"}))

feature_cols = [c for c in features.columns if c not in ("Patient_ID", "Test_ID", "Label")]
X = features[feature_cols]
y = features["Label"]
groups = features["Patient_ID"]

splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups))  # 1 fold ~= 20% test

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

assert set(groups.iloc[train_idx]).isdisjoint(groups.iloc[test_idx]), "patient leaked across split!"

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTrain samples: {len(X_train)} (patients: {groups.iloc[train_idx].nunique()}), "
      f"Test samples: {len(X_test)} (patients: {groups.iloc[test_idx].nunique()})")


scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
)
model.fit(X_train_scaled, y_train)

y_proba = model.predict_proba(X_test_scaled)[:, 1]

default_preds = (y_proba >= 0.5).astype(int)
print("\n=== Default threshold (0.5) ===")
print(confusion_matrix(y_test, default_preds))
print(classification_report(y_test, default_preds, target_names=["Control", "Parkinson"], zero_division=0))

precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
target_recall = 0.9
valid = recalls[:-1] >= target_recall
tuned_threshold = thresholds[valid][-1] if valid.any() else 0.5
tuned_preds = (y_proba >= tuned_threshold).astype(int)

print(f"\n=== Tuned threshold ({tuned_threshold:.3f}, targeting recall >= {target_recall}) ===")
cm = confusion_matrix(y_test, tuned_preds)
print(cm)
print(classification_report(y_test, tuned_preds, target_names=["Control", "Parkinson"], zero_division=0))
if len(np.unique(y_test)) > 1:
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")

feat_importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\n--- Feature importances ---")
print(feat_importance)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Control", "Parkinson"])
disp.plot(cmap="Blues")
plt.title(f"Confusion Matrix (threshold={tuned_threshold:.2f})")
plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "confusion_matrix.png")
plt.savefig(out_path)
print(f"\nSaved confusion matrix plot to {out_path}")
cv_accuracy, cv_recall_pd, cv_auc = [], [], []
for fold_train_idx, fold_test_idx in splitter.split(X, y, groups):
    fold_scaler = StandardScaler()
    Xtr = fold_scaler.fit_transform(X.iloc[fold_train_idx])
    Xte = fold_scaler.transform(X.iloc[fold_test_idx])
    ytr, yte = y.iloc[fold_train_idx], y.iloc[fold_test_idx]

    fold_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        scale_pos_weight=(ytr == 0).sum() / (ytr == 1).sum(),
        random_state=42,
    )
    fold_model.fit(Xtr, ytr)
    fold_proba = fold_model.predict_proba(Xte)[:, 1]
    fold_preds = (fold_proba >= 0.5).astype(int)

    cv_accuracy.append((fold_preds == yte).mean())
    cv_recall_pd.append(confusion_matrix(yte, fold_preds, labels=[0, 1])[1, 1] / max((yte == 1).sum(), 1))
    if len(np.unique(yte)) > 1:
        cv_auc.append(roc_auc_score(yte, fold_proba))

print("\n=== 5-fold patient-grouped cross-validation (default 0.5 threshold) ===")
print(f"Accuracy:            {np.mean(cv_accuracy):.3f} +/- {np.std(cv_accuracy):.3f}")
print(f"Parkinson recall:    {np.mean(cv_recall_pd):.3f} +/- {np.std(cv_recall_pd):.3f}")
if cv_auc:
    print(f"ROC-AUC:             {np.mean(cv_auc):.3f} +/- {np.std(cv_auc):.3f}")
print("(Only 40 patients total - treat these as directional, not production-grade, estimates.)")
