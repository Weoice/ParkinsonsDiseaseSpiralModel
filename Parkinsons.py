import os
import glob
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
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

import features as feat

FOLDER_PATH = r"C:\Users\cvmpr\Downloads\parkinson+disease+spiral+drawings+using+digitized+graphics+tablet\hw_dataset"
LABELS = {"control": 0, "parkinson": 1}
ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

# Test_ID 2 is a circular "point stability" test, not a spiral, and isn't
# recorded for every subject (30/40 have it, 10/40 don't) - so only the
# two spiral tasks are used, keeping every subject's feature set complete.
SPIRAL_TEST_IDS = {0, 1}

# ----------------------------------------------------------------------
# 1. Acquire and clean the data
# ----------------------------------------------------------------------
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
# No honest browser-equivalent signal for stylus tilt/twist exists, so this
# is dropped rather than carried by a model that a browser demo can't feed.
df = df.drop(columns=["GripAngle"])

print("--- First 5 Rows ---")
print(df.head())
print("\n--- DataFrame Info ---")
df.info()

print(f"\nRaw rows: {len(df):,}")
df = df.dropna(subset=["Timestamp"])
df = df[df["Pressure"] > 0]  # drop pen-lift samples
df = df[df["Test_ID"].isin(SPIRAL_TEST_IDS)]
print(f"Rows after cleaning: {len(df):,}")

# ----------------------------------------------------------------------
# 2 & 3. Engineer kinematic features (velocity, acceleration, jerk) and
#         a rolling pressure-variance feature, per patient per test.
#         Each (Patient_ID, Test_ID) pair is its own continuous pen
#         trajectory with its own timestamp origin, so diffs must never
#         cross that boundary. Shared with app.py via features.py so
#         training and inference can never drift apart on this logic.
# ----------------------------------------------------------------------
df = feat.compute_point_kinematics(df, ["Patient_ID", "Test_ID"])

# ----------------------------------------------------------------------
# Aggregate to one feature row per (patient, test)
#
# The label is a per-patient diagnosis, not a per-instant one - so the
# unit XGBoost should classify is "one spiral-drawing attempt", not one
# raw timestep. This also matters for the train/test split: timesteps
# from the same patient are highly correlated (same hand, same tremor),
# so splitting rows 80/20 directly would leak that patient's signal into
# both sets and inflate the reported accuracy.
# ----------------------------------------------------------------------
feature_table = feat.aggregate_feature_table(df, ["Patient_ID", "Test_ID"])

labels = df.groupby(["Patient_ID", "Test_ID"])["Label"].first().reset_index()
feature_table = feature_table.merge(labels, on=["Patient_ID", "Test_ID"])

print(f"\nFeature table: {feature_table.shape[0]} samples (patients x tests), "
      f"{feature_table.shape[1] - 3} engineered features")
print(feature_table["Label"].value_counts().rename({0: "Control", 1: "Parkinson"}))

feature_cols = [c for c in feature_table.columns if c not in ("Patient_ID", "Test_ID", "Label")]
X = feature_table[feature_cols]
y = feature_table["Label"]
groups = feature_table["Patient_ID"]

# ----------------------------------------------------------------------
# 4/5. Split by patient (group), not by row, then normalize
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# 5 (cont). Train an XGBoost classifier
# ----------------------------------------------------------------------
MODEL_PARAMS = dict(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
)

model = xgb.XGBClassifier(scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(), **MODEL_PARAMS)
model.fit(X_train_scaled, y_train)

# ----------------------------------------------------------------------
# 6. Evaluate this single split, then tune the decision threshold to
#    minimize false negatives (a missed diagnosis is worse than a false
#    alarm). This split is for reporting only - the artifact shipped to
#    the app below is refit on all patients, see the CV section.
# ----------------------------------------------------------------------
y_proba = model.predict_proba(X_test_scaled)[:, 1]

default_preds = (y_proba >= 0.5).astype(int)
print("\n=== Default threshold (0.5), single 80/20 split ===")
print(confusion_matrix(y_test, default_preds))
print(classification_report(y_test, default_preds, target_names=["Control", "Parkinson"], zero_division=0))

target_recall = 0.9
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
valid = recalls[:-1] >= target_recall
tuned_threshold = thresholds[valid][-1] if valid.any() else 0.5
tuned_preds = (y_proba >= tuned_threshold).astype(int)

print(f"\n=== Tuned threshold ({tuned_threshold:.3f}, targeting recall >= {target_recall}), single split ===")
cm = confusion_matrix(y_test, tuned_preds)
print(cm)
print(classification_report(y_test, tuned_preds, target_names=["Control", "Parkinson"], zero_division=0))
if len(np.unique(y_test)) > 1:
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")

feat_importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\n--- Feature importances (single-split model) ---")
print(feat_importance)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Control", "Parkinson"])
disp.plot(cmap="Blues")
plt.title(f"Confusion Matrix (threshold={tuned_threshold:.2f})")
plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "confusion_matrix.png")
plt.savefig(out_path)
print(f"\nSaved confusion matrix plot to {out_path}")

# ----------------------------------------------------------------------
# 7. Robustness check + out-of-fold threshold + final deployed model
#
# With only 40 patients, a single 80/20 split has just ~8 held-out
# patients - a lucky/unlucky split can make the reported numbers look
# far better (or worse) than the model really is. Re-run across all 5
# patient-grouped folds, report the spread, and also collect each
# patient's out-of-fold probability so every patient contributes to the
# threshold decision exactly once (rather than tuning the shipped
# threshold off the one ~8-patient split above).
# ----------------------------------------------------------------------
cv_accuracy, cv_recall_pd, cv_auc = [], [], []
oof_proba = pd.Series(index=X.index, dtype=float)

for fold_train_idx, fold_test_idx in splitter.split(X, y, groups):
    fold_scaler = StandardScaler()
    Xtr = fold_scaler.fit_transform(X.iloc[fold_train_idx])
    Xte = fold_scaler.transform(X.iloc[fold_test_idx])
    ytr, yte = y.iloc[fold_train_idx], y.iloc[fold_test_idx]

    fold_model = xgb.XGBClassifier(scale_pos_weight=(ytr == 0).sum() / (ytr == 1).sum(), **MODEL_PARAMS)
    fold_model.fit(Xtr, ytr)
    fold_proba = fold_model.predict_proba(Xte)[:, 1]
    fold_preds = (fold_proba >= 0.5).astype(int)

    oof_proba.iloc[fold_test_idx] = fold_proba
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

oof_precisions, oof_recalls, oof_thresholds = precision_recall_curve(y, oof_proba)
oof_valid = oof_recalls[:-1] >= target_recall
tuned_threshold_final = oof_thresholds[oof_valid][-1] if oof_valid.any() else 0.5
print(f"\nOut-of-fold tuned threshold (all 40 patients, recall >= {target_recall}): {tuned_threshold_final:.3f}")

# Refit on every patient for the artifact that actually ships - with a
# dataset this small, holding out 20% for the deployed model wastes signal
# the app doesn't need to hold back.
final_scaler = StandardScaler().fit(X)
final_model = xgb.XGBClassifier(scale_pos_weight=(y == 0).sum() / (y == 1).sum(), **MODEL_PARAMS)
final_model.fit(final_scaler.transform(X), y)

os.makedirs(ARTIFACT_DIR, exist_ok=True)
final_model.save_model(os.path.join(ARTIFACT_DIR, "xgb_model.json"))
joblib.dump(final_scaler, os.path.join(ARTIFACT_DIR, "scaler.joblib"))
with open(os.path.join(ARTIFACT_DIR, "feature_columns.json"), "w") as f:
    json.dump(feature_cols, f)
with open(os.path.join(ARTIFACT_DIR, "metadata.json"), "w") as f:
    json.dump(
        {
            "tuned_threshold": float(tuned_threshold_final),
            "target_recall": target_recall,
            "min_kinematic_rows": feat.MIN_KINEMATIC_ROWS,
            "cv_accuracy_mean": float(np.mean(cv_accuracy)),
            "cv_recall_parkinson_mean": float(np.mean(cv_recall_pd)),
            "cv_roc_auc_mean": float(np.mean(cv_auc)) if cv_auc else None,
        },
        f,
        indent=2,
    )
print(f"\nSaved deployment artifacts (model refit on all {len(X)} samples) to {ARTIFACT_DIR}")
