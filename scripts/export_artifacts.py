"""Rebuild artifacts/ from UCI hw_dataset. Run: python scripts/export_artifacts.py"""
from pathlib import Path
import json
import glob
import os
import sys

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import features as feat

FOLDER_PATH = r"C:\Users\cvmpr\Downloads\parkinson+disease+spiral+drawings+using+digitized+graphics+tablet\hw_dataset"
LABELS = {"control": 0, "parkinson": 1}
SPIRAL_TEST_IDS = {0, 1}
ARTIFACT_DIR = ROOT / "artifacts"
TARGET_RECALL = 0.9


def load_raw() -> pd.DataFrame:
    if not os.path.isdir(FOLDER_PATH):
        raise FileNotFoundError(
            f"Dataset folder not found: {FOLDER_PATH}\n"
            "Download the UCI Parkinson spiral drawings hw_dataset and update FOLDER_PATH."
        )
    records = []
    for folder_name, label in LABELS.items():
        search = os.path.join(FOLDER_PATH, folder_name, "*.txt")
        for file_path in glob.glob(search):
            temp = pd.read_csv(file_path, sep=";", engine="python", header=None)
            temp["Patient_ID"] = os.path.basename(file_path).replace(".txt", "")
            temp["Label"] = label
            records.append(temp)
    df = pd.concat(records, ignore_index=True)
    return df.rename(
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


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["Timestamp"])
    df = df[df["Pressure"] > 0]
    return df[df["Test_ID"].isin(SPIRAL_TEST_IDS)].copy()


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    kinematics = feat.compute_point_kinematics(df, ["Patient_ID", "Test_ID"])
    features = feat.aggregate_feature_table(kinematics, ["Patient_ID", "Test_ID"], feat.AGG_FUNCS)
    labels = kinematics.groupby(["Patient_ID", "Test_ID"])["Label"].first().reset_index()
    return features.merge(labels, on=["Patient_ID", "Test_ID"])


def tune_threshold(y_true, y_proba, target_recall: float = TARGET_RECALL) -> float:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    valid = recalls[:-1] >= target_recall
    return float(thresholds[valid][-1]) if valid.any() else 0.5


def run_cv(X, y, groups, splitter) -> dict:
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

        cv_accuracy.append(float((fold_preds == yte).mean()))
        cv_recall_pd.append(
            float(confusion_matrix(yte, fold_preds, labels=[0, 1])[1, 1] / max((yte == 1).sum(), 1))
        )
        if len(np.unique(yte)) > 1:
            cv_auc.append(float(roc_auc_score(yte, fold_proba)))

    return {
        "cv_accuracy_mean": float(np.mean(cv_accuracy)),
        "cv_recall_parkinson_mean": float(np.mean(cv_recall_pd)),
        "cv_roc_auc_mean": float(np.mean(cv_auc)) if cv_auc else None,
    }


def main() -> None:
    print(f"Loading dataset from {FOLDER_PATH}")
    df = clean(load_raw())
    print(f"Rows after cleaning: {len(df):,}")

    features = build_feature_table(df)
    feature_cols = [c for c in features.columns if c not in ("Patient_ID", "Test_ID", "Label")]
    expected = feat.feature_column_names(feat.AGG_FUNCS)
    if feature_cols != expected:
        raise RuntimeError(f"Feature columns mismatch.\nGot: {feature_cols}\nExpected: {expected}")

    X = features[feature_cols]
    y = features["Label"]
    groups = features["Patient_ID"]

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    if not set(groups.iloc[train_idx]).isdisjoint(set(groups.iloc[test_idx])):
        raise RuntimeError("patient leaked across split!")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

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
    tuned_threshold = tune_threshold(y_test, y_proba)
    cv_metrics = run_cv(X, y, groups, splitter)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, ARTIFACT_DIR / "scaler.joblib")
    model.save_model(str(ARTIFACT_DIR / "xgb_model.json"))
    (ARTIFACT_DIR / "feature_columns.json").write_text(json.dumps(feature_cols, indent=2))

    metadata = {
        "tuned_threshold": tuned_threshold,
        "display_threshold": 0.5,
        "target_recall": TARGET_RECALL,
        "min_kinematic_rows": feat.MIN_KINEMATIC_ROWS,
        **cv_metrics,
    }
    (ARTIFACT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"Wrote artifacts to {ARTIFACT_DIR}")
    print(f"feature_columns ({len(feature_cols)}): {feature_cols}")
    print(f"tuned_threshold={tuned_threshold:.4f}, display_threshold=0.5")
    print(
        f"CV accuracy={cv_metrics['cv_accuracy_mean']:.3f}, "
        f"PD recall={cv_metrics['cv_recall_parkinson_mean']:.3f}, "
        f"AUC={cv_metrics['cv_roc_auc_mean']}"
    )


if __name__ == "__main__":
    main()
