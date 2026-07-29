"""Shared prediction pipeline for the Vercel API and local tests."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
import xgboost as xgb

import features as feat

DISPLAY_THRESHOLD = 0.5
SHORT_STROKE_MSG = "Need a longer continuous trace before a stable score is possible."


def build_feature_row(raw_points: list) -> pd.DataFrame | None:
    if not raw_points:
        return None
    points_df = pd.DataFrame(raw_points).rename(
        columns={"x": "X", "y": "Y", "t": "Timestamp", "pressure": "Pressure", "stroke": "_stroke"}
    )
    kinematics = feat.compute_point_kinematics(points_df, ["_stroke"])
    if len(kinematics) < feat.MIN_KINEMATIC_ROWS:
        return None
    kinematics = kinematics.assign(_session=0)
    table = feat.aggregate_feature_table(kinematics, ["_session"], feat.AGG_FUNCS)
    return table.drop(columns=["_session"])


@lru_cache(maxsize=4)
def _load_artifacts(artifact_dir: str):
    root = Path(artifact_dir)
    model = xgb.XGBClassifier()
    model.load_model(str(root / "xgb_model.json"))
    scaler = joblib.load(root / "scaler.joblib")
    feature_cols = json.loads((root / "feature_columns.json").read_text())
    return model, scaler, feature_cols


def predict_from_points(raw_points: list, artifact_dir: Path) -> dict:
    feature_row = build_feature_row(raw_points)
    if feature_row is None:
        raise ValueError(SHORT_STROKE_MSG)

    model, scaler, feature_cols = _load_artifacts(str(artifact_dir.resolve()))
    feature_row = feature_row[feature_cols]
    proba = float(model.predict_proba(scaler.transform(feature_row))[0, 1])
    flagged = proba >= DISPLAY_THRESHOLD
    return {
        "probability": proba,
        "flagged": flagged,
        "threshold": DISPLAY_THRESHOLD,
        "similarity": "parkinson" if flagged else "control",
        "features": {col: float(feature_row.iloc[0][col]) for col in feature_cols},
    }
