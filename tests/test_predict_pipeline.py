from pathlib import Path
import math

from predict_core import build_feature_row, predict_from_points

ROOT = Path(__file__).resolve().parents[1]


def _fake_stroke(n=80):
    pts = []
    for i in range(n):
        ang = i * 0.25
        r = 5 + i * 0.8
        pts.append(
            {
                "x": 240 + r * math.cos(ang),
                "y": 240 + r * math.sin(ang),
                "t": i * 16.0,
                "pressure": 0.5,
                "stroke": 1,
            }
        )
    return pts


def test_short_stroke_returns_none():
    assert build_feature_row([]) is None
    assert build_feature_row(_fake_stroke(5)) is None


def test_predict_uses_half_threshold():
    out = predict_from_points(_fake_stroke(100), ROOT / "artifacts")
    assert out["threshold"] == 0.5
    assert out["flagged"] == (out["probability"] >= 0.5)
    assert out["similarity"] in ("control", "parkinson")
    assert 0.0 <= out["probability"] <= 1.0
