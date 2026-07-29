# Vercel Static + Predict API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a Vercel-deployable Parkinson’s spiral demo: clinical-teal static UI + Python `/api/predict`, with a regenerated scaler and a 0.5 display threshold.

**Architecture:** Static files in `public/` call `POST /api/predict`. The handler reuses root `features.py` and `artifacts/` (XGBoost JSON model + StandardScaler + feature column list). No Streamlit on Vercel.

**Tech Stack:** Vanilla HTML/CSS/JS, Vercel Python serverless (`BaseHTTPRequestHandler` or ASGI `app`), xgboost, scikit-learn, pandas, numpy, joblib, pytest.

## Global Constraints

- No em dashes in user-facing copy; formal but human tone (student research project, not marketing).
- Live binary label uses threshold **0.5** (not `tuned_threshold` ~0.923).
- Production `requirements.txt` must not include Streamlit.
- Visual direction: clinical teal (`#3d7a86`, coral stroke `#c45c5c`, paper wash `#f4f8f9`/`#eef4f6`), Source Serif 4 + Source Sans 3, left disease sidebar.
- Title: **Parkinson’s Disease Prediction Model**.
- This folder may not be a git repo; treat commit steps as optional (init/commit only if `.git` exists or user asks).

---

## File map

| File | Responsibility |
| --- | --- |
| `scripts/export_artifacts.py` | Retrain-or-refit path that writes `artifacts/scaler.joblib`, refreshes model/metadata/columns aligned with GripAngle-excluded features |
| `api/predict.py` | HTTP handler: parse points JSON, feature row, scale, predict, JSON response |
| `features.py` | Unchanged kinematic helpers (imported by API) |
| `public/index.html` | Markup: sidebar + main canvas + result |
| `public/styles.css` | Clinical-teal layout and typography |
| `public/app.js` | Canvas drawing, POST predict, render result |
| `vercel.json` | Static `public` + function routing if needed |
| `requirements.txt` | Deploy deps only |
| `.gitignore` | Ignore `.superpowers/`, `__pycache__/`, `.vercel/`, local logs |
| `tests/test_predict_pipeline.py` | Feature row + threshold behavior without needing full HTTP stack |
| `app.py` | Leave as local Streamlit; do not use for deploy |

---

### Task 1: Export missing scaler (and aligned artifacts)

**Files:**
- Create: `scripts/export_artifacts.py`
- Create/overwrite: `artifacts/scaler.joblib`
- Possibly refresh: `artifacts/xgb_model.json`, `artifacts/feature_columns.json`, `artifacts/metadata.json`
- Test: `tests/test_artifacts_exist.py`

**Interfaces:**
- Consumes: UCI hw_dataset at `Parkinsons.py`’s `FOLDER_PATH` (verified present on author machine); `features.py` AGG_FUNCS (no GripAngle)
- Produces: `artifacts/scaler.joblib` loadable via `joblib.load`; model via `xgb.XGBClassifier().load_model(...)`; `metadata.json` includes `tuned_threshold`, CV metrics, and `display_threshold: 0.5`

- [ ] **Step 1: Write failing artifact presence test**

```python
# tests/test_artifacts_exist.py
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"

def test_scaler_exists():
    assert (ART / "scaler.joblib").is_file()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_artifacts_exist.py -v`  
Expected: FAIL (`scaler.joblib` missing)

- [ ] **Step 3: Implement `scripts/export_artifacts.py`**

Script outline (mirror training in `Parkinsons.py`, but use `features.AGG_FUNCS` / no GripAngle so columns match the demo):

```python
"""Rebuild artifacts/ from UCI hw_dataset. Run: python scripts/export_artifacts.py"""
from pathlib import Path
import json, glob, os
import joblib, numpy as np, pandas as pd, xgboost as xgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_curve

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import features as feat

FOLDER_PATH = r"C:\Users\cvmpr\Downloads\parkinson+disease+spiral+drawings+using+digitized+graphics+tablet\hw_dataset"
# ... load/clean like Parkinsons.py ...
# engineer with feat.compute_point_kinematics(..., ["Patient_ID", "Test_ID"])
# aggregate with feat.AGG_FUNCS
# fit StandardScaler + XGBClassifier on patient-grouped fold 0 train split
# dump scaler.joblib, xgb_model.json, feature_columns.json, metadata.json
# metadata["display_threshold"] = 0.5
# metadata["tuned_threshold"] = <from PR curve as before>
```

Keep the script self-contained enough to re-run; hardcode the known local dataset path (same as `Parkinsons.py`) and fail with a clear message if the folder is missing.

- [ ] **Step 4: Run export, then re-run test**

Run: `python scripts/export_artifacts.py` then `pytest tests/test_artifacts_exist.py -v`  
Expected: PASS; `artifacts/scaler.joblib` present; `feature_columns.json` length 13 (no GripAngle)

- [ ] **Step 5: Optional commit**

```bash
git add scripts/export_artifacts.py artifacts/ tests/test_artifacts_exist.py
git commit -m "Add artifact export script and regenerate scaler"
```

---

### Task 2: Predict pipeline helpers + unit tests

**Files:**
- Create: `predict_core.py` (shared pure logic used by API; keeps `api/predict.py` thin)
- Create: `tests/test_predict_pipeline.py`

**Interfaces:**
- Consumes: `features.build` helpers; artifacts via Path
- Produces:
  - `build_feature_row(raw_points: list) -> pd.DataFrame | None` (same semantics as current `app.py`)
  - `predict_from_points(raw_points: list, artifact_dir: Path) -> dict` returning keys `probability`, `flagged`, `threshold`, `similarity`, `features` or raises `ValueError` with user-facing message

- [ ] **Step 1: Write failing tests**

```python
# tests/test_predict_pipeline.py
from pathlib import Path
import pytest
from predict_core import build_feature_row, predict_from_points

ROOT = Path(__file__).resolve().parents[1]

def _fake_stroke(n=80):
    # spiral-ish points with increasing t
    pts = []
    for i in range(n):
        ang = i * 0.25
        r = 5 + i * 0.8
        pts.append({
            "x": 240 + r * __import__("math").cos(ang),
            "y": 240 + r * __import__("math").sin(ang),
            "t": i * 16.0,
            "pressure": 0.5,
            "stroke": 1,
        })
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
```

- [ ] **Step 2: Run tests (expect fail)**

Run: `pytest tests/test_predict_pipeline.py -v`  
Expected: FAIL import / missing module

- [ ] **Step 3: Implement `predict_core.py`**

Port `build_feature_row` from `app.py`. Load model/scaler/columns once via module-level cache or function args. Use `threshold = 0.5` for `flagged` / `similarity` (map flagged True → `"parkinson"`, else `"control"`). Include feature dict from the single row.

- [ ] **Step 4: Run tests (expect pass)**

Run: `pytest tests/test_predict_pipeline.py -v`  
Expected: PASS

- [ ] **Step 5: Optional commit**

```bash
git add predict_core.py tests/test_predict_pipeline.py
git commit -m "Add shared predict pipeline with 0.5 display threshold"
```

---

### Task 3: Vercel Python API endpoint

**Files:**
- Create: `api/predict.py`
- Modify: `requirements.txt` (remove streamlit; keep xgboost, scikit-learn, pandas, numpy, joblib)
- Create: `vercel.json` if needed for static + API
- Create: `.gitignore`

**Interfaces:**
- Consumes: `predict_core.predict_from_points`
- Produces: HTTP `POST /api/predict` JSON per design spec; `OPTIONS` for CORS if local tools need it

- [ ] **Step 1: Write handler**

```python
# api/predict.py
from http.server import BaseHTTPRequestHandler
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from predict_core import predict_from_points

class handler(BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        points = body.get("points") or []
        try:
            result = predict_from_points(points, ROOT / "artifacts")
            payload, code = result, 200
        except ValueError as e:
            payload, code = {"error": str(e)}, 400
        except Exception:
            payload, code = {"error": "Prediction failed. Try a longer continuous trace."}, 500
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self._cors()
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
```

Raise `ValueError` from `predict_core` when stroke too short, with message:  
`Need a longer continuous trace before a stable score is possible.`

- [ ] **Step 2: Update `requirements.txt`**

```
xgboost==3.3.0
scikit-learn==1.7.2
pandas==2.3.3
numpy==2.3.4
joblib==1.5.2
```

(Pin close to current; adjust only if Vercel Python 3.12 cannot resolve.)

- [ ] **Step 3: Add `vercel.json` and `.gitignore`**

```json
{
  "version": 2,
  "outputDirectory": "public"
}
```

If Vercel’s Python detection needs an explicit include for artifacts/features, add `functions` / `includeFiles` per current Vercel Python docs so `artifacts/**` and root modules ship with the function.

`.gitignore`:

```
.superpowers/
.vercel/
__pycache__/
*.pyc
.venv/
app_streamlit.log
```

- [ ] **Step 4: Smoke-test handler locally**

Run one of:
- `vercel dev` (preferred), or
- a tiny script that imports `predict_from_points` with a fake stroke (already covered by Task 2)

Expected: 200 JSON with `threshold: 0.5`

- [ ] **Step 5: Optional commit**

```bash
git add api/predict.py requirements.txt vercel.json .gitignore
git commit -m "Add Vercel predict API and deploy config"
```

---

### Task 4: Static clinical-teal UI

**Files:**
- Create: `public/index.html`
- Create: `public/styles.css`
- Create: `public/app.js`

**Interfaces:**
- Consumes: `POST /api/predict` response shape from Task 3
- Produces: Working canvas demo matching approved mockup

- [ ] **Step 1: Markup (`public/index.html`)**

Structure:
- Sidebar: About Parkinson’s (short human paragraphs), why kinematics/spirals, dataset/model line, passion-project disclaimer
- Main: eyebrow “Informational research demo”, H1 “Parkinson’s Disease Prediction Model”, instruction, canvas, Clear button, status text, result panel (probability, label, bar, note about tablet vs browser)

Copy rules: no em dashes; plain, formal, human.

- [ ] **Step 2: Styles (`public/styles.css`)**

CSS variables for teal palette; Source Serif 4 / Source Sans 3 via Google Fonts link in HTML; sidebar + main grid; responsive single column under ~800px; canvas white surface with teal border; result bar Control→Parkinson’s.

- [ ] **Step 3: Canvas + API client (`public/app.js`)**

Port drawing logic from `app.py`’s JS (guide spiral, pointer events, clear). On stroke end, `fetch('/api/predict', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({points}) })`. Render probability as percent, set bar width, show similarity text using 0.5 threshold fields from API. On clear, reset result UI. Handle 400/500 with the API’s `error` string.

- [ ] **Step 4: Manual UI check**

Open `public/index.html` via `vercel dev` (so `/api/predict` works). Draw a long spiral; confirm result updates; Clear resets. Skim all visible text for em dashes and stiff AI phrasing.

- [ ] **Step 5: Optional commit**

```bash
git add public/
git commit -m "Add clinical-teal static demo UI"
```

---

### Task 5: Verification and deploy readiness

**Files:**
- Possibly tweak `vercel.json` / path imports if deploy bundling fails
- Optional: short `README.md` with run/deploy steps (only if useful; keep brief)

- [ ] **Step 1: Run full pytest**

Run: `pytest -v`  
Expected: all pass

- [ ] **Step 2: Local `vercel dev` end-to-end**

Draw spiral → probability + label; short scribble → friendly error; no Streamlit process required.

- [ ] **Step 3: Confirm deploy checklist**

- [ ] `artifacts/scaler.joblib` committed or present for upload  
- [ ] `requirements.txt` has no streamlit  
- [ ] User-facing copy has no em dashes  
- [ ] Display threshold is 0.5  

- [ ] **Step 4: Optional commit of any deploy fixes**

---

## Spec coverage checklist

| Spec item | Task |
| --- | --- |
| Static frontend + Python API | 3, 4 |
| Reuse `features.py` + artifacts | 1, 2 |
| Regenerate missing scaler | 1 |
| Threshold 0.5 for live label | 2, 3, 4 |
| Clinical teal + sidebar + title | 4 |
| Copy rules (no em dashes, human formal) | 4 |
| `vercel.json` / requirements without Streamlit | 3 |
| Success criteria / verification | 5 |

## Self-review notes

- No TBD placeholders in steps.
- `predict_from_points` / response keys stay consistent across Tasks 2–4.
- GripAngle stays excluded everywhere via `features.AGG_FUNCS`.
