# Design: Vercel static frontend + Python predict API

Date: 2026-07-29  
Status: pending user review

## Goal

Deploy the Parkinson spiral demo on Vercel. Replace Streamlit with a static site and a Python serverless endpoint. Keep the same feature pipeline and model behavior, with a clearer UI and a fairer decision threshold for browser drawings.

## Architecture

| Piece | Role |
| --- | --- |
| `public/index.html`, `public/styles.css`, `public/app.js` | Static UI: spiral canvas, results, disease sidebar |
| `api/predict.py` | Vercel Python function: validate points, engineer features, scale, predict |
| `features.py` | Shared kinematic feature code (reuse as-is) |
| `artifacts/` | `xgb_model.json`, `scaler.joblib`, `feature_columns.json`, `metadata.json` |
| `vercel.json` | Static root `public/`; route `POST /api/predict` |
| `requirements.txt` | API deps only (no Streamlit): xgboost, scikit-learn, pandas, numpy, joblib |

**Request flow**

1. User draws on canvas; pointer-up sends JSON `{ "points": [{x,y,t,pressure,stroke}, ...] }` to `POST /api/predict`.
2. API builds a feature row via `features.py`, applies the saved scaler, runs XGBoost `predict_proba`.
3. Response: probability, flagged (threshold 0.5), features, or a clear error if the stroke is too short.

**Local Streamlit (`app.py`)**

Keep for local experiments if useful, but Vercel deployment does not use it. Do not list Streamlit in the production `requirements.txt`.

## Missing artifact

`artifacts/scaler.joblib` is not in the repo today, but the app expects it. During implementation, regenerate and save it from the training path in `Parkinsons.py` (same feature set as `feature_columns.json`, GripAngle excluded) so predictions match training.

## Prediction fix

**Problem:** The UI used `metadata.tuned_threshold` (~0.923). That threshold was tuned for high Parkinson recall on tablet holdout data. Browser mouse/touch strokes almost never reach it, so the label almost always reads as control even when the raw probability is meaningful.

**Fix for the deployed demo:**

- Primary display: probability (percent) plus a Control to Parkinson's bar.
- Binary label uses threshold **0.5**.
- Copy explains that browser input differs from the digitizing-tablet study.
- Keep the original tuned threshold in methodology text / metadata for honesty about training, but do not use it for the live label.

Also note domain shift: constant or fake pressure from mouse/trackpad weakens pressure features. The disclaimer stays visible.

## UI / visual design

Approved direction: **Clinical teal** (mockup `layout-sidebar-v1`).

- Background: cool paper wash (`#f4f8f9` to `#eef4f6`).
- Accent: muted teal `#3d7a86`; stroke: restrained coral `#c45c5c`.
- Fonts: **Source Serif 4** (headings), **Source Sans 3** (body/UI). Load from Google Fonts or equivalent.
- Layout: left sidebar (~280px) + main content. On narrow screens, sidebar stacks above or below main (single column).
- Title: **Parkinson's Disease Prediction Model**.
- Eyebrow: Informational research demo.
- Sidebar: short facts about Parkinson's, why spiral/kinematic signals matter, dataset/model line, passion-project disclaimer.
- Main: short instruction, canvas + Clear, result panel with probability and bar.
- One composition; canvas and result are functional surfaces only (no decorative card clutter).

### Copy rules

- No em dashes. Prefer commas, periods, or plain hyphens where needed.
- Formal but human. Sound like a careful student researcher, not marketing or chatbot filler.
- Avoid hype, emoji clutter, and vague filler phrasing.
- Keep the medical disclaimer clear and plain.

## API contract

**Success (200)**

```json
{
  "probability": 0.41,
  "flagged": false,
  "threshold": 0.5,
  "similarity": "control",
  "features": { "Jerk_mean": 0.01 }
}
```

**Too short / empty (400)**

```json
{ "error": "Need a longer continuous trace before a stable score is possible." }
```

CORS: same-origin on Vercel; allow local `vercel dev` if needed.

## Out of scope

- Retraining the model
- Auth, analytics, databases
- Mobile-native apps
- Changing the UCI training pipeline beyond exporting the missing scaler

## Success criteria

1. `vercel` (or `vercel --prod`) deploys without Streamlit.
2. Drawing a spiral returns a probability and a 0.5-threshold label.
3. UI matches the approved clinical-teal + sidebar mockup.
4. Site copy has no em dashes and reads formal but natural.
5. Scaler artifact is present and used consistently with `feature_columns.json`.
