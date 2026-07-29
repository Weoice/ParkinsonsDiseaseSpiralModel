import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

import features as feat

APP_DIR = Path(__file__).parent
ARTIFACT_DIR = APP_DIR / "artifacts"
CANVAS_SIZE = 480


@st.cache_resource
def load_artifacts():
    model = xgb.XGBClassifier()
    model.load_model(str(ARTIFACT_DIR / "xgb_model.json"))
    scaler = joblib.load(ARTIFACT_DIR / "scaler.joblib")
    feature_cols = json.loads((ARTIFACT_DIR / "feature_columns.json").read_text())
    metadata = json.loads((ARTIFACT_DIR / "metadata.json").read_text())
    return model, scaler, feature_cols, metadata


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


HTML = """
<div class="spiral-wrap">
  <canvas id="spiral-canvas" width="__SIZE__" height="__SIZE__"></canvas>
  <div class="controls">
    <button id="clear-btn" type="button">Clear</button>
    <span id="status-text">Draw a spiral above</span>
  </div>
</div>
""".replace("__SIZE__", str(CANVAS_SIZE))

CSS = """
.spiral-wrap { display: flex; flex-direction: column; align-items: flex-start; gap: 8px; }
#spiral-canvas {
  border: 1px solid var(--st-border-color, #888);
  border-radius: 8px;
  touch-action: none;
  cursor: crosshair;
  background: transparent;
}
.controls { display: flex; align-items: center; gap: 12px; }
#clear-btn {
  padding: 4px 14px;
  border-radius: 6px;
  border: 1px solid var(--st-border-color, #888);
  background: var(--st-secondary-background-color, #f0f0f0);
  cursor: pointer;
}
#status-text { font-size: 0.85em; opacity: 0.75; }
"""

JS = """
export default function(component) {
    const { setTriggerValue, parentElement } = component;
    const canvas = parentElement.querySelector('#spiral-canvas');
    const clearBtn = parentElement.querySelector('#clear-btn');
    const statusText = parentElement.querySelector('#status-text');
    const ctx = canvas.getContext('2d');

    const SIZE = __SIZE__;
    const CENTER = SIZE / 2;
    const MAX_RADIUS = SIZE * 0.42;
    const TURNS = 4.5;

    function drawGuide() {
        ctx.clearRect(0, 0, SIZE, SIZE);
        ctx.save();
        ctx.strokeStyle = 'rgba(128, 128, 128, 0.35)';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        const thetaMax = TURNS * 2 * Math.PI;
        const steps = 400;
        for (let i = 0; i <= steps; i++) {
            const theta = (i / steps) * thetaMax;
            const r = (MAX_RADIUS / thetaMax) * theta;
            const x = CENTER + r * Math.cos(theta);
            const y = CENTER + r * Math.sin(theta);
            if (i === 0) { ctx.moveTo(x, y); } else { ctx.lineTo(x, y); }
        }
        ctx.stroke();
        ctx.restore();
    }

    let points = [];
    let strokeId = 0;
    let drawing = false;
    let lastX = null;
    let lastY = null;

    function addPoint(e) {
        points.push({
            x: e.offsetX,
            y: e.offsetY,
            t: e.timeStamp,
            pressure: e.pressure,
            stroke: strokeId,
        });
    }

    canvas.addEventListener('pointerdown', (e) => {
        drawing = true;
        strokeId += 1;
        canvas.setPointerCapture(e.pointerId);
        lastX = e.offsetX;
        lastY = e.offsetY;
        addPoint(e);
        statusText.textContent = 'Drawing...';
    });

    canvas.addEventListener('pointermove', (e) => {
        if (!drawing) return;
        addPoint(e);
        ctx.strokeStyle = '#e04452';
        ctx.lineWidth = 2.5;
        ctx.lineCap = 'round';
        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(e.offsetX, e.offsetY);
        ctx.stroke();
        lastX = e.offsetX;
        lastY = e.offsetY;
    });

    function endStroke(e) {
        if (!drawing) return;
        drawing = false;
        addPoint(e);
        statusText.textContent = points.length + ' points captured';
        setTriggerValue('points', points);
    }

    canvas.addEventListener('pointerup', endStroke);
    canvas.addEventListener('pointercancel', endStroke);

    clearBtn.addEventListener('click', () => {
        points = [];
        strokeId = 0;
        drawing = false;
        drawGuide();
        statusText.textContent = 'Draw a spiral above';
        setTriggerValue('cleared', true);
    });

    drawGuide();
}
""".replace("__SIZE__", str(CANVAS_SIZE))


st.set_page_config(page_title="Spiral Tremor Demo", page_icon="🌀")

model, scaler, feature_cols, metadata = load_artifacts()

st.title("Spiral Drawing — Kinematic Tremor Demo")

st.warning(
    "**Research / educational proof-of-concept — not a medical device.** "
    "This model was trained on 40 subjects using a calibrated digitizing "
    "tablet. Your mouse, trackpad, or touchscreen samples motion "
    "differently, and unless you're using a pressure-sensitive stylus it "
    "reports a constant, fake pressure value. Treat any result below as "
    "illustrative only — never as a real health assessment."
)

st.write(
    "Trace the faint spiral guide below in one continuous stroke, then "
    "lift your pointer — a prediction appears automatically. Click "
    "**Clear** to try again."
)

spiral_component = st.components.v2.component("spiral_canvas", html=HTML, css=CSS, js=JS)

if "stroke_points" not in st.session_state:
    st.session_state["stroke_points"] = []

result = spiral_component(
    key="spiral_input",
    on_points_change=lambda: None,
    on_cleared_change=lambda: None,
)

if result.cleared:
    st.session_state["stroke_points"] = []
if result.points:
    st.session_state["stroke_points"] = result.points

st.divider()

feature_row = build_feature_row(st.session_state["stroke_points"])

if not st.session_state["stroke_points"]:
    st.info("Draw a spiral above to get a prediction.")
elif feature_row is None:
    st.info("Keep drawing — a stable prediction needs a longer continuous trace.")
else:
    feature_row = feature_row[feature_cols]
    proba = float(model.predict_proba(scaler.transform(feature_row))[0, 1])
    threshold = metadata["tuned_threshold"]
    flagged = proba >= threshold
    similarity = "Parkinson's-labeled" if flagged else "control"

    st.metric("Model probability", f"{proba:.1%}")
    st.write(
        f"This drawing's kinematics look more similar to the "
        f"**{similarity}** training examples (decision threshold: {threshold:.1%})."
    )
    st.caption("Not a diagnosis. See disclaimer above.")

    with st.expander("Computed features for this drawing"):
        st.dataframe(feature_row.T.rename(columns={feature_row.index[0]: "value"}))

with st.expander("How this works"):
    st.write(
        f"Trained on the UCI Parkinson's spiral-drawing digitizing-tablet dataset "
        f"(40 subjects, control vs. Parkinson's). Velocity, acceleration, and jerk "
        f"are derived from successive time-differences of the pen's X/Y position; "
        f"jerk (rate of change of acceleration) is the strongest signal, consistent "
        f"with neurological tremor rather than raw drawing speed or shape.\n\n"
        f"Patient-grouped 5-fold cross-validation during training: "
        f"**{metadata['cv_accuracy_mean']:.1%}** accuracy, "
        f"**{metadata['cv_recall_parkinson_mean']:.1%}** Parkinson recall, "
        f"**{metadata['cv_roc_auc_mean']:.3f}** ROC-AUC — measured on the original "
        f"tablet data, not on browser drawings like this one."
    )
