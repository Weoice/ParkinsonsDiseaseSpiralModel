import numpy as np
import pandas as pd

# GripAngle intentionally excluded: no honest browser-equivalent signal
# exists (stylus tilt/twist support is inconsistent across devices and
# browsers, and a plain mouse reports none at all).
AGG_FUNCS = {
    "Velocity": ["mean", "std", "max"],
    "Acceleration": ["mean", "std", "max"],
    "Jerk": ["mean", "std", "max"],
    "Pressure": ["mean", "std"],
    "Pressure_RollStd": ["mean", "max"],
}

# Jerk needs 3 successive diffs and the rolling pressure-std needs a
# handful of samples to be meaningful - below this many surviving rows,
# aggregated stats are noise, not signal.
MIN_KINEMATIC_ROWS = 10


def engineer_kinematics(group: pd.DataFrame) -> pd.DataFrame:
    """Per-point Velocity/Acceleration/Jerk and rolling pressure-std for one
    continuous pen trajectory. Caller must group by whatever column(s)
    define "one continuous trajectory" (e.g. (Patient_ID, Test_ID) for the
    training recordings, or a stroke id for a live browser drawing) - diffs
    must never cross that boundary."""
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

    # Sampling interval in the training dataset is a near-constant ~7
    # time-units per sample, so derive a ~50-unit window in samples from
    # each group's own median dt rather than hardcoding a sample count.
    median_dt = dt.median()
    window = max(3, int(round(50 / median_dt))) if median_dt and median_dt > 0 else 5
    group["Pressure_RollStd"] = group["Pressure"].rolling(window=window, min_periods=2).std()

    return group


def compute_point_kinematics(df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    """groupby(group_cols) + engineer_kinematics, then drop rows left
    unusable by diff()-of-diff()-of-diff() NaNs (first few rows of every
    group) or dt<=0 segment-boundary infs."""
    out = df.groupby(group_cols, group_keys=False).apply(engineer_kinematics)
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.dropna(subset=["Velocity", "Acceleration", "Jerk", "Pressure_RollStd"])


def aggregate_feature_table(df: pd.DataFrame, group_cols: list, agg_funcs: dict = AGG_FUNCS) -> pd.DataFrame:
    """Collapse point-level kinematics into one summary row per group,
    using the f'{base}_{stat}' column naming convention."""
    table = df.groupby(group_cols).agg(agg_funcs)
    table.columns = ["_".join(c) for c in table.columns]
    return table.reset_index()


def feature_column_names(agg_funcs: dict = AGG_FUNCS) -> list:
    return [f"{base}_{stat}" for base, stats in agg_funcs.items() for stat in stats]
