"""
reward_idea_evaluator_v2.py
===========================

Evaluate the PID trajectories selected by each reward-function idea.

For every selected PID run, calculate in PHYSICAL units:

    ISE = integral [ (P - P_sp)^2 + (T - T_sp)^2 ] dt

and over the final N hours:

    final_Nh_mse = mean [ (P - P_sp)^2 + (T - T_sp)^2 ]

Each selected trajectory gets its own row in one shared CSV:

    idea_name, run_id, ise, final_10h_mse

So if each reward idea selects 20 PID trajectories, each idea contributes
20 rows to the comparison CSV.

Re-running the same idea_name replaces all previous rows for that idea.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


RESULT_COLUMNS = [
    "idea_name",
    "run_id",
    "ise",
    "final_10h_mse",
]


def _load_dataframe(data: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Load a CSV or return a defensive copy of an existing DataFrame."""
    if isinstance(data, pd.DataFrame):
        return data.copy()

    path = Path(data)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    return pd.read_csv(path)


def evaluate_reward_idea(
    raw_data: str | Path | pd.DataFrame,
    run_ids: Iterable[int],
    idea_name: str,
    results_csv: str | Path = "reward_idea_performance.csv",
    *,
    p_setpoint: float = 100.0,
    t_setpoint: float = 350.0,
    final_hours: float = 10.0,
    replace_existing: bool = True,
) -> pd.DataFrame:
    """
    Evaluate the PID trajectories selected by one reward-function idea.

    Returns one row per selected PID trajectory:
        idea_name, run_id, ise, final_10h_mse
    """

    idea_name = str(idea_name).strip()
    if not idea_name:
        raise ValueError("idea_name must be a non-empty string.")

    if final_hours <= 0:
        raise ValueError("final_hours must be positive.")

    df = _load_dataframe(raw_data)

    required = {"run_id", "time_min", "P", "T"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"Raw PID data is missing required columns: {sorted(missing)}"
        )

    # Preserve supplied ordering while removing duplicates.
    run_ids = list(dict.fromkeys(int(run_id) for run_id in run_ids))
    if not run_ids:
        raise ValueError("run_ids must contain at least one PID run ID.")

    available_ids = set(df["run_id"].astype(int).unique())
    missing_ids = [run_id for run_id in run_ids if run_id not in available_ids]
    if missing_ids:
        raise ValueError(
            "The following run IDs are not present in the raw PID data: "
            f"{missing_ids}"
        )

    rows = []

    for run_id in run_ids:
        run = (
            df[df["run_id"].astype(int) == run_id]
            .sort_values("time_min", kind="stable")
            .copy()
        )

        if len(run) < 2:
            raise ValueError(
                f"Run {run_id} has fewer than two samples; ISE cannot be calculated."
            )

        time_hours = run["time_min"].to_numpy(dtype=float) / 60.0
        err_p = run["P"].to_numpy(dtype=float) - float(p_setpoint)
        err_t = run["T"].to_numpy(dtype=float) - float(t_setpoint)
        squared_tracking_error = err_p**2 + err_t**2

        # Full-trajectory integral squared error.
        ise = float(
            np.trapezoid(
                squared_tracking_error,
                x=time_hours,
            )
        )

        # MSE over the final N hours of this trajectory.
        final_time = float(time_hours[-1])
        window_start = final_time - float(final_hours)
        final_mask = time_hours >= window_start

        if not np.any(final_mask):
            raise ValueError(
                f"Run {run_id} has no samples in the final {final_hours:g} hours."
            )

        final_mse = float(np.mean(squared_tracking_error[final_mask]))

        rows.append(
            {
                "idea_name": idea_name,
                "run_id": int(run_id),
                "ise": ise,
                "final_10h_mse": final_mse,
            }
        )

    idea_results = pd.DataFrame(rows, columns=RESULT_COLUMNS)

    results_csv = Path(results_csv)
    results_csv.parent.mkdir(parents=True, exist_ok=True)

    if results_csv.exists():
        existing = pd.read_csv(results_csv)

        missing_result_cols = set(RESULT_COLUMNS).difference(existing.columns)
        if missing_result_cols:
            raise ValueError(
                "Existing results CSV has incompatible columns. "
                f"Expected {RESULT_COLUMNS}; missing {sorted(missing_result_cols)}."
            )

        existing = existing[RESULT_COLUMNS].copy()

        if replace_existing:
            existing = existing[
                existing["idea_name"].astype(str) != idea_name
            ]

        combined = pd.concat([existing, idea_results], ignore_index=True)
    else:
        combined = idea_results.copy()

    combined.to_csv(results_csv, index=False)

    print(
        f"Saved {len(idea_results)} trajectories for '{idea_name}' "
        f"to {results_csv}"
    )

    return idea_results
