"""
idea4_mse_change_reward.py
==========================

Idea 4:
    Use the fixed Idea 2 scaling.
    Reward only:

        E_t     = err_P_t^2 + err_T_t^2
        E_next  = err_P_(t+1)^2 + err_T_(t+1)^2
        reward_t = E_t - E_next

Positive reward means tracking improved over the transition.

The final row of every run is removed because there is no next state.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

S_CONC = 100.0
P_SETPOINT_PHYS = 100.0
T_SETPOINT_PHYS = 350.0

FI_MIN = 0.0
FI_MAX = 2.5
TC_MIN = 280.0
TC_MAX = 400.0


def _load_dataframe(data: str | Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()

    path = Path(data)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    return pd.read_csv(path)


def _apply_idea2_scaling(raw_data: str | Path | pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Apply the fixed Idea 2 scaling.

    Unchanged from Idea 2:
        A, I, P, R -> /100
        T -> dataset min-max normalization
        int_err_P -> original HOFLON concentration-integral convention
        int_err_T -> scaled-temperature integral / max absolute value
        f_I -> [0,1]
        T_c -> [0,1]
    """
    raw = _load_dataframe(raw_data)

    required = {
        "run_id", "step", "time_min",
        "A", "I", "P", "R", "T",
        "err_P", "err_T",
        "int_err_P", "int_err_T",
        "f_I", "T_c",
    }

    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(
            f"Raw PID data is missing required columns: {sorted(missing)}"
        )

    raw = (
        raw
        .sort_values(["run_id", "step"], kind="stable")
        .reset_index(drop=True)
    )

    df = raw.copy()

    # Concentration states: original HOFLON scaling.
    df["A"] = raw["A"] / S_CONC
    df["I"] = raw["I"] / S_CONC
    df["P"] = raw["P"] / S_CONC
    df["R"] = raw["R"] / S_CONC

    p_setpoint_scaled = P_SETPOINT_PHYS / S_CONC  # 1.0

    # Temperature: Idea 2 dataset-based min-max scaling.
    t_min = float(raw["T"].min())
    t_max = float(raw["T"].max())
    t_range = t_max - t_min

    if not np.isfinite(t_range) or t_range <= 0:
        raise ValueError(
            f"Temperature range must be positive. Got "
            f"T_min={t_min}, T_max={t_max}."
        )

    df["T"] = (raw["T"] - t_min) / t_range
    t_setpoint_scaled = (T_SETPOINT_PHYS - t_min) / t_range

    # Scaled tracking errors.
    df["err_P"] = p_setpoint_scaled - df["P"]
    df["err_T"] = t_setpoint_scaled - df["T"]

    # Polymer integral error: original HOFLON convention.
    df["int_err_P"] = raw["int_err_P"] / S_CONC / 1000.0

    # Temperature integral error: Idea 2 convention.
    int_t_scaled = raw["int_err_T"] / t_range
    max_abs_int_t = float(np.max(np.abs(int_t_scaled.to_numpy(dtype=float))))

    if not np.isfinite(max_abs_int_t) or max_abs_int_t <= 0:
        raise ValueError(
            "Maximum absolute scaled temperature integral error must be positive."
        )

    df["int_err_T"] = int_t_scaled / max_abs_int_t

    # Actions: original HOFLON [0,1] domain.
    df["u_I"] = (
        (raw["f_I"] - FI_MIN) / (FI_MAX - FI_MIN)
    ).clip(0.0, 1.0)

    df["u_Tc"] = (
        (raw["T_c"] - TC_MIN) / (TC_MAX - TC_MIN)
    ).clip(0.0, 1.0)

    scaling = {
        "t_min": t_min,
        "t_max": t_max,
        "t_range": t_range,
        "p_setpoint_scaled": p_setpoint_scaled,
        "t_setpoint_scaled": t_setpoint_scaled,
        "max_abs_int_t_scaled": max_abs_int_t,
    }

    return df, scaling


def rank_pid_runs(
    processed_data: str | Path | pd.DataFrame,
    top_n: int = 20,
) -> pd.DataFrame:
    """Rank runs by cumulative reward. Higher is better."""
    df = _load_dataframe(processed_data)

    required = {"run_id", "reward"}
    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"Processed data is missing required columns: {sorted(missing)}"
        )

    ranking = (
        df.groupby("run_id", as_index=False)["reward"]
        .sum()
        .rename(columns={"reward": "cumulative_reward"})
        .sort_values("cumulative_reward", ascending=False)
        .reset_index(drop=True)
    )

    ranking.insert(
        0,
        "rank",
        np.arange(1, len(ranking) + 1),
    )

    return ranking.head(int(top_n)).copy()


def apply_idea4_mse_change_reward(
    raw_data: str | Path | pd.DataFrame,
    output_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    Apply Idea 2 scaling and Idea 4 reward:

        reward_t = E_t - E_(t+1)

    This aligns reward_t with the transition:
        state_t, action_t -> state_(t+1)
    """

    df, scaling = _apply_idea2_scaling(raw_data)

    # Current scaled tracking squared error.
    df["tracking_mse"] = (
        df["err_P"] ** 2
        + df["err_T"] ** 2
    )

    # Next-state tracking error within the same run.
    df["next_tracking_mse"] = (
        df.groupby("run_id")["tracking_mse"]
        .shift(-1)
    )

    # Correctly aligned progress reward.
    df["reward"] = (
        df["tracking_mse"]
        - df["next_tracking_mse"]
    )

    # No next state for final row of each trajectory.
    df = (
        df
        .dropna(subset=["next_tracking_mse"])
        .reset_index(drop=True)
    )

    final_columns = [
        "run_id", "step", "time_min",
        "A", "I", "P", "R", "T",
        "err_P", "err_T",
        "int_err_P", "int_err_T",
        "u_I", "u_Tc",
        "tracking_mse",
        "next_tracking_mse",
        "reward",
    ]

    processed = df[final_columns].copy()

    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        processed.to_csv(output_csv, index=False)

    print("Idea 4: change-in-MSE reward")
    print(f"  T_min             = {scaling['t_min']:.8f} K")
    print(f"  T_max             = {scaling['t_max']:.8f} K")
    print(f"  T_setpoint_scaled = {scaling['t_setpoint_scaled']:.8f}")
    print("  reward            = E_t - E_(t+1)")
    print("  final row/run     = dropped")

    return processed
