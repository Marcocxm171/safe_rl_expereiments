"""
original_hoflon_reward.py
=========================

Post-process the NEW physical/unscaled PID dataset into the ORIGINAL HOFLON
representation and calculate the ORIGINAL HOFLON reward.

This file DOES NOT simulate the reactor. It only transforms an already-generated
physical PID trajectory CSV/DataFrame.

Expected raw columns
--------------------
run_id, step, time_min,
A, I, P, R, T,
err_P, err_T,
int_err_P, int_err_T,
f_I, T_c

Original HOFLON scaling reproduced here
---------------------------------------
Concentration states:
    A_s = A / 100
    I_s = I / 100
    P_s = P / 100
    R_s = R / 100

Temperature:
    T_s = T / 350

Tracking errors:
    err_P_s = 1 - P_s
    err_T_s = 1 - T_s

Integral-error features:
The original controller integrated the SCALED errors and then stored /1000.
The new raw generator integrates PHYSICAL errors, therefore the exact coordinate
conversion is:
    int_err_P_HOFLON = int_err_P_phys / (100 * 1000)
    int_err_T_HOFLON = int_err_T_phys / (350 * 1000)

Actions:
    u_I  = f_I / 2.5
    u_Tc = (T_c - 280) / 120

Original reward
---------------
Let
    E_t = (P_s - 1)^2 + (T_s - 1)^2
    dE  = E_{t-1} - E_t

Then
    proximity = -E_t + exp(-E_t / sigma^2)
    progress  = dE

    reward = 50 * proximity + 5000 * progress

The original code calculated a control-change penalty but multiplied it by 0,
so it has no numerical effect and is omitted from the final expression here.

For the first row of every run, the original generator used the nominal initial
scaled state P=0, T=1 as the previous state. This implementation reproduces
that convention.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Original HOFLON constants
# ---------------------------------------------------------------------------

S_CONC = 100.0
S_TEMP = 350.0

P_SETPOINT_SCALED = 1.0
T_SETPOINT_SCALED = 1.0

FI_MIN = 0.0
FI_MAX = 2.5
TC_MIN = 280.0
TC_MAX = 400.0

DEFAULT_SIGMA = 0.05
PROXIMITY_WEIGHT = 50.0
PROGRESS_WEIGHT = 5_000.0

# Original nominal scaled initial state had P=0 and T=1.
INITIAL_P_SCALED = 0.0
INITIAL_T_SCALED = 1.0


REQUIRED_RAW_COLUMNS = {
    "run_id",
    "step",
    "time_min",
    "A",
    "I",
    "P",
    "R",
    "T",
    "err_P",
    "err_T",
    "int_err_P",
    "int_err_T",
    "f_I",
    "T_c",
}


OUTPUT_COLUMNS = [
    "run_id",
    "step",
    "time_min",
    "A",
    "I",
    "P",
    "R",
    "T",
    "err_P",
    "err_T",
    "int_err_P",
    "int_err_T",
    "u_I",
    "u_Tc",
    "reward",
]


def compute_original_hoflon_reward(
    P_scaled: np.ndarray,
    T_scaled: np.ndarray,
    prev_P_scaled: np.ndarray,
    prev_T_scaled: np.ndarray,
    *,
    sigma: float = DEFAULT_SIGMA,
) -> np.ndarray:
    """Vectorised implementation of the original HOFLON reward."""

    if sigma <= 0.0:
        raise ValueError("sigma must be positive.")

    err_now = (
        (P_scaled - P_SETPOINT_SCALED) ** 2
        + (T_scaled - T_SETPOINT_SCALED) ** 2
    )

    err_prev = (
        (prev_P_scaled - P_SETPOINT_SCALED) ** 2
        + (prev_T_scaled - T_SETPOINT_SCALED) ** 2
    )

    delta_err = err_prev - err_now

    proximity = -err_now + np.exp(-err_now / sigma**2)
    progress = delta_err

    return (
        PROXIMITY_WEIGHT * proximity
        + PROGRESS_WEIGHT * progress
    )


def apply_original_hoflon(
    data: Union[str, Path, pd.DataFrame],
    *,
    output_csv: str | Path | None = None,
    sigma: float = DEFAULT_SIGMA,
) -> pd.DataFrame:
    """
    Convert raw physical PID data into the original HOFLON scaled format and
    calculate the original HOFLON reward.

    Parameters
    ----------
    data:
        Path to the raw physical CSV OR an existing pandas DataFrame.

    output_csv:
        Optional path for saving the transformed dataset.
        If None, nothing is written.

    sigma:
        Original proximity width. Defaults to 0.05.

    Returns
    -------
    pandas.DataFrame
        Original-HOFLON-style scaled dataset with reward.
    """

    if isinstance(data, (str, Path)):
        raw = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        raw = data.copy()
    else:
        raise TypeError("data must be a CSV path or pandas DataFrame.")

    missing = REQUIRED_RAW_COLUMNS - set(raw.columns)
    if missing:
        raise ValueError(
            "Input data is missing required columns: "
            f"{sorted(missing)}"
        )

    # Explicit chronological ordering within each trajectory.
    raw = (
        raw
        .sort_values(["run_id", "step"])
        .reset_index(drop=True)
    )

    df = pd.DataFrame(index=raw.index)

    # IDs / timing remain untouched.
    df["run_id"] = raw["run_id"].astype(int)
    df["step"] = raw["step"].astype(int)
    df["time_min"] = raw["time_min"].astype(float)

    # -----------------------------------------------------------------------
    # 1. Original state scaling
    # -----------------------------------------------------------------------
    df["A"] = raw["A"] / S_CONC
    df["I"] = raw["I"] / S_CONC
    df["P"] = raw["P"] / S_CONC
    df["R"] = raw["R"] / S_CONC
    df["T"] = raw["T"] / S_TEMP

    # Recalculate scaled tracking errors from the scaled states.
    # This avoids compounding any rounding in the raw error columns.
    df["err_P"] = P_SETPOINT_SCALED - df["P"]
    df["err_T"] = T_SETPOINT_SCALED - df["T"]

    # -----------------------------------------------------------------------
    # 2. Reproduce original integral-error feature coordinates
    # -----------------------------------------------------------------------
    # New generator stores physical-error integrals [physical error * seconds].
    # Old generator integrated scaled error, then divided the integral by 1000.
    df["int_err_P"] = raw["int_err_P"] / (S_CONC * 1000.0)
    df["int_err_T"] = raw["int_err_T"] / (S_TEMP * 1000.0)

    # -----------------------------------------------------------------------
    # 3. Original action scaling
    # -----------------------------------------------------------------------
    df["u_I"] = (raw["f_I"] - FI_MIN) / (FI_MAX - FI_MIN)
    df["u_Tc"] = (raw["T_c"] - TC_MIN) / (TC_MAX - TC_MIN)

    # -----------------------------------------------------------------------
    # 4. Previous state within each trajectory
    # -----------------------------------------------------------------------
    prev_P = df.groupby("run_id", sort=False)["P"].shift(1)
    prev_T = df.groupby("run_id", sort=False)["T"].shift(1)

    first_row = df.groupby("run_id", sort=False).cumcount().eq(0)
    prev_P.loc[first_row] = INITIAL_P_SCALED
    prev_T.loc[first_row] = INITIAL_T_SCALED

    # -----------------------------------------------------------------------
    # 5. Original HOFLON reward
    # -----------------------------------------------------------------------
    df["reward"] = compute_original_hoflon_reward(
        P_scaled=df["P"].to_numpy(dtype=float),
        T_scaled=df["T"].to_numpy(dtype=float),
        prev_P_scaled=prev_P.to_numpy(dtype=float),
        prev_T_scaled=prev_T.to_numpy(dtype=float),
        sigma=sigma,
    )

    df = df[OUTPUT_COLUMNS]

    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"Saved original-HOFLON processed dataset to: {output_csv.resolve()}")

    return df


def rank_pid_runs(
    processed_df: pd.DataFrame,
    *,
    top_n: int = 20,
) -> pd.DataFrame:
    """Rank PID trajectories by highest cumulative original-HOFLON reward."""

    required = {"run_id", "reward"}
    missing = required - set(processed_df.columns)
    if missing:
        raise ValueError(
            f"processed_df is missing required columns: {sorted(missing)}"
        )

    if top_n < 1:
        raise ValueError("top_n must be >= 1.")

    ranking = (
        processed_df
        .groupby("run_id", as_index=False)["reward"]
        .sum()
        .rename(columns={"reward": "cumulative_reward"})
        .sort_values("cumulative_reward", ascending=False)
        .reset_index(drop=True)
    )

    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))

    return ranking.head(top_n).copy()


if __name__ == "__main__":
    # Example standalone usage.
    processed = apply_original_hoflon(
        "unscaled_pid_dataset.csv",
        output_csv="original_hoflon_processed.csv",
    )

    top20 = rank_pid_runs(processed, top_n=20)
    print(top20.to_string(index=False))
