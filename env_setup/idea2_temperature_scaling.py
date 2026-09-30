"""
idea2_temperature_scaling.py
============================

Reward Idea 2:
    Keep the original HOFLON reward and all original concentration/action
    scaling, but change ONLY:

    1. Temperature scaling:
           T_scaled = (T - T_min) / (T_max - T_min)

       where T_min and T_max are taken from the supplied raw PID dataset.

    2. Temperature integral-error feature:
           first convert the physical integral error into the new scaled
           temperature coordinates, then divide by the maximum absolute
           scaled temperature integral error across the dataset.

Everything else remains as in original HOFLON:
    A, I, P, R -> divide by 100
    concentration integral error -> original HOFLON convention
    f_I -> [0, 1] using [0, 2.5]
    T_c -> [0, 1] using [280, 400]
    same proximity + progress reward

Expected raw columns:
    run_id, step, time_min,
    A, I, P, R, T,
    err_P, err_T,
    int_err_P, int_err_T,
    f_I, T_c

The raw input dataframe is never modified in place.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Original HOFLON constants retained
# ---------------------------------------------------------------------------

S_CONC = 100.0

P_SETPOINT_PHYS = 100.0
T_SETPOINT_PHYS = 350.0

FI_MIN = 0.0
FI_MAX = 2.5

TC_MIN = 280.0
TC_MAX = 400.0

SIGMA = 0.05


def _load_dataframe(
    data: str | Path | pd.DataFrame,
) -> pd.DataFrame:
    """Load a CSV or return a defensive copy of a DataFrame."""
    if isinstance(data, pd.DataFrame):
        return data.copy()

    path = Path(data)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    return pd.read_csv(path)


def apply_idea2_temperature_scaling(
    raw_data: str | Path | pd.DataFrame,
    output_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    Apply Reward Idea 2 to the raw physical PID dataset.

    Returns a processed dataframe containing the transformed observations,
    transformed actions, and per-timestep reward.

    Notes
    -----
    Reward uses the SAME original HOFLON equation:

        E_t = (P_t - P_sp)^2 + (T_t - T_sp)^2

        proximity_t = -E_t + exp(-E_t / sigma^2)

        progress_t = E_(t-1) - E_t

        reward_t = 50 * proximity_t + 5000 * progress_t

    The control penalty remains numerically zero, matching the original
    implementation.
    """

    raw = _load_dataframe(raw_data)

    required = {
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

    # -----------------------------------------------------------------------
    # 1. ORIGINAL concentration scaling
    # -----------------------------------------------------------------------

    df["A"] = raw["A"] / S_CONC
    df["I"] = raw["I"] / S_CONC
    df["P"] = raw["P"] / S_CONC
    df["R"] = raw["R"] / S_CONC

    p_setpoint_scaled = P_SETPOINT_PHYS / S_CONC  # = 1.0

    # -----------------------------------------------------------------------
    # 2. NEW dataset-based temperature scaling
    # -----------------------------------------------------------------------

    t_min = float(raw["T"].min())
    t_max = float(raw["T"].max())
    t_range = t_max - t_min

    if not np.isfinite(t_range) or t_range <= 0:
        raise ValueError(
            f"Temperature range must be positive. "
            f"Got T_min={t_min}, T_max={t_max}."
        )

    df["T"] = (
        raw["T"] - t_min
    ) / t_range

    t_setpoint_scaled = (
        T_SETPOINT_PHYS - t_min
    ) / t_range

    # -----------------------------------------------------------------------
    # 3. Error columns in the transformed coordinates
    # -----------------------------------------------------------------------

    df["err_P"] = (
        p_setpoint_scaled
        - df["P"]
    )

    df["err_T"] = (
        t_setpoint_scaled
        - df["T"]
    )

    # -----------------------------------------------------------------------
    # 4. Integral-error features
    # -----------------------------------------------------------------------
    #
    # Raw generator stores physical integral errors:
    #
    #     int_err_P_phys = integral (100 - P_phys) dt
    #     int_err_T_phys = integral (350 - T_phys) dt
    #
    # Original HOFLON concentration error used:
    #
    #     e_P_scaled = (100 - P_phys) / 100
    #
    # and then logged:
    #
    #     int_err_P_feature = integral(e_P_scaled dt) / 1000
    #
    # Therefore:
    #
    #     raw_int_P / 100 / 1000
    #
    # -----------------------------------------------------------------------

    df["int_err_P"] = (
        raw["int_err_P"]
        / S_CONC
        / 1000.0
    )

    # New scaled temperature error:
    #
    #     e_T_scaled = (350 - T_phys) / (T_max - T_min)
    #
    # Convert physical integral error into those coordinates.
    int_t_scaled = (
        raw["int_err_T"]
        / t_range
    )

    max_abs_int_t = float(
        np.max(
            np.abs(
                int_t_scaled.to_numpy(dtype=float)
            )
        )
    )

    if (
        not np.isfinite(max_abs_int_t)
        or max_abs_int_t <= 0
    ):
        raise ValueError(
            "Maximum absolute scaled temperature integral error "
            "must be positive."
        )

    df["int_err_T"] = (
        int_t_scaled
        / max_abs_int_t
    )

    # -----------------------------------------------------------------------
    # 5. ORIGINAL HOFLON action scaling -> both remain in [0, 1]
    # -----------------------------------------------------------------------

    df["u_I"] = (
        raw["f_I"] - FI_MIN
    ) / (FI_MAX - FI_MIN)

    df["u_Tc"] = (
        raw["T_c"] - TC_MIN
    ) / (TC_MAX - TC_MIN)

    # Keep numerical clipping consistent with the physical actuator bounds.
    df["u_I"] = df["u_I"].clip(0.0, 1.0)
    df["u_Tc"] = df["u_Tc"].clip(0.0, 1.0)

    # -----------------------------------------------------------------------
    # 6. SAME ORIGINAL HOFLON reward
    # -----------------------------------------------------------------------

    tracking_error = (
        (df["P"] - p_setpoint_scaled) ** 2
        + (df["T"] - t_setpoint_scaled) ** 2
    )

    # Previous tracking error within each trajectory.
    previous_error = (
        tracking_error
        .groupby(df["run_id"])
        .shift(1)
    )

    # For the first sample of each trajectory, original HOFLON effectively
    # compares against the known initial setpoint error:
    # P_initial = 0, T_initial = 350 K.
    initial_p_scaled = 0.0
    initial_t_scaled = (
        T_SETPOINT_PHYS - t_min
    ) / t_range

    initial_error = (
        (initial_p_scaled - p_setpoint_scaled) ** 2
        + (initial_t_scaled - t_setpoint_scaled) ** 2
    )

    previous_error = previous_error.fillna(initial_error)

    progress = (
        previous_error
        - tracking_error
    )

    proximity = (
        -tracking_error
        + np.exp(
            -tracking_error
            / (SIGMA ** 2)
        )
    )

    df["reward"] = (
        50.0 * proximity
        + 5000.0 * progress
    )

    # -----------------------------------------------------------------------
    # 7. Return a clean processed dataset
    # -----------------------------------------------------------------------

    final_columns = [
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

    processed = df[
        final_columns
    ].copy()

    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        processed.to_csv(
            output_csv,
            index=False,
        )

    # Useful values printed once so the exact transformation is auditable.
    print("Idea 2 temperature scaling:")
    print(f"  T_min                = {t_min:.8f} K")
    print(f"  T_max                = {t_max:.8f} K")
    print(f"  T_range              = {t_range:.8f} K")
    print(f"  T_setpoint_scaled    = {t_setpoint_scaled:.8f}")
    print(f"  max_abs_int_T_scaled = {max_abs_int_t:.8f}")
    print(
        f"  u_I range            = "
        f"[{processed['u_I'].min():.6f}, "
        f"{processed['u_I'].max():.6f}]"
    )
    print(
        f"  u_Tc range           = "
        f"[{processed['u_Tc'].min():.6f}, "
        f"{processed['u_Tc'].max():.6f}]"
    )

    return processed


def rank_pid_runs(
    processed_data: str | Path | pd.DataFrame,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Rank PID trajectories by cumulative reward.

    Higher cumulative reward = better according to this reward idea.
    """

    df = _load_dataframe(
        processed_data
    )

    required = {
        "run_id",
        "reward",
    }

    missing = required.difference(
        df.columns
    )

    if missing:
        raise ValueError(
            f"Processed data is missing required columns: "
            f"{sorted(missing)}"
        )

    ranking = (
        df.groupby(
            "run_id",
            as_index=False,
        )["reward"]
        .sum()
        .rename(
            columns={
                "reward":
                "cumulative_reward"
            }
        )
        .sort_values(
            "cumulative_reward",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    ranking.insert(
        0,
        "rank",
        np.arange(
            1,
            len(ranking) + 1,
        ),
    )

    return (
        ranking
        .head(int(top_n))
        .copy()
    )
