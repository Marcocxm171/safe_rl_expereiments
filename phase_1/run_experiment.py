"""
run_experiment.py
=================

Generic HOFLON controller evaluator.

Any controller can be evaluated as long as it is wrapped as:

    action = control_fn(obs)

where:
    obs    = 9D Idea 3 observation
    action = [u_I, u_Tc] in [0, 1]^2

Outputs:
1. Saves one trajectory plot as PDF in plots/
2. Appends model_name and total_integral_error to
   trajectories/control_performance.csv

No printing is performed.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from poly_reactor_env import PolymerisationEnv


DEFAULT_PLOTS_DIR = Path("plots")
DEFAULT_RESULTS_CSV = Path("trajectories/control_performance.csv")

P_SETPOINT = 100.0
T_SETPOINT = 350.0


def _safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_")
    return cleaned or "model"


def _append_result(
    csv_path: Path,
    model_name: str,
    total_integral_error: float,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = csv_path.exists()

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(
                [
                    "model_name",
                    "total_integral_error",
                ]
            )

        writer.writerow(
            [
                model_name,
                float(total_integral_error),
            ]
        )


def run_experiment(
    control_fn: Callable[[np.ndarray], np.ndarray],
    model_name: str,
    *,
    seed: int = 11,
    plots_dir: str | Path = DEFAULT_PLOTS_DIR,
    results_csv: str | Path = DEFAULT_RESULTS_CSV,
    env_kwargs: dict | None = None,
) -> None:
    """
    Run one controller rollout in the settled Idea 3 environment.

    control_fn must satisfy:

        action = control_fn(obs)

    where action is [u_I, u_Tc].

    ISE:
        integral(err_P_scaled^2 + err_T_scaled^2) dt

    The errors are taken from s_(t+1), matching the settled
    Idea 3 transition/reward convention.
    """

    if not callable(control_fn):
        raise TypeError("control_fn must be callable.")

    env_kwargs = {} if env_kwargs is None else dict(env_kwargs)
    env_kwargs.pop("seed", None)

    env = PolymerisationEnv(
        seed=seed,
        **env_kwargs,
    )

    obs, info = env.reset(seed=seed)

    terminated = False
    truncated = False

    dt_hours = float(env.dt) / 3600.0
    total_ise = 0.0

    time_hours = [float(info["time_s"]) / 3600.0]
    P_values = [float(info["P"])]
    T_values = [float(info["T"])]

    action_time_hours = []
    u_I_values = []
    u_Tc_values = []

    while not (terminated or truncated):
        action = np.asarray(
            control_fn(
                np.asarray(obs, dtype=np.float32).copy()
            ),
            dtype=np.float32,
        ).reshape(-1)

        if action.size != 2:
            raise ValueError(
                "control_fn must return exactly [u_I, u_Tc]."
            )

        if not np.all(np.isfinite(action)):
            raise ValueError(
                f"control_fn returned non-finite action: {action}"
            )

        action = np.clip(action, 0.0, 1.0)

        action_time_hours.append(
            float(info["time_s"]) / 3600.0
        )
        u_I_values.append(float(action[0]))
        u_Tc_values.append(float(action[1]))

        next_obs, _, terminated, truncated, next_info = env.step(action)

        err_p_next = float(next_obs[5])
        err_t_next = float(next_obs[6])

        total_ise += (
            err_p_next**2
            + err_t_next**2
        ) * dt_hours

        time_hours.append(
            float(next_info["time_s"]) / 3600.0
        )
        P_values.append(float(next_info["P"]))
        T_values.append(float(next_info["T"]))

        obs = next_obs
        info = next_info

    if hasattr(env, "close"):
        env.close()

    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    safe_name = _safe_filename(model_name)
    plot_path = plots_dir / f"{safe_name}.pdf"

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11, 7),
    )

    axes[0, 0].plot(time_hours, P_values)
    axes[0, 0].axhline(P_SETPOINT, linestyle="--")
    axes[0, 0].set_xlabel("Time [h]")
    axes[0, 0].set_ylabel("Polymer P")
    axes[0, 0].set_title("Polymer tracking")
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(time_hours, T_values)
    axes[0, 1].axhline(T_SETPOINT, linestyle="--")
    axes[0, 1].set_xlabel("Time [h]")
    axes[0, 1].set_ylabel("Temperature [K]")
    axes[0, 1].set_title("Temperature tracking")
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].step(
        action_time_hours,
        u_I_values,
        where="post",
    )
    axes[1, 0].set_xlabel("Time [h]")
    axes[1, 0].set_ylabel("u_I")
    axes[1, 0].set_ylim(-0.05, 1.05)
    axes[1, 0].set_title("Initiator action")
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].step(
        action_time_hours,
        u_Tc_values,
        where="post",
    )
    axes[1, 1].set_xlabel("Time [h]")
    axes[1, 1].set_ylabel("u_Tc")
    axes[1, 1].set_ylim(-0.05, 1.05)
    axes[1, 1].set_title("Coolant action")
    axes[1, 1].grid(alpha=0.3)

    fig.suptitle(
        f"{model_name} | ISE = {total_ise:.6f}"
    )

    fig.tight_layout()
    fig.savefig(
        plot_path,
        format="pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

    _append_result(
        csv_path=Path(results_csv),
        model_name=model_name,
        total_integral_error=total_ise,
    )
