"""
physical_pid_data_generation.py
===============================

Standalone PID trajectory generator for the polymerisation CSTR.

IMPORTANT DESIGN RULE
---------------------
This file contains NO RL/state/action scaling whatsoever.

Everything is expressed in physical units:
    ODE state       [A, I, R, P] : physical concentration units
                    T            : K
    PID setpoints   P            : 100.0
                    T            : 350.0 K
    PID errors      e_P          : P_sp - P
                    e_T          : T_sp - T
    PID integrals   error * seconds
    actions         f_I          : kg h^-1
                    T_c          : K

The only clipping is physical actuator clipping:
    0 <= f_I <= 2.5 kg h^-1
    280 <= T_c <= 400 K

Reward construction is deliberately NOT included. Candidate rewards should be
computed later from this fixed PID dataset so every reward is evaluated on the
same trajectories.

The CSV stores only the noisy measurements used by the PID controller:
    A, I, P, R, T

The noise-free internal plant state is used only for numerical integration and
is not written to the dataset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


# ===========================================================================
# 1. PHYSICAL PLANT PARAMETERS
# ===========================================================================

R_GAS = 8.314
RHO = 1_000.0
CP = 2_000.0
UA = 550.0
V = 1.0

W_A = 100.0
W_S = 80.0
T_FEED = 350.0

MW_A = 0.100
MW_I = 0.165

A_D = 1e9
E_D = 125e3
A_P = 4e4
E_P = 25e3
A_T = 1e6
E_T = 15e3

DELTA_H_RXN = 100e3


# ===========================================================================
# 2. PHYSICAL ACTUATOR LIMITS AND SETPOINTS
# ===========================================================================

F_I_MIN = 0.0
F_I_MAX = 2.5          # kg h^-1
TC_MIN = 280.0         # K
TC_MAX = 400.0         # K

DEFAULT_SETPOINTS = {
    "P": 100.0,
    "T": 350.0,
}


# ===========================================================================
# 3. PHYSICAL INITIAL CONDITION AND MEASUREMENT NOISE
# ===========================================================================

# Physical equivalent of the old scaled initial condition
# [5.55555456, 0, 0, 0, 1].
# ODE state order is [A, I, R, P, T].
DEFAULT_Y0 = np.array(
    [555.555456, 0.0, 0.0, 0.0, 350.0],
    dtype=np.float64,
)

# Physical equivalent of the previous measurement noise.
# Order: [A, I, R, P, T].
DEFAULT_NOISE_STD = np.array(
    [2.0, 0.0, 0.0, 0.5, 0.0005714 * 350.0],
    dtype=np.float64,
)


# ===========================================================================
# 4. PHYSICAL REACTOR MODEL -- NO SCALING
# ===========================================================================

def arrhenius(pre_exponential: float, activation_energy: float, T: float) -> float:
    return pre_exponential * np.exp(
        -activation_energy / (R_GAS * T)
    )


def reactor_odes(
    t: float,
    y: np.ndarray,
    f_I: float,
    T_c: float,
) -> np.ndarray:
    """
    Physical reactor equations.

    Parameters
    ----------
    y : array-like
        [A, I, R, P, T] in physical units.
    f_I : float
        Initiator feed [kg h^-1].
    T_c : float
        Coolant temperature [K].

    Returns
    -------
    np.ndarray
        [dA/dt, dI/dt, dR/dt, dP/dt, dT/dt] in physical coordinates.
    """
    del t  # autonomous system

    A, I, R_rad, P, T = map(float, y)

    # Volumetric flow [m^3 s^-1] and dilution rate [s^-1].
    F = (f_I + W_A + W_S) / (RHO * 3600.0)
    D = F / V

    k_d = arrhenius(A_D, E_D, T)
    k_p = arrhenius(A_P, E_P, T)
    k_t = arrhenius(A_T, E_T, T)

    r_d = k_d * I / MW_I
    r_p = k_p * R_rad * A / MW_A
    r_t = k_t * R_rad**2

    dA = (W_A / 3600.0) / V - D * A - r_p * MW_A
    dI = (f_I / 3600.0) / V - D * I - r_d * MW_I
    dR = 2.0 * r_d - 2.0 * r_t
    dP = r_p * MW_A - D * P

    Q_rxn = DELTA_H_RXN * r_p * V
    Q_cool = UA * (T - T_c)
    Q_flow = RHO * CP * F * (T_FEED - T)
    dT = (Q_rxn - Q_cool + Q_flow) / (RHO * CP * V)

    return np.array(
        [dA, dI, dR, dP, dT],
        dtype=np.float64,
    )


# ===========================================================================
# 5. PHYSICAL PI CONTROLLER
# ===========================================================================
#
# These gains are the exact algebraic physical-units equivalent of the old
# scaled controller, purely so changing coordinates does NOT also change the
# baseline PID behaviour.
#
# Old controller:
#   eP_scaled = eP_phys / 100
#   eT_scaled = eT_phys / 350
#   u_I  = Kp_P*eP_scaled + Ki_P*int(eP_scaled)
#   f_I  = 2.5*u_I
#   u_Tc = 0.535 + Kp_T*eT_scaled + Ki_T*int(eT_scaled)
#   T_c  = 280 + 120*u_Tc
#
# After algebraic conversion, the controller operates directly on physical
# errors and returns physical actions.

BASE_GAINS_PHYSICAL = {
    "Kp_P": 2.5 * (2.6 * 0.8) / 100.0,
    "Ki_P": 2.5 * (0.0002 * 0.8) / 100.0,
    "Kp_T": 120.0 * (0.03 * 0.8) / 350.0,
    "Ki_T": 120.0 * (0.000095 * 0.8) / 350.0,
}

DEFAULT_TC_BIAS = 280.0 + 120.0 * 0.535  # 344.2 K


def physical_pi_controller(
    *,
    P: float,
    T: float,
    int_P: float,
    int_T: float,
    setpoints: dict[str, float],
    dt: float,
    Kp_P: float,
    Ki_P: float,
    Kp_T: float,
    Ki_T: float,
    T_c_bias: float = DEFAULT_TC_BIAS,
) -> tuple[float, float, float, float]:
    """Two-loop PI controller with conditional-integration anti-windup."""

    err_P = float(setpoints["P"] - P)
    err_T = float(setpoints["T"] - T)

    # Polymer loop -> initiator feed.
    f_I_raw = Kp_P * err_P + Ki_P * int_P
    f_I = float(np.clip(f_I_raw, F_I_MIN, F_I_MAX))

    if F_I_MIN <= f_I_raw <= F_I_MAX:
        int_P_next = int_P + err_P * dt
    else:
        int_P_next = int_P

    # Temperature loop -> coolant temperature.
    T_c_raw = T_c_bias + Kp_T * err_T + Ki_T * int_T
    T_c = float(np.clip(T_c_raw, TC_MIN, TC_MAX))

    if TC_MIN <= T_c_raw <= TC_MAX:
        int_T_next = int_T + err_T * dt
    else:
        int_T_next = int_T

    return (
        f_I,
        T_c,
        float(int_P_next),
        float(int_T_next),
    )


# ===========================================================================
# 6. DATASET GENERATION
# ===========================================================================

TRAJECTORY_COLUMNS = [
    "run_id",
    "step",
    "time_min",
    # measurement seen by PID
    "A",
    "I",
    "P",
    "R",
    "T",
    # controller tracking errors and memory
    "err_P",
    "err_T",
    "int_err_P",
    "int_err_T",
    # physical action chosen at this state
    "f_I",
    "T_c",
]


def _measure(
    y_true: np.ndarray,
    rng: np.random.Generator,
    noise_std: np.ndarray,
) -> np.ndarray:
    return y_true + rng.normal(
        loc=0.0,
        scale=noise_std,
        size=5,
    )


def simulate_pid_runs(
    *,
    n_runs: int = 100,
    t_total: float = 100.0 * 3600.0,
    dt: float = 30.0 * 60.0,
    setpoints: dict[str, float] | None = None,
    seed: int | None = 123,
    noise_std: np.ndarray | None = None,
    y0: np.ndarray | None = None,
    randomize_gains: bool = True,
    gain_low: float = 0.3,
    gain_high: float = 1.0,
) -> pd.DataFrame:
    """
    Generate a fixed collection of physical PID trajectories.

    Alignment convention
    --------------------
    Each row contains state s_t, controller memory at t, and action a_t.
    Only AFTER the row is defined is a_t applied to integrate the plant to
    s_(t+1). This prevents the one-step action/state ambiguity present in the
    previous implementation.
    """

    if n_runs < 1:
        raise ValueError("n_runs must be >= 1.")
    if dt <= 0.0 or t_total <= 0.0:
        raise ValueError("dt and t_total must be positive.")
    if not 0.0 <= gain_low <= gain_high:
        raise ValueError("Require 0 <= gain_low <= gain_high.")

    n_steps = int(t_total // dt)
    if n_steps < 1:
        raise ValueError("t_total must contain at least one full time step.")

    setpoints = (
        DEFAULT_SETPOINTS.copy()
        if setpoints is None
        else {"P": float(setpoints["P"]), "T": float(setpoints["T"])}
    )

    y0 = np.asarray(
        DEFAULT_Y0 if y0 is None else y0,
        dtype=np.float64,
    ).reshape(5)

    noise_std = np.asarray(
        DEFAULT_NOISE_STD if noise_std is None else noise_std,
        dtype=np.float64,
    ).reshape(5)
    if np.any(noise_std < 0.0):
        raise ValueError("noise_std values must be non-negative.")

    master_rng = np.random.default_rng(seed)

    base_gains = np.array(
        [
            BASE_GAINS_PHYSICAL["Kp_P"],
            BASE_GAINS_PHYSICAL["Ki_P"],
            BASE_GAINS_PHYSICAL["Kp_T"],
            BASE_GAINS_PHYSICAL["Ki_T"],
        ],
        dtype=np.float64,
    )

    all_trajectories: list[pd.DataFrame] = []

    for run_id in range(n_runs):
        print(f"run {run_id + 1}/{n_runs}")

        multipliers = (
            master_rng.uniform(gain_low, gain_high, size=4)
            if randomize_gains
            else np.ones(4, dtype=np.float64)
        )
        Kp_P, Ki_P, Kp_T, Ki_T = map(
            float,
            base_gains * multipliers,
        )

        env_seed = int(
            master_rng.integers(
                0,
                np.iinfo(np.uint32).max,
                dtype=np.uint32,
            )
        )
        run_rng = np.random.default_rng(env_seed)

        # True physical state order: [A, I, R, P, T].
        y_true = y0.copy()
        y_meas = _measure(y_true, run_rng, noise_std)

        int_P = 0.0
        int_T = 0.0
        rows: list[dict] = []

        for step in range(n_steps):
            A_m, I_m, R_m, P_m, T_m = map(float, y_meas)

            err_P = float(setpoints["P"] - P_m)
            err_T = float(setpoints["T"] - T_m)

            # Integral memory used to construct a_t.
            int_P_current = float(int_P)
            int_T_current = float(int_T)

            f_I, T_c, int_P_next, int_T_next = physical_pi_controller(
                P=P_m,
                T=T_m,
                int_P=int_P_current,
                int_T=int_T_current,
                setpoints=setpoints,
                dt=dt,
                Kp_P=Kp_P,
                Ki_P=Ki_P,
                Kp_T=Kp_T,
                Ki_T=Ki_T,
            )

            # Store (s_t, controller memory_t, a_t).
            rows.append(
                {
                    "run_id": run_id,
                    "step": step,
                    "time_min": step * dt / 60.0,
                    "A": A_m,
                    "I": I_m,
                    "P": P_m,
                    "R": R_m,
                    "T": T_m,
                    "err_P": err_P,
                    "err_T": err_T,
                    "int_err_P": int_P_current,
                    "int_err_T": int_T_current,
                    "f_I": f_I,
                    "T_c": T_c,
                }
            )

            # Apply a_t only after the current row is defined.
            sol = solve_ivp(
                lambda t, y: reactor_odes(t, y, f_I, T_c),
                (0.0, dt),
                y_true,
                method="BDF",
                atol=1e-8,
                rtol=1e-6,
            )
            if not sol.success:
                raise RuntimeError(
                    f"Integration failed in run {run_id}, step {step}: "
                    f"{sol.message}"
                )

            y_true = sol.y[:, -1].astype(np.float64)
            y_meas = _measure(y_true, run_rng, noise_std)
            int_P = int_P_next
            int_T = int_T_next

        all_trajectories.append(
            pd.DataFrame(rows, columns=TRAJECTORY_COLUMNS)
        )


    trajectories = pd.concat(all_trajectories, ignore_index=True)
    return trajectories


def generate_pid_dataset(
    *,
    n_runs: int = 100,
    t_total: float = 100.0 * 3600.0,
    dt: float = 30.0 * 60.0,
    seed: int | None = 123,
    trajectory_csv: str | Path = "physical_pid_trajectories.csv",
    setpoints: dict[str, float] | None = None,
    noise_std: np.ndarray | None = None,
    y0: np.ndarray | None = None,
    randomize_gains: bool = True,
    gain_low: float = 0.3,
    gain_high: float = 1.0,
) -> pd.DataFrame:
    """Generate the dataset, save the trajectory CSV, and return it as a DataFrame."""

    trajectories = simulate_pid_runs(
        n_runs=n_runs,
        t_total=t_total,
        dt=dt,
        setpoints=setpoints,
        seed=seed,
        noise_std=noise_std,
        y0=y0,
        randomize_gains=randomize_gains,
        gain_low=gain_low,
        gain_high=gain_high,
    )

    trajectory_csv = Path(trajectory_csv)
    trajectory_csv.parent.mkdir(parents=True, exist_ok=True)
    trajectories.to_csv(trajectory_csv, index=False)

    print(f"Saved trajectories to: {trajectory_csv.resolve()}")
    return trajectories


if __name__ == "__main__":
    generate_pid_dataset(
        n_runs=100,
        t_total=100.0 * 3600.0,
        dt=30.0 * 60.0,
        seed=123,
    )
