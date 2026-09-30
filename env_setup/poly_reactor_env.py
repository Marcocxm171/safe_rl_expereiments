"""
poly_reactor_idea2_env.py
=========================

Gym/Gymnasium environment for the polymerisation reactor using the fixed
"Idea 2" scaling.

DESIGN
------
Actor/environment interface:
    action in  : [u_I, u_Tc] in [0, 1]^2
    obs out    : 9-dimensional scaled observation

Internal plant:
    entirely in PHYSICAL units [A, I, R, P, T]
    with the same mechanistic ODE used by physical_pid_data_generation.py.

Observation:
    [A_s, I_s, P_s, R_s, T_s,
     err_P_s, err_T_s,
     int_err_P_s, int_err_T_s]

Scaling:
    A_s = A / 100
    I_s = I / 100
    P_s = P / 100
    R_s = R / 100

    T_s = (T - T_min) / (T_max - T_min)

    err_P_s = 1 - P_s
    err_T_s = T_sp_s - T_s

    int_err_P_s = int_err_P_phys / 100 / 1000

    int_err_T_scaled = int_err_T_phys / (T_max - T_min)
    int_err_T_s = int_err_T_scaled / max_abs_int_T_scaled

where T_min, T_max, and max_abs_int_T_scaled are fitted ONCE from the raw
unscaled PID training dataset and then frozen for every rollout.

Action conversion:
    f_I = 2.5 * u_I
    T_c = 280 + 120 * u_Tc

Integral-memory update:
    To match the PID dataset convention, each integral accumulates the
    measured physical tracking error only when the corresponding chosen action
    is not saturated at 0 or 1. At saturation, integration freezes.

Reward:
    This environment deliberately does NOT embed a candidate RL reward.
    step() returns reward = 0.0. Reward ideas remain separate from the plant.

The `info` dictionary exposes physical values for evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # fallback for older installations
    import gym
    from gym import spaces


# ===========================================================================
# 1. PHYSICAL REACTOR PARAMETERS
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
# 2. PHYSICAL LIMITS / SETPOINTS
# ===========================================================================

F_I_MIN = 0.0
F_I_MAX = 2.5

TC_MIN = 280.0
TC_MAX = 400.0

P_SETPOINT = 100.0
T_SETPOINT = 350.0

S_CONC = 100.0

DEFAULT_Y0 = np.array(
    [555.555456, 0.0, 0.0, 0.0, 350.0],
    dtype=np.float64,
)

# Order: [A, I, R, P, T]
DEFAULT_NOISE_STD = np.array(
    [2.0, 0.0, 0.0, 0.5, 0.0005714 * 350.0],
    dtype=np.float64,
)


# ===========================================================================
# 3. IDEA-2 SCALER
# ===========================================================================

@dataclass(frozen=True)
class Idea2Scaler:
    """Frozen scaling constants fitted from the raw PID training dataset."""

    t_min: float
    t_max: float
    t_range: float
    t_setpoint_scaled: float
    max_abs_int_t_scaled: float

    @classmethod
    def fit(
        cls,
        raw_data: str | Path | pd.DataFrame,
    ) -> "Idea2Scaler":
        """
        Fit Idea-2 temperature constants ONCE from the raw unscaled PID data.
        """

        if isinstance(raw_data, pd.DataFrame):
            df = raw_data.copy()
        else:
            path = Path(raw_data)
            if not path.exists():
                raise FileNotFoundError(
                    f"Raw PID dataset not found: {path}"
                )
            df = pd.read_csv(path)

        required = {"T", "int_err_T"}
        missing = required.difference(df.columns)

        if missing:
            raise ValueError(
                f"Raw PID data is missing required columns: "
                f"{sorted(missing)}"
            )

        t_min = float(df["T"].min())
        t_max = float(df["T"].max())
        t_range = t_max - t_min

        if not np.isfinite(t_range) or t_range <= 0.0:
            raise ValueError(
                f"Invalid temperature range: "
                f"T_min={t_min}, T_max={t_max}"
            )

        t_setpoint_scaled = (
            T_SETPOINT - t_min
        ) / t_range

        int_t_scaled = (
            df["int_err_T"].to_numpy(dtype=float)
            / t_range
        )

        max_abs_int_t_scaled = float(
            np.max(np.abs(int_t_scaled))
        )

        if (
            not np.isfinite(max_abs_int_t_scaled)
            or max_abs_int_t_scaled <= 0.0
        ):
            raise ValueError(
                "Maximum absolute scaled temperature integral error "
                "must be positive."
            )

        return cls(
            t_min=t_min,
            t_max=t_max,
            t_range=t_range,
            t_setpoint_scaled=t_setpoint_scaled,
            max_abs_int_t_scaled=max_abs_int_t_scaled,
        )

    def scale_observation(
        self,
        measurement_phys: np.ndarray,
        int_err_p_phys: float,
        int_err_t_phys: float,
    ) -> np.ndarray:
        """
        Convert physical measured state + physical integral memories
        into the fixed 9D Idea-2 observation.
        """

        measurement_phys = np.asarray(
            measurement_phys,
            dtype=np.float64,
        ).reshape(5)

        # Internal physical state order is [A, I, R, P, T].
        A, I, R_rad, P, T = map(
            float,
            measurement_phys,
        )

        A_s = A / S_CONC
        I_s = I / S_CONC
        P_s = P / S_CONC
        R_s = R_rad / S_CONC

        T_s = (
            T - self.t_min
        ) / self.t_range

        p_setpoint_scaled = (
            P_SETPOINT / S_CONC
        )

        err_p_s = (
            p_setpoint_scaled
            - P_s
        )

        err_t_s = (
            self.t_setpoint_scaled
            - T_s
        )

        # Original HOFLON concentration integral convention.
        int_p_s = (
            float(int_err_p_phys)
            / S_CONC
            / 1000.0
        )

        # Idea-2 temperature integral convention.
        int_t_scaled = (
            float(int_err_t_phys)
            / self.t_range
        )

        int_t_s = (
            int_t_scaled
            / self.max_abs_int_t_scaled
        )

        # Observation order matches the processed Idea-3/4 datasets:
        # [A, I, P, R, T, err_P, err_T, int_err_P, int_err_T]
        return np.array(
            [
                A_s,
                I_s,
                P_s,
                R_s,
                T_s,
                err_p_s,
                err_t_s,
                int_p_s,
                int_t_s,
            ],
            dtype=np.float32,
        )


# ===========================================================================
# 4. PHYSICAL REACTOR MODEL -- NO SCALING INSIDE THE ODE
# ===========================================================================

def _arrhenius(
    pre_exponential: float,
    activation_energy: float,
    T: float,
) -> float:
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
    Physical reactor ODE.

    State order:
        [A, I, R, P, T]

    Inputs:
        f_I : kg h^-1
        T_c : K
    """
    del t

    A, I, R_rad, P, T = map(float, y)

    F = (
        f_I + W_A + W_S
    ) / (RHO * 3600.0)

    D = F / V

    k_d = _arrhenius(A_D, E_D, T)
    k_p = _arrhenius(A_P, E_P, T)
    k_t = _arrhenius(A_T, E_T, T)

    r_d = k_d * I / MW_I
    r_p = k_p * R_rad * A / MW_A
    r_t = k_t * R_rad**2

    dA = (
        (W_A / 3600.0) / V
        - D * A
        - r_p * MW_A
    )

    dI = (
        (f_I / 3600.0) / V
        - D * I
        - r_d * MW_I
    )

    dR = (
        2.0 * r_d
        - 2.0 * r_t
    )

    dP = (
        r_p * MW_A
        - D * P
    )

    Q_rxn = (
        DELTA_H_RXN
        * r_p
        * V
    )

    Q_cool = (
        UA
        * (T - T_c)
    )

    Q_flow = (
        RHO
        * CP
        * F
        * (T_FEED - T)
    )

    dT = (
        Q_rxn
        - Q_cool
        + Q_flow
    ) / (
        RHO
        * CP
        * V
    )

    return np.array(
        [
            dA,
            dI,
            dR,
            dP,
            dT,
        ],
        dtype=np.float64,
    )


# ===========================================================================
# 5. GYM ENVIRONMENT
# ===========================================================================

class PolymerisationIdea2Env(gym.Env):
    """
    Polymerisation environment with:

        scaled action in
        physical reactor internally
        scaled Idea-2 observation out

    Reward is deliberately neutral (0.0). Use physical quantities in `info`
    for evaluation or wrap the environment with a separate reward definition.
    """

    metadata = {
        "render_modes": [],
    }

    def __init__(
        self,
        raw_pid_data: str | Path | pd.DataFrame,
        *,
        dt: float = 30.0 * 60.0,
        t_total: float = 100.0 * 3600.0,
        seed: int | None = None,
        y0: np.ndarray | None = None,
        noise_std: np.ndarray | None = None,
        p_setpoint: float = P_SETPOINT,
        t_setpoint: float = T_SETPOINT,
        saturation_tol: float = 1e-8,
    ):
        super().__init__()

        if dt <= 0.0:
            raise ValueError("dt must be positive.")

        if t_total <= 0.0:
            raise ValueError(
                "t_total must be positive."
            )

        self.dt = float(dt)
        self.t_total = float(t_total)
        self.n_steps = int(
            self.t_total // self.dt
        )

        if self.n_steps < 1:
            raise ValueError(
                "t_total must contain at least one full step."
            )

        # Keep setpoints configurable, but defaults match the PID dataset.
        self.p_setpoint = float(p_setpoint)
        self.t_setpoint = float(t_setpoint)

        if (
            self.p_setpoint != P_SETPOINT
            or self.t_setpoint != T_SETPOINT
        ):
            raise ValueError(
                "Idea-2 scaler currently assumes fixed physical "
                "setpoints P=100 and T=350 to match the training dataset."
            )

        self.y0 = np.asarray(
            DEFAULT_Y0 if y0 is None else y0,
            dtype=np.float64,
        ).reshape(5)

        self.noise_std = np.asarray(
            DEFAULT_NOISE_STD
            if noise_std is None
            else noise_std,
            dtype=np.float64,
        ).reshape(5)

        if np.any(
            self.noise_std < 0.0
        ):
            raise ValueError(
                "noise_std values must be non-negative."
            )

        self.saturation_tol = float(
            saturation_tol
        )

        # Fit once, then freeze.
        self.scaler = Idea2Scaler.fit(
            raw_pid_data
        )

        # Actor always works in normalized HOFLON action coordinates.
        self.action_space = spaces.Box(
            low=np.zeros(
                2,
                dtype=np.float32,
            ),
            high=np.ones(
                2,
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        # Several scaled state quantities may legitimately leave their
        # training-data nominal ranges during closed-loop evaluation.
        # Do not clip observations; expose out-of-distribution behaviour.
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(9,),
            dtype=np.float32,
        )

        self._initial_seed = seed

        # Runtime state populated by reset().
        self._rng: np.random.Generator | None = None
        self.y_true = self.y0.copy()
        self.y_meas = self.y0.copy()

        self.int_err_p_phys = 0.0
        self.int_err_t_phys = 0.0

        self.step_count = 0
        self.time_s = 0.0

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _measure(
        self,
        y_true: np.ndarray,
    ) -> np.ndarray:
        return (
            y_true
            + self._rng.normal(
                loc=0.0,
                scale=self.noise_std,
                size=5,
            )
        )

    @staticmethod
    def _scaled_to_physical_action(
        action: np.ndarray,
    ) -> tuple[float, float]:
        u_i = float(action[0])
        u_tc = float(action[1])

        f_i = (
            F_I_MIN
            + u_i
            * (F_I_MAX - F_I_MIN)
        )

        t_c = (
            TC_MIN
            + u_tc
            * (TC_MAX - TC_MIN)
        )

        return (
            float(f_i),
            float(t_c),
        )

    def _current_physical_errors(
        self,
    ) -> tuple[float, float]:
        # y_meas order: [A, I, R, P, T]
        P_meas = float(
            self.y_meas[3]
        )
        T_meas = float(
            self.y_meas[4]
        )

        err_p = (
            self.p_setpoint
            - P_meas
        )

        err_t = (
            self.t_setpoint
            - T_meas
        )

        return (
            float(err_p),
            float(err_t),
        )

    def _observation(
        self,
    ) -> np.ndarray:
        return self.scaler.scale_observation(
            measurement_phys=self.y_meas,
            int_err_p_phys=self.int_err_p_phys,
            int_err_t_phys=self.int_err_t_phys,
        )

    def _info(
        self,
        *,
        action_scaled: np.ndarray | None = None,
        action_physical: tuple[float, float] | None = None,
    ) -> dict:
        err_p, err_t = (
            self._current_physical_errors()
        )

        info = {
            "time_s": float(self.time_s),
            "time_min": float(
                self.time_s / 60.0
            ),

            # Internal noise-free plant state.
            "A_true": float(
                self.y_true[0]
            ),
            "I_true": float(
                self.y_true[1]
            ),
            "R_true": float(
                self.y_true[2]
            ),
            "P_true": float(
                self.y_true[3]
            ),
            "T_true": float(
                self.y_true[4]
            ),

            # Noisy measurement seen by the actor.
            "A": float(
                self.y_meas[0]
            ),
            "I": float(
                self.y_meas[1]
            ),
            "R": float(
                self.y_meas[2]
            ),
            "P": float(
                self.y_meas[3]
            ),
            "T": float(
                self.y_meas[4]
            ),

            "err_P_phys": err_p,
            "err_T_phys": err_t,

            "int_err_P_phys": float(
                self.int_err_p_phys
            ),
            "int_err_T_phys": float(
                self.int_err_t_phys
            ),
        }

        if action_scaled is not None:
            info["u_I"] = float(
                action_scaled[0]
            )
            info["u_Tc"] = float(
                action_scaled[1]
            )

        if action_physical is not None:
            info["f_I"] = float(
                action_physical[0]
            )
            info["T_c"] = float(
                action_physical[1]
            )

        return info

    # -----------------------------------------------------------------------
    # Gym API
    # -----------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ):
        # Let Gym/Gymnasium initialize its own RNG bookkeeping when possible.
        try:
            super().reset(
                seed=seed
            )
        except TypeError:
            # Older Gym versions.
            super().reset()

        del options

        effective_seed = (
            self._initial_seed
            if seed is None
            else seed
        )

        self._rng = np.random.default_rng(
            effective_seed
        )

        self.y_true = (
            self.y0.copy()
        )

        self.int_err_p_phys = 0.0
        self.int_err_t_phys = 0.0

        self.step_count = 0
        self.time_s = 0.0

        # Same convention as PID dataset:
        # actor/controller sees a noisy measurement at the current state.
        self.y_meas = self._measure(
            self.y_true
        )

        obs = self._observation()
        info = self._info()

        return obs, info

    def step(
        self,
        action,
    ):
        action = np.asarray(
            action,
            dtype=np.float64,
        ).reshape(2)

        if not np.all(
            np.isfinite(action)
        ):
            raise ValueError(
                f"Action contains non-finite values: {action}"
            )

        # Safety clip at the environment boundary.
        action = np.clip(
            action,
            0.0,
            1.0,
        )

        # ---------------------------------------------------------------
        # Current row semantics:
        #
        # observation_t -> action_t -> plant -> observation_(t+1)
        # ---------------------------------------------------------------

        err_p_phys, err_t_phys = (
            self._current_physical_errors()
        )

        # Dataset-compatible integral-memory update:
        # integrate only when the corresponding action is not saturated.
        u_i = float(
            action[0]
        )
        u_tc = float(
            action[1]
        )

        if (
            u_i > self.saturation_tol
            and u_i < 1.0 - self.saturation_tol
        ):
            int_p_next = (
                self.int_err_p_phys
                + err_p_phys * self.dt
            )
        else:
            int_p_next = (
                self.int_err_p_phys
            )

        if (
            u_tc > self.saturation_tol
            and u_tc < 1.0 - self.saturation_tol
        ):
            int_t_next = (
                self.int_err_t_phys
                + err_t_phys * self.dt
            )
        else:
            int_t_next = (
                self.int_err_t_phys
            )

        f_i, t_c = (
            self._scaled_to_physical_action(
                action
            )
        )

        # Physical reactor integration.
        sol = solve_ivp(
            lambda t, y: reactor_odes(
                t,
                y,
                f_i,
                t_c,
            ),
            (0.0, self.dt),
            self.y_true,
            method="BDF",
            atol=1e-8,
            rtol=1e-6,
        )

        if not sol.success:
            raise RuntimeError(
                "Reactor integration failed: "
                f"{sol.message}"
            )

        self.y_true = (
            sol.y[:, -1]
            .astype(np.float64)
        )

        self.int_err_p_phys = float(
            int_p_next
        )
        self.int_err_t_phys = float(
            int_t_next
        )

        self.step_count += 1
        self.time_s = (
            self.step_count
            * self.dt
        )

        # New noisy measurement corresponds to genuine next state.
        self.y_meas = self._measure(
            self.y_true
        )

        obs_next = self._observation()

        # Deliberately no embedded candidate reward.
        reward = 0.0

        terminated = False
        truncated = (
            self.step_count
            >= self.n_steps
        )

        info = self._info(
            action_scaled=action,
            action_physical=(
                f_i,
                t_c,
            ),
        )

        return (
            obs_next,
            reward,
            terminated,
            truncated,
            info,
        )


# ===========================================================================
# 6. CONVENIENCE FACTORY
# ===========================================================================

def make_env(
    raw_pid_data: str | Path | pd.DataFrame,
    *,
    dt: float = 30.0 * 60.0,
    t_total: float = 100.0 * 3600.0,
    seed: int | None = 123,
    noise_std: np.ndarray | None = None,
) -> PolymerisationIdea2Env:
    """Convenience constructor for notebook use."""

    return PolymerisationIdea2Env(
        raw_pid_data=raw_pid_data,
        dt=dt,
        t_total=t_total,
        seed=seed,
        noise_std=noise_std,
    )
