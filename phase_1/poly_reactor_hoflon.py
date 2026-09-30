"""
poly_reactor_hoflon.py
======================

OG HOFLON port for the settled PolymerisationEnv / Idea-3 representation.

Purpose
-------
Preserve the original HOFLON method:
    1) Adversarial Autoencoder (AAE) in-distribution penalty
    2) XGBoost discounted-return Q model
    3) Online RTO objective:
           J = f1 * AAE_reconstruction_MSE
             - f2 * Q(s, a)
             + f3 * ||a - a_prev||^2
    4) Original three-optimiser search:
           Nelder-Mead + Powell + DIRECT
       and select the candidate with the smallest objective.

What changed
------------
Only the data/interface representation is aligned to the settled environment.

Environment observation order:
    [A, I, P, R, T, err_P, err_T, int_err_P, int_err_T]

Action:
    [u_I, u_Tc] in [0, 1]^2

The environment owns:
    - reactor dynamics
    - physical/scaled conversion
    - measurement noise
    - integral-error evolution
    - reward
    - transition timing

HOFLON receives an already-scaled observation and returns an already-scaled action.

The OG feature choices are retained:
    AAE:
        [A, I, P, R, T, err_P, err_T, u_I, u_Tc]  (9D)
        i.e. integral errors are omitted, as in the original AAE.

    Q:
        [A, I, P, R, T, err_P, err_T,
         int_err_P, int_err_T, u_I, u_Tc]          (11D)

Training defaults are retained from the supplied OG scripts:
    AAE:
        latent_dim=4
        batch_size=256
        epochs=500
        lr=1e-5
        MinMaxScaler(feature_range=(-1, 1))
        tanh encoder/decoder
        adversarial N(0,I) latent prior

    XGBoost:
        gamma=0.90
        n_estimators=1600
        learning_rate=0.03
        max_depth=8
        subsample=0.9
        colsample_bytree=0.9
        reg_lambda=1.0
        tree_method="hist"
        random_state=42

    RTO:
        f1=0.5, f2=6.0, f3=0.025
        Nelder-Mead maxiter=250
        Powell maxiter=250
        DIRECT maxiter=40, eps=1e-1

Dataset requirement
-------------------
The CSV supplied to train_hoflon must already use the settled Idea-3
representation and contain at least:

    run_id
    A, I, P, R, T
    err_P, err_T
    int_err_P, int_err_T
    u_I, u_Tc
    reward

A time column is optional. If present, "step" is preferred, otherwise
"time_min" is used to order each run before discounted returns are calculated.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from xgboost import XGBRegressor

try:
    from scipy.optimize import direct
    HAVE_DIRECT = True
except ImportError:
    direct = None
    HAVE_DIRECT = False


# =============================================================================
# CANONICAL SETTLED ENVIRONMENT REPRESENTATION
# =============================================================================

OBS_COLS = [
    "A",
    "I",
    "P",
    "R",
    "T",
    "err_P",
    "err_T",
    "int_err_P",
    "int_err_T",
]

ACTION_COLS = [
    "u_I",
    "u_Tc",
]

# Preserve the OG AAE feature choice: no integral-error inputs.
AAE_STATE_COLS = [
    "A",
    "I",
    "P",
    "R",
    "T",
    "err_P",
    "err_T",
]

AAE_FEATURE_COLS = AAE_STATE_COLS + ACTION_COLS
Q_FEATURE_COLS = OBS_COLS + ACTION_COLS

REQUIRED_TRAINING_COLS = (
    ["run_id", "reward"]
    + OBS_COLS
    + ACTION_COLS
)


# =============================================================================
# OG DEFAULT PARAMETERS
# =============================================================================

@dataclass(frozen=True)
class AAETrainingConfig:
    latent_dim: int = 4
    batch_size: int = 256
    epochs: int = 500
    lr: float = 1e-5


@dataclass(frozen=True)
class QTrainingConfig:
    gamma: float = 0.90
    test_size: float = 0.20
    random_state: int = 42

    n_estimators: int = 1600
    learning_rate: float = 0.03
    max_depth: int = 8
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    reg_lambda: float = 1.0
    tree_method: str = "hist"


@dataclass(frozen=True)
class RTOConfig:
    # OG objective weights.
    f1: float = 0.5
    f2: float = 6.0
    f3: float = 0.025

    nelder_mead_maxiter: int = 250
    powell_maxiter: int = 250

    direct_maxiter: int = 40
    direct_eps: float = 1e-1

    # OG initial guess when there is no previous action.
    initial_action_u_I: float = 0.7
    initial_action_u_Tc: float = 0.5


# =============================================================================
# AAE NETWORKS -- SAME OG ARCHITECTURE
# =============================================================================

class Encoder(nn.Module):
    def __init__(self, input_dim: int = 9, latent_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 16),
            nn.Tanh(),
            nn.Linear(16, latent_dim),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, output_dim: int = 9, latent_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.Tanh(),
            nn.Linear(16, 64),
            nn.Tanh(),
            nn.Linear(64, output_dim),
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, latent_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)


class _TensorDataset(Dataset):
    def __init__(self, array: np.ndarray):
        self.x = torch.from_numpy(array.astype(np.float32))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index]


# =============================================================================
# DATASET VALIDATION
# =============================================================================

def _load_training_dataframe(dataset_path: str | Path) -> pd.DataFrame:
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)

    missing = [
        col
        for col in REQUIRED_TRAINING_COLS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Dataset is missing required settled Idea-3 columns: "
            + ", ".join(missing)
        )

    # Explicitly select/cast rather than relying on accidental dataframe order.
    numeric_cols = list(dict.fromkeys(
        OBS_COLS + ACTION_COLS + ["reward"]
    ))

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="raise")

    return df


def _sort_trajectory_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if "step" in df.columns:
        sort_cols = ["run_id", "step"]
    elif "time_min" in df.columns:
        sort_cols = ["run_id", "time_min"]
    else:
        # Stable sort by run_id keeps original row order within each trajectory.
        sort_cols = ["run_id"]

    return (
        df.sort_values(sort_cols, kind="stable")
        .reset_index(drop=True)
    )


# =============================================================================
# AAE TRAINING
# =============================================================================

def train_aae(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    config: AAETrainingConfig | None = None,
    device: str | torch.device | None = None,
    verbose: bool = True,
) -> dict:
    """
    Train the OG AAE on the settled Idea-3 dataset representation.

    The AAE still uses the original feature concept:
        7 state/error features + 2 actions = 9 inputs.

    It deliberately omits int_err_P and int_err_T to preserve OG HOFLON.
    """
    config = config or AAETrainingConfig()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = _load_training_dataframe(dataset_path)

    X = df[AAE_FEATURE_COLS].to_numpy(dtype=np.float32)
    run_id = df["run_id"].to_numpy()

    # Preserve the original split convention where possible:
    # runs < 95 are training and runs >= 95 are test.
    train_mask = run_id < 95
    test_mask = run_id >= 95

    # If a smaller/new dataset does not contain both sides of that boundary,
    # fail clearly rather than silently changing the baseline experiment.
    if not np.any(train_mask):
        raise ValueError(
            "AAE OG split requires at least one row with run_id < 95."
        )

    if not np.any(test_mask):
        raise ValueError(
            "AAE OG split requires at least one row with run_id >= 95."
        )

    X_train = X[train_mask]
    X_test = X[test_mask]

    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    loader = DataLoader(
        _TensorDataset(X_train_scaled),
        batch_size=config.batch_size,
        shuffle=True,
    )

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(device)

    enc = Encoder(
        input_dim=len(AAE_FEATURE_COLS),
        latent_dim=config.latent_dim,
    ).to(device)

    dec = Decoder(
        output_dim=len(AAE_FEATURE_COLS),
        latent_dim=config.latent_dim,
    ).to(device)

    disc = Discriminator(
        latent_dim=config.latent_dim,
    ).to(device)

    # Same OG Xavier initialization.
    for module in (
        list(enc.modules())
        + list(dec.modules())
        + list(disc.modules())
    ):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    recon_loss = nn.MSELoss()
    bce = nn.BCELoss()

    opt_ae = torch.optim.Adam(
        list(enc.parameters()) + list(dec.parameters()),
        lr=config.lr,
    )

    opt_disc = torch.optim.Adam(
        disc.parameters(),
        lr=config.lr,
    )

    history = []

    for epoch in range(1, config.epochs + 1):
        enc.train()
        dec.train()
        disc.train()

        rec_sum = 0.0
        adv_sum = 0.0

        for x in loader:
            x = x.to(device)

            # -------------------------------------------------------------
            # Phase 1: reconstruction
            # -------------------------------------------------------------
            z = enc(x)
            x_hat = dec(z)
            loss_r = recon_loss(x_hat, x)

            opt_ae.zero_grad()
            loss_r.backward()
            opt_ae.step()

            # -------------------------------------------------------------
            # Phase 2: discriminator
            # -------------------------------------------------------------
            z_real = torch.randn_like(z)

            d_real = disc(z_real.detach())
            d_fake = disc(z.detach())

            lbl_real = torch.ones_like(d_real)
            lbl_fake = torch.zeros_like(d_fake)

            loss_d = (
                bce(d_real, lbl_real)
                + bce(d_fake, lbl_fake)
            )

            opt_disc.zero_grad()
            loss_d.backward()
            opt_disc.step()

            # -------------------------------------------------------------
            # Phase 3: encoder fools discriminator
            # -------------------------------------------------------------
            d_fake = disc(enc(x))
            loss_g = bce(d_fake, lbl_real)

            opt_ae.zero_grad()
            loss_g.backward()
            opt_ae.step()

            rec_sum += loss_r.item() * x.size(0)
            adv_sum += loss_g.item() * x.size(0)

        n_train = len(X_train_scaled)

        epoch_record = {
            "epoch": epoch,
            "reconstruction_mse": rec_sum / n_train,
            "adversarial_loss": adv_sum / n_train,
        }
        history.append(epoch_record)

        if verbose:
            print(
                f"AAE epoch {epoch:>3}/{config.epochs} | "
                f"recon MSE={epoch_record['reconstruction_mse']:.6f} | "
                f"adv={epoch_record['adversarial_loss']:.4f}"
            )

    weights_path = output_dir / "aae_weights.pth"
    scaler_path = output_dir / "ae_scaler.pkl"
    history_path = output_dir / "aae_training_history.csv"

    torch.save(
        {
            "encoder": enc.state_dict(),
            "decoder": dec.state_dict(),
            "latent_dim": config.latent_dim,
            "feature_cols": AAE_FEATURE_COLS,
        },
        weights_path,
    )

    joblib.dump(scaler, scaler_path)
    pd.DataFrame(history).to_csv(history_path, index=False)

    # OG-style reconstruction diagnostic, returned rather than plotted.
    enc.eval()
    dec.eval()

    with torch.no_grad():
        X_test_tensor = torch.from_numpy(
            X_test_scaled
        ).to(device)

        X_recon_scaled = (
            dec(enc(X_test_tensor))
            .cpu()
            .numpy()
        )

    X_recon = scaler.inverse_transform(X_recon_scaled)
    test_mse = float(np.mean((X_test - X_recon) ** 2))

    return {
        "weights_path": weights_path,
        "scaler_path": scaler_path,
        "history_path": history_path,
        "test_reconstruction_mse": test_mse,
        "feature_cols": list(AAE_FEATURE_COLS),
    }


# =============================================================================
# XGBOOST Q TRAINING
# =============================================================================

def _discounted_returns(
    df: pd.DataFrame,
    gamma: float,
) -> np.ndarray:
    df = _sort_trajectory_dataframe(df)

    discounted = np.zeros(len(df), dtype=np.float64)

    for _, group in df.groupby("run_id", sort=False):
        g = 0.0

        for idx in reversed(group.index):
            g = float(df.at[idx, "reward"]) + gamma * g
            discounted[idx] = g

    return discounted


def train_q(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    config: QTrainingConfig | None = None,
    verbose: bool = True,
) -> dict:
    """
    Train the OG XGBoost discounted-return model on the settled Idea-3 data.

    Q input:
        [9D environment observation, 2D action]
    """
    config = config or QTrainingConfig()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = _load_training_dataframe(dataset_path)
    df = _sort_trajectory_dataframe(df)

    df["disc_return"] = _discounted_returns(
        df,
        gamma=config.gamma,
    )

    X = df[Q_FEATURE_COLS].to_numpy(dtype=np.float32)
    y = df["disc_return"].to_numpy(dtype=np.float32)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
    )

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        max_depth=config.max_depth,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        reg_lambda=config.reg_lambda,
        tree_method=config.tree_method,
        random_state=config.random_state,
    )

    model.fit(X_train, y_train)

    model_path = output_dir / "q_discounted.joblib"
    joblib.dump(model, model_path)

    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)

    diagnostics = {
        "train_r2": float(r2_score(y_train, y_pred_train)),
        "val_r2": float(r2_score(y_val, y_pred_val)),
        # Preserve the OG supplied script's actual metric call:
        # mean_squared_error was printed under the label "RMSE".
        "val_mse": float(mean_squared_error(y_val, y_pred_val)),
        "gamma": config.gamma,
        "feature_cols": list(Q_FEATURE_COLS),
        "model_path": model_path,
    }

    if verbose:
        print(f"Q train R² : {diagnostics['train_r2']}")
        print(f"Q val   R² : {diagnostics['val_r2']}")
        print(f"Q val   MSE: {diagnostics['val_mse']}")
        print(f"Saved Q model -> {model_path}")

    return diagnostics


# =============================================================================
# COMBINED TRAINING ENTRY POINT
# =============================================================================

def train_hoflon(
    dataset_path: str | Path,
    output_dir: str | Path = "hoflon_models",
    *,
    aae_config: AAETrainingConfig | None = None,
    q_config: QTrainingConfig | None = None,
    device: str | torch.device | None = None,
    verbose: bool = True,
) -> dict:
    """
    Train both OG-HOFLON learned components using the same dataset.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    aae_result = train_aae(
        dataset_path=dataset_path,
        output_dir=output_dir,
        config=aae_config,
        device=device,
        verbose=verbose,
    )

    q_result = train_q(
        dataset_path=dataset_path,
        output_dir=output_dir,
        config=q_config,
        verbose=verbose,
    )

    return {
        "output_dir": output_dir,
        "aae": aae_result,
        "q": q_result,
    }


# =============================================================================
# ONLINE OG-HOFLON CONTROLLER
# =============================================================================

class HOFLONController:
    """
    Stateful OG-HOFLON controller compatible with:

        action = controller(obs)

    and therefore directly compatible with run_experiment(control_fn=...).

    Important
    ---------
    obs must already be the 9D settled Idea-3 observation returned by
    PolymerisationEnv. No reactor/environment scaling occurs here.
    """

    def __init__(
        self,
        model_dir: str | Path = "hoflon_models",
        *,
        rto_config: RTOConfig | None = None,
        device: str | torch.device | None = None,
    ):
        self.model_dir = Path(model_dir)
        self.rto_config = rto_config or RTOConfig()

        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        weights_path = self.model_dir / "aae_weights.pth"
        scaler_path = self.model_dir / "ae_scaler.pkl"
        q_path = self.model_dir / "q_discounted.joblib"

        for path in (weights_path, scaler_path, q_path):
            if not path.exists():
                raise FileNotFoundError(
                    f"Required HOFLON model file not found: {path}"
                )

        checkpoint = torch.load(
            weights_path,
            map_location=self.device,
        )

        latent_dim = int(
            checkpoint.get("latent_dim", 4)
        )

        self.encoder = Encoder(
            input_dim=len(AAE_FEATURE_COLS),
            latent_dim=latent_dim,
        ).to(self.device)

        self.decoder = Decoder(
            output_dim=len(AAE_FEATURE_COLS),
            latent_dim=latent_dim,
        ).to(self.device)

        self.encoder.load_state_dict(
            checkpoint["encoder"]
        )

        self.decoder.load_state_dict(
            checkpoint["decoder"]
        )

        self.encoder.eval()
        self.decoder.eval()

        self.aae_scaler = joblib.load(
            scaler_path
        )

        self.q_model = joblib.load(
            q_path
        )

        self.reset()

    def reset(self) -> None:
        """
        Reset HOFLON's controller-only memory.

        The environment itself is reset separately by run_experiment.
        """
        self.u_prev = np.zeros(2, dtype=np.float64)

    @staticmethod
    def _validate_obs(obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(
            obs,
            dtype=np.float64,
        ).reshape(-1)

        if obs.size != len(OBS_COLS):
            raise ValueError(
                "HOFLON expects the 9D settled environment observation "
                f"{OBS_COLS}; received shape {obs.shape}."
            )

        if not np.all(np.isfinite(obs)):
            raise ValueError(
                f"HOFLON received non-finite observation: {obs}"
            )

        return obs

    @staticmethod
    def _validate_action(action: np.ndarray) -> np.ndarray:
        action = np.asarray(
            action,
            dtype=np.float64,
        ).reshape(-1)

        if action.size != 2:
            raise ValueError(
                "HOFLON action must contain exactly [u_I, u_Tc]."
            )

        return np.clip(
            action,
            0.0,
            1.0,
        )

    @staticmethod
    def _aae_input(
        obs: np.ndarray,
        action: np.ndarray,
    ) -> np.ndarray:
        """
        OG AAE features using the new canonical observation ordering.

        obs:
            [A,I,P,R,T,err_P,err_T,int_P,int_T]

        AAE:
            [A,I,P,R,T,err_P,err_T,u_I,u_Tc]
        """
        return np.array(
            [
                obs[0],  # A
                obs[1],  # I
                obs[2],  # P
                obs[3],  # R
                obs[4],  # T
                obs[5],  # err_P
                obs[6],  # err_T
                action[0],
                action[1],
            ],
            dtype=np.float64,
        ).reshape(1, -1)

    @staticmethod
    def _q_input(
        obs: np.ndarray,
        action: np.ndarray,
    ) -> np.ndarray:
        """
        Q features:
            complete settled 9D observation + 2D action.
        """
        return np.concatenate(
            [
                obs,
                action,
            ]
        ).astype(
            np.float32
        ).reshape(1, -1)

    def _aae_reconstruction_mse(
        self,
        obs: np.ndarray,
        action: np.ndarray,
    ) -> float:
        x = self._aae_input(
            obs,
            action,
        )

        x_scaled = (
            self.aae_scaler
            .transform(x)
            .astype(np.float32)
        )

        x_tensor = torch.from_numpy(
            x_scaled
        ).to(self.device)

        with torch.no_grad():
            reconstruction_scaled = self.decoder(
                self.encoder(
                    x_tensor
                )
            ).cpu().numpy()

        reconstruction = (
            self.aae_scaler
            .inverse_transform(
                reconstruction_scaled
            )
        )

        # Preserve OG objective: MSE in the original model-input coordinates
        # after inverse-transforming the AAE reconstruction.
        return float(
            np.mean(
                (x - reconstruction) ** 2
            )
        )

    def _q_value(
        self,
        obs: np.ndarray,
        action: np.ndarray,
    ) -> float:
        x = self._q_input(
            obs,
            action,
        )

        return float(
            self.q_model.predict(x)[0]
        )

    def objective(
        self,
        action: np.ndarray,
        obs: np.ndarray,
    ) -> float:
        """
        Exact OG objective form:

            f1 * recon_mse
          + f2 * (-Q)
          + f3 * delta_u_squared
        """
        action = self._validate_action(action)

        cfg = self.rto_config

        reconstruction_mse = (
            self._aae_reconstruction_mse(
                obs,
                action,
            )
        )

        q_value = self._q_value(
            obs,
            action,
        )

        delta_u = (
            action
            - self.u_prev
        )

        action_change = float(
            np.sum(delta_u**2)
        )

        return float(
            cfg.f1 * reconstruction_mse
            + cfg.f2 * (-q_value)
            + cfg.f3 * action_change
        )

    def optimise_action(
        self,
        obs: np.ndarray,
    ) -> np.ndarray:
        """
        OG three-solver RTO:
            Nelder-Mead
            Powell
            DIRECT

        Candidate actions are clipped to [0,1]^2 and the candidate with the
        smallest evaluated HOFLON objective is returned.
        """
        obs = self._validate_obs(obs)
        cfg = self.rto_config

        bounds = [
            (0.0, 1.0),
            (0.0, 1.0),
        ]

        if np.any(self.u_prev):
            x0 = self.u_prev.copy()
        else:
            x0 = np.array(
                [
                    cfg.initial_action_u_I,
                    cfg.initial_action_u_Tc,
                ],
                dtype=np.float64,
            )

        def obj(action):
            return self.objective(
                action=action,
                obs=obs,
            )

        # -------------------------------------------------------------
        # 1) Nelder-Mead -- preserve OG unbounded solver then clip
        # -------------------------------------------------------------
        result_nm = minimize(
            obj,
            x0,
            method="Nelder-Mead",
            options={
                "maxiter": cfg.nelder_mead_maxiter,
                "disp": False,
            },
        )

        action_nm = np.clip(
            result_nm.x,
            0.0,
            1.0,
        )

        objective_nm = obj(
            action_nm
        )

        # -------------------------------------------------------------
        # 2) Powell -- bounded
        # -------------------------------------------------------------
        result_pw = minimize(
            obj,
            x0,
            method="Powell",
            bounds=bounds,
            options={
                "maxiter": cfg.powell_maxiter,
                "disp": False,
            },
        )

        action_pw = np.clip(
            result_pw.x,
            0.0,
            1.0,
        )

        objective_pw = obj(
            action_pw
        )

        # -------------------------------------------------------------
        # 3) DIRECT
        # -------------------------------------------------------------
        if HAVE_DIRECT:
            result_direct = direct(
                obj,
                bounds,
                maxiter=cfg.direct_maxiter,
                eps=cfg.direct_eps,
            )

            action_direct = np.clip(
                result_direct.x,
                0.0,
                1.0,
            )

            objective_direct = obj(
                action_direct
            )
        else:
            action_direct = None
            objective_direct = np.inf

        candidate_actions = [
            action_nm,
            action_pw,
            action_direct,
        ]

        candidate_objectives = np.array(
            [
                objective_nm,
                objective_pw,
                objective_direct,
            ],
            dtype=np.float64,
        )

        best_index = int(
            np.argmin(
                candidate_objectives
            )
        )

        best_action = np.asarray(
            candidate_actions[best_index],
            dtype=np.float64,
        )

        return np.clip(
            best_action,
            0.0,
            1.0,
        ).astype(np.float32)

    def __call__(
        self,
        obs: np.ndarray,
    ) -> np.ndarray:
        """
        run_experiment-compatible controller interface:

            action = controller(obs)
        """
        action = self.optimise_action(
            obs
        )

        self.u_prev = (
            np.asarray(
                action,
                dtype=np.float64,
            )
            .copy()
        )

        return action


# =============================================================================
# CONVENIENCE LOADER
# =============================================================================

def load_hoflon(
    model_dir: str | Path = "hoflon_models",
    *,
    rto_config: RTOConfig | None = None,
    device: str | torch.device | None = None,
) -> HOFLONController:
    """
    Load a trained HOFLON controller ready for run_experiment.
    """
    return HOFLONController(
        model_dir=model_dir,
        rto_config=rto_config,
        device=device,
    )
