
"""
train_weighted_bc.py

Train a weighted behavioural-cloning actor from any processed HOFLON dataset.

Expected default columns
------------------------
Observations:
    A, I, P, R, T, err_P, err_T, int_err_P, int_err_T

Actions:
    u_I, u_Tc

Reward:
    reward

The reward is min-max scaled to [0, 1] and used ONLY as the sample weight:

    w_i = (r_i - r_min) / (r_max - r_min)

Weighted BC loss:

    L = sum_i w_i * ||pi(s_i) - a_i||^2 / sum_i w_i

Network:
    9 -> 32 -> 32 -> 2

The final sigmoid constrains both actions to [0, 1].

Default training:
    100 epochs
"""

from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
import importlib

# PyTorch 2.14 compatibility fix
if not hasattr(torch, "_utils"):
    torch._utils = importlib.import_module("torch._utils")

import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


DEFAULT_OBS_COLS = [
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

DEFAULT_ACTION_COLS = [
    "u_I",
    "u_Tc",
]


class WeightedBCActor(nn.Module):
    """Deterministic 32-32 BC actor with [0,1] action output."""

    def __init__(
        self,
        obs_dim: int = 9,
        action_dim: int = 2,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
            nn.Sigmoid(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _minmax_reward(reward: np.ndarray) -> np.ndarray:
    """
    Convert reward to weights in [0, 1].

    If every reward is identical, every sample receives weight 1.
    """
    reward = np.asarray(reward, dtype=np.float32)

    r_min = float(np.min(reward))
    r_max = float(np.max(reward))

    if np.isclose(r_max, r_min):
        return np.ones_like(reward, dtype=np.float32)

    weights = (reward - r_min) / (r_max - r_min)

    return np.clip(
        weights,
        0.0,
        1.0,
    ).astype(np.float32)


def train_weighted_bc(
    dataset_path: str | Path,
    output_path: str | Path = "weighted_bc_actor.pth",
    *,
    obs_cols: list[str] | None = None,
    action_cols: list[str] | None = None,
    reward_col: str = "reward",
    epochs: int = 100,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    seed: int = 123,
    device: str | None = None,
    verbose: bool = True,
):
    """
    Train one weighted-BC actor and save one .pth checkpoint.

    Parameters
    ----------
    dataset_path:
        Processed CSV containing scaled observations, scaled actions,
        and reward.

    output_path:
        Path for the saved actor checkpoint.

    obs_cols:
        Observation columns. Defaults to the fixed 9D HOFLON observation.

    action_cols:
        Action columns. Defaults to ["u_I", "u_Tc"].

    reward_col:
        Reward column to min-max into BC weights.

    epochs:
        Default = 100.

    Returns
    -------
    model:
        Trained WeightedBCActor.

    history:
        List of mean training losses by epoch.
    """

    _set_seed(seed)

    obs_cols = (
        DEFAULT_OBS_COLS
        if obs_cols is None
        else list(obs_cols)
    )

    action_cols = (
        DEFAULT_ACTION_COLS
        if action_cols is None
        else list(action_cols)
    )

    dataset_path = Path(dataset_path)
    output_path = Path(output_path)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}"
        )

    df = pd.read_csv(dataset_path)

    required_cols = (
        obs_cols
        + action_cols
        + [reward_col]
    )

    missing = [
        col
        for col in required_cols
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Dataset is missing required columns: "
            + ", ".join(missing)
        )

    # Keep only rows usable for BC training.
    train_df = (
        df[required_cols]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .copy()
    )

    if len(train_df) == 0:
        raise ValueError(
            "No valid rows remain after removing NaN/inf values."
        )

    obs_np = train_df[
        obs_cols
    ].to_numpy(
        dtype=np.float32
    )

    actions_np = train_df[
        action_cols
    ].to_numpy(
        dtype=np.float32
    )

    # The actor/environment convention is scaled actions in [0,1].
    if (
        np.min(actions_np) < -1e-6
        or np.max(actions_np) > 1.0 + 1e-6
    ):
        raise ValueError(
            "Action data are not in [0,1]. "
            "Use the processed/scaled u_I and u_Tc columns."
        )

    actions_np = np.clip(
        actions_np,
        0.0,
        1.0,
    )

    reward_np = train_df[
        reward_col
    ].to_numpy(
        dtype=np.float32
    )

    weights_np = _minmax_reward(
        reward_np
    )

    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    device = torch.device(device)

    obs_tensor = torch.from_numpy(
        obs_np
    )

    action_tensor = torch.from_numpy(
        actions_np
    )

    weight_tensor = torch.from_numpy(
        weights_np
    )

    dataset = TensorDataset(
        obs_tensor,
        action_tensor,
        weight_tensor,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        drop_last=False,
    )

    model = WeightedBCActor(
        obs_dim=len(obs_cols),
        action_dim=len(action_cols),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    history: list[float] = []

    for epoch in range(1, epochs + 1):
        model.train()

        total_loss = 0.0
        total_samples = 0

        for obs_batch, action_batch, weight_batch in loader:
            obs_batch = obs_batch.to(device)
            action_batch = action_batch.to(device)
            weight_batch = weight_batch.to(device)

            pred_action = model(obs_batch)

            # Per-row action MSE across the two action dimensions.
            per_sample_loss = (
                (pred_action - action_batch) ** 2
            ).mean(dim=1)

            weight_sum = weight_batch.sum()

            # A batch can theoretically contain only minimum-reward rows,
            # all of which have weight 0 after min-max scaling.
            if weight_sum.item() <= 0.0:
                continue

            loss = (
                weight_batch * per_sample_loss
            ).sum() / weight_sum

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_n = obs_batch.shape[0]

            total_loss += (
                float(loss.item())
                * batch_n
            )
            total_samples += batch_n

        epoch_loss = (
            total_loss / total_samples
            if total_samples > 0
            else float("nan")
        )

        history.append(epoch_loss)

        if verbose and (
            epoch == 1
            or epoch % 10 == 0
            or epoch == epochs
        ):
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"weighted BC loss = {epoch_loss:.8f}"
            )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "obs_cols": obs_cols,
        "action_cols": action_cols,
        "reward_col": reward_col,
        "obs_dim": len(obs_cols),
        "action_dim": len(action_cols),
        "hidden_sizes": [32, 32],
        "activation": "ReLU",
        "output_activation": "Sigmoid",
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "reward_min": float(
            np.min(reward_np)
        ),
        "reward_max": float(
            np.max(reward_np)
        ),
        "weight_min": float(
            np.min(weights_np)
        ),
        "weight_max": float(
            np.max(weights_np)
        ),
    }

    torch.save(
        checkpoint,
        output_path,
    )

    if verbose:
        print()
        print(
            f"Training rows: {len(train_df)}"
        )
        print(
            f"Reward range: "
            f"[{checkpoint['reward_min']:.6g}, "
            f"{checkpoint['reward_max']:.6g}]"
        )
        print(
            f"Weight range: "
            f"[{checkpoint['weight_min']:.6g}, "
            f"{checkpoint['weight_max']:.6g}]"
        )
        print(
            f"Saved actor to: "
            f"{output_path.resolve()}"
        )

    return model, history


def load_weighted_bc(
    checkpoint_path: str | Path,
    *,
    device: str | None = None,
) -> WeightedBCActor:
    """Load a model saved by train_weighted_bc()."""

    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    device = torch.device(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model = WeightedBCActor(
        obs_dim=checkpoint["obs_dim"],
        action_dim=checkpoint["action_dim"],
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    return model


if __name__ == "__main__":
    # Example:
    #
    # python train_weighted_bc.py idea3_mse_processed.csv idea3_bc_actor.pth
    #
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "dataset",
        type=str,
        help="Processed CSV containing observations, actions and reward.",
    )

    parser.add_argument(
        "output",
        type=str,
        nargs="?",
        default="weighted_bc_actor.pth",
        help="Output .pth model path.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=123,
    )

    args = parser.parse_args()

    train_weighted_bc(
        dataset_path=args.dataset,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
    )
