
from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

if not hasattr(torch, "_utils"):
    torch._utils = importlib.import_module("torch._utils")

import d3rlpy


OBS_COLS = [
    "A", "I", "P", "R", "T",
    "err_P", "err_T", "int_err_P", "int_err_T",
]

ACTION_COLS = ["u_I", "u_Tc"]


def build_dataset(
    data_path: str | Path,
    *,
    obs_cols: Sequence[str] = OBS_COLS,
    action_cols: Sequence[str] = ACTION_COLS,
    reward_col: str = "reward",
    run_col: str = "run_id",
    time_col: str = "time_min",
):
    """
    Final HOFLON convention:
        row t = (s_t, a_t, r_t)
        r_t belongs to s_t --a_t--> s_(t+1)
    """

    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path).copy()

    df = df.rename(
        columns={
            "R_rad": "R",
            "intP_feat": "int_err_P",
            "intT_feat": "int_err_T",
        }
    )

    if run_col not in df.columns:
        df[run_col] = 0

    sort_cols = [run_col]
    if time_col in df.columns:
        sort_cols.append(time_col)
    elif "step" in df.columns:
        sort_cols.append("step")

    df = df.sort_values(
        sort_cols,
        kind="stable",
    ).reset_index(drop=True)

    required = (
        list(obs_cols)
        + list(action_cols)
        + [reward_col, run_col]
    )

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Dataset missing required columns: {missing}"
        )

    observations = []
    actions = []
    rewards = []
    terminals = []
    timeouts = []

    for _, run in df.groupby(run_col, sort=False):
        run = run.reset_index(drop=True)

        if len(run) < 2:
            continue

        obs = run[list(obs_cols)].to_numpy(np.float32)
        act = run[list(action_cols)].to_numpy(np.float32)
        rew = run[reward_col].to_numpy(np.float32)

        if not np.all(np.isfinite(obs)):
            raise ValueError("Observation data contain NaN/inf.")

        if not np.all(np.isfinite(act)):
            raise ValueError("Action data contain NaN/inf.")

        if not np.all(np.isfinite(rew)):
            raise ValueError("Reward data contain NaN/inf.")

        if np.min(act) < -1e-6 or np.max(act) > 1.0 + 1e-6:
            raise ValueError(
                "Expected u_I and u_Tc to already be scaled to [0,1]."
            )

        act = np.clip(act, 0.0, 1.0)

        n = len(run)

        terminal = np.zeros(n, dtype=np.float32)
        timeout = np.zeros(n, dtype=np.float32)
        timeout[-1] = 1.0

        observations.append(obs)
        actions.append(act)
        rewards.append(rew)
        terminals.append(terminal)
        timeouts.append(timeout)

    if not observations:
        raise ValueError("No usable trajectories found.")

    observations = np.concatenate(observations)
    actions = np.concatenate(actions)
    rewards = np.concatenate(rewards)
    terminals = np.concatenate(terminals)
    timeouts = np.concatenate(timeouts)

    dataset = d3rlpy.dataset.MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
        timeouts=timeouts,
    )

    print(f"Dataset: {data_path}")
    print(f"Runs: {df[run_col].nunique()}")
    print(f"Stored rows: {len(observations):,}")
    print(f"d3rlpy transitions: {dataset.transition_count:,}")
    print(f"Reward range: [{rewards.min():.6g}, {rewards.max():.6g}]")
    print(
        "Action min/max:",
        actions.min(axis=0),
        actions.max(axis=0),
    )

    return dataset


def train_iql(
    data_path: str | Path,
    *,
    run_name: str = "iql_run",
    output_dir: str | Path = "iql_runs",

    epochs: int = 100,
    steps_per_epoch: int = 1000,
    batch_size: int = 256,
    gamma: float = 0.99,

    actor_learning_rate: float = 3e-4,
    critic_learning_rate: float = 3e-4,

    expectile: float = 0.95,
    weight_temp: float = 3.0,
    max_weight: float = 100.0,
    tau: float = 0.005,
    n_critics: int = 2,

    hidden_units: Sequence[int] = (256, 256),
    use_layer_norm: bool = True,

    standardize_observations: bool = False,
    standardize_rewards: bool = False,

    obs_cols: Sequence[str] = OBS_COLS,
    action_cols: Sequence[str] = ACTION_COLS,
    reward_col: str = "reward",

    checkpoint_every: int = 10,
    seed: int = 0,
    device: str | None = None,
    show_progress: bool = True,
):
    """
    Train IQL for exactly:
        total_steps = epochs * steps_per_epoch

    Saves:
        output_dir/run_name/checkpoints/epoch_XXXX.d3
        output_dir/run_name/final_model.d3
        output_dir/run_name/training_config.json
    """

    if epochs < 1:
        raise ValueError("epochs must be >= 1.")
    if steps_per_epoch < 1:
        raise ValueError("steps_per_epoch must be >= 1.")
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be >= 1.")

    dataset = build_dataset(
        data_path,
        obs_cols=obs_cols,
        action_cols=action_cols,
        reward_col=reward_col,
    )

    run_dir = Path(output_dir) / run_name
    checkpoint_dir = run_dir / "checkpoints"
    log_dir = run_dir / "logs"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    d3rlpy.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if device is None:
        device = (
            "cuda:0"
            if torch.cuda.is_available()
            else "cpu:0"
        )

    def encoder():
        return d3rlpy.models.VectorEncoderFactory(
            list(hidden_units),
            use_layer_norm=use_layer_norm,
        )

    observation_scaler = (
        d3rlpy.preprocessing.StandardObservationScaler()
        if standardize_observations
        else None
    )

    reward_scaler = (
        d3rlpy.preprocessing.StandardRewardScaler()
        if standardize_rewards
        else None
    )

    # External HOFLON actions are [0,1].
    # d3rlpy maps them internally to [-1,1].
    action_scaler = d3rlpy.preprocessing.MinMaxActionScaler(
        minimum=np.zeros(len(action_cols), dtype=np.float32),
        maximum=np.ones(len(action_cols), dtype=np.float32),
    )

    config = d3rlpy.algos.IQLConfig(
        batch_size=batch_size,
        gamma=gamma,
        actor_learning_rate=actor_learning_rate,
        critic_learning_rate=critic_learning_rate,
        actor_encoder_factory=encoder(),
        critic_encoder_factory=encoder(),
        value_encoder_factory=encoder(),
        observation_scaler=observation_scaler,
        action_scaler=action_scaler,
        reward_scaler=reward_scaler,
        tau=tau,
        n_critics=n_critics,
        expectile=expectile,
        weight_temp=weight_temp,
        max_weight=max_weight,
    )

    agent = config.create(device=device)

    total_steps = epochs * steps_per_epoch

    config_record = {
        "algorithm": "IQL",
        "data_path": str(data_path),
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "batch_size": batch_size,
        "gamma": gamma,
        "actor_learning_rate": actor_learning_rate,
        "critic_learning_rate": critic_learning_rate,
        "expectile": expectile,
        "weight_temp": weight_temp,
        "max_weight": max_weight,
        "tau": tau,
        "n_critics": n_critics,
        "hidden_units": list(hidden_units),
        "use_layer_norm": use_layer_norm,
        "standardize_observations": standardize_observations,
        "standardize_rewards": standardize_rewards,
        "checkpoint_every": checkpoint_every,
        "seed": seed,
        "device": str(device),
        "obs_cols": list(obs_cols),
        "action_cols": list(action_cols),
        "reward_col": reward_col,
    }

    with open(
        run_dir / "training_config.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(config_record, f, indent=2)

    print("\n" + "=" * 60)
    print("IQL TRAINING")
    print("=" * 60)
    print(f"run_name         : {run_name}")
    print(f"device           : {device}")
    print(f"epochs           : {epochs}")
    print(f"steps_per_epoch  : {steps_per_epoch}")
    print(f"total_steps      : {total_steps:,}")
    print(f"batch_size       : {batch_size}")
    print(f"gamma            : {gamma}")
    print(f"expectile        : {expectile}")
    print(f"weight_temp      : {weight_temp}")
    print(f"checkpoint_every : {checkpoint_every}")
    print("=" * 60 + "\n")

    saved_checkpoints = []

    for epoch, metrics in agent.fitter(
        dataset,
        n_steps=total_steps,
        n_steps_per_epoch=steps_per_epoch,
        experiment_name="train",
        with_timestamp=False,
        logger_adapter=d3rlpy.logging.FileAdapterFactory(
            root_dir=str(log_dir)
        ),
        save_interval=epochs + 1,
        show_progress=show_progress,
    ):
        epoch = int(epoch)

        loss_text = " ".join(
            f"{k}={float(v):.5g}"
            for k, v in metrics.items()
            if isinstance(v, (int, float, np.floating))
        )

        print(
            f"epoch {epoch:4d}/{epochs} | {loss_text}"
        )

        if (
            epoch % checkpoint_every == 0
            or epoch == epochs
        ):
            path = (
                checkpoint_dir
                / f"epoch_{epoch:04d}.d3"
            )

            agent.save(str(path))
            saved_checkpoints.append(path)

            print(f"saved checkpoint: {path}")

    final_path = run_dir / "final_model.d3"
    agent.save(str(final_path))

    print(f"\nFinal IQL model: {final_path}")

    return agent, saved_checkpoints


def load_iql(
    model_path: str | Path,
    *,
    device: str | None = None,
):
    if device is None:
        device = (
            "cuda:0"
            if torch.cuda.is_available()
            else "cpu:0"
        )

    return d3rlpy.load_learnable(
        str(model_path),
        device=device,
    )
