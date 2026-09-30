
"""
train_hybrid_bc_sweep.py

Build four hybrid reward datasets and train one weighted-BC actor for each.

Hybrid reward:
    R = R_abs + lambda * R_change

where:
    E_t       = err_P(t)^2 + err_T(t)^2
    R_abs     = MinMax(-E_{t+1})
    R_change  = MinMax(E_t - E_{t+1})

Lambdas:
    0.05, 0.10, 0.20, 0.50

The resulting hybrid reward is saved in the `reward` column.
`train_weighted_bc.py` then min-maxes that final reward to [0,1]
and uses it as the BC sample weight.

Default actor:
    9 -> 32 -> 32 -> 2
    100 epochs
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from train_weighted_bc import train_weighted_bc


LAMBDAS = [0.05, 0.10, 0.20, 0.50]


def minmax(series: pd.Series) -> pd.Series:
    """Min-max scale a pandas Series to [0,1]."""

    x_min = float(series.min())
    x_max = float(series.max())

    if np.isclose(x_max, x_min):
        return pd.Series(
            np.ones(len(series), dtype=np.float32),
            index=series.index,
        )

    return (
        (series - x_min)
        / (x_max - x_min)
    ).astype(np.float32)


def build_hybrid_datasets(
    base_dataset: str | Path,
    output_dir: str | Path = "hybrid_reward_datasets",
    lambdas: list[float] | None = None,
) -> dict[float, Path]:
    """
    Build all hybrid reward datasets from a processed Idea-2-scaled dataset.
    """

    lambdas = LAMBDAS if lambdas is None else list(lambdas)

    base_dataset = Path(base_dataset)
    output_dir = Path(output_dir)

    if not base_dataset.exists():
        raise FileNotFoundError(
            f"Base dataset not found: {base_dataset}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(base_dataset)

    required = {"run_id", "step", "err_P", "err_T"}
    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    # Ensure t -> t+1 is always within the same trajectory.
    df = (
        df.sort_values(["run_id", "step"])
        .reset_index(drop=True)
        .copy()
    )

    # Tracking error in fixed Idea-2 scaled coordinates.
    df["E_t"] = (
        df["err_P"].astype(float) ** 2
        + df["err_T"].astype(float) ** 2
    )

    df["E_next"] = (
        df.groupby("run_id")["E_t"]
        .shift(-1)
    )

    # Last row of each trajectory has no valid next state.
    df = (
        df.dropna(subset=["E_next"])
        .reset_index(drop=True)
        .copy()
    )

    # Idea 3 component: good next state.
    df["r_abs_raw"] = -df["E_next"]

    # Idea 4 component: improvement over transition.
    df["r_change_raw"] = (
        df["E_t"]
        - df["E_next"]
    )

    # Normalize components separately before combining.
    df["r_abs_norm"] = minmax(df["r_abs_raw"])
    df["r_change_norm"] = minmax(df["r_change_raw"])

    print(
        "Absolute component range:",
        (
            float(df["r_abs_norm"].min()),
            float(df["r_abs_norm"].max()),
        ),
    )
    print(
        "Change component range:",
        (
            float(df["r_change_norm"].min()),
            float(df["r_change_norm"].max()),
        ),
    )

    saved: dict[float, Path] = {}

    for lam in lambdas:
        hybrid = df.copy()

        hybrid["reward"] = (
            hybrid["r_abs_norm"]
            + float(lam) * hybrid["r_change_norm"]
        )

        label = f"{lam:g}".replace(".", "p")

        path = output_dir / f"hybrid_lambda_{label}.csv"

        hybrid.to_csv(path, index=False)
        saved[lam] = path

        print(f"Saved lambda={lam:g} dataset -> {path}")

    return saved


def train_hybrid_sweep(
    base_dataset: str | Path = "idea3_mse_processed.csv",
    dataset_dir: str | Path = "hybrid_reward_datasets",
    model_dir: str | Path = "hybrid_bc_models",
    lambdas: list[float] | None = None,
    epochs: int = 100,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    seed: int = 123,
    device: str | None = None,
):
    """
    Build the hybrid datasets and train one BC actor per lambda.
    """

    lambdas = LAMBDAS if lambdas is None else list(lambdas)

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    datasets = build_hybrid_datasets(
        base_dataset=base_dataset,
        output_dir=dataset_dir,
        lambdas=lambdas,
    )

    models = {}
    histories = {}

    for lam in lambdas:
        label = f"{lam:g}".replace(".", "p")
        output_model = model_dir / f"hybrid_bc_lambda_{label}.pth"

        print()
        print("=" * 60)
        print(f"Training hybrid weighted BC | lambda = {lam:g}")
        print("=" * 60)

        model, history = train_weighted_bc(
            dataset_path=datasets[lam],
            output_path=output_model,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
            device=device,
        )

        models[lam] = model
        histories[lam] = history

    print()
    print("Finished all hybrid BC models.")

    return models, histories, datasets


if __name__ == "__main__":
    train_hybrid_sweep()
