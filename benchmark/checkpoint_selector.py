from pathlib import Path
import re

import d3rlpy
import matplotlib.pyplot as plt
import numpy as np

from poly_reactor_env import PolymerisationEnv


# Idea 3 observation layout:
# [A, I, P, R, T, err_P, err_T, int_err_P, int_err_T]
ERR_P_IDX = 5
ERR_T_IDX = 6


def _checkpoint_number(path: Path):
    """Extract the final integer in a checkpoint filename for plotting/sorting."""
    numbers = re.findall(r"\d+", path.stem)
    return int(numbers[-1]) if numbers else None


def _evaluate_checkpoint(model_path, seeds, device):
    """
    Evaluate one d3rlpy checkpoint using the settled Idea 3 environment.

    Idea 3 convention:
        s_t --a_t--> s_(t+1)

        r_t = -(err_P,t+1^2 + err_T,t+1^2)

    ISE is therefore accumulated from the NEXT state's scaled tracking errors.
    No dataset is passed into the environment.
    """
    model = d3rlpy.load_learnable(
        str(model_path),
        device=device,
    )

    run_ises = []

    for seed in seeds:
        env = PolymerisationEnv(seed=seed)
        obs, _ = env.reset(seed=seed)

        terminated = False
        truncated = False

        ise = 0.0
        dt_hours = float(env.dt) / 3600.0

        while not (terminated or truncated):
            obs_batch = np.asarray(obs, dtype=np.float32).reshape(1, -1)

            action = model.predict(obs_batch)[0]
            action = np.clip(action, 0.0, 1.0)

            next_obs, _, terminated, truncated, _ = env.step(action)

            # Idea 3: evaluate the consequence of a_t using s_(t+1).
            err_P_next = float(next_obs[ERR_P_IDX])
            err_T_next = float(next_obs[ERR_T_IDX])

            ise += (
                err_P_next ** 2
                + err_T_next ** 2
            ) * dt_hours

            obs = next_obs

        run_ises.append(ise)

        if hasattr(env, "close"):
            env.close()

    return float(np.mean(run_ises))


def select_best_checkpoint(
    folder_name,
    n_runs=5,
    seed_start=1000,
    device="cuda:0",
):
    """
    Evaluate every d3rlpy .d3 checkpoint in a folder.

    Parameters
    ----------
    folder_name : str or Path
        Folder containing checkpoint .d3 files.

    n_runs : int, default=5
        Number of fixed evaluation seeds used for every checkpoint.

    seed_start : int, default=1000
        First evaluation seed. For example:
            n_runs=5, seed_start=11
        uses seeds:
            11, 12, 13, 14, 15

    device : str, default="cuda:0"
        Device used to load d3rlpy models.

    Output
    ------
    Saves:
        checkpoint_ise_curve.png

    Prints only:
        checkpoint filename with the lowest mean ISE

    Returns
    -------
    str
        Best checkpoint filename.
    """
    folder = Path(folder_name)

    if not folder.exists():
        raise FileNotFoundError(f"Checkpoint folder does not exist: {folder}")

    checkpoints = list(folder.glob("*.d3"))

    if not checkpoints:
        raise FileNotFoundError(f"No .d3 checkpoints found in: {folder}")

    # Sort numbered checkpoints numerically.
    checkpoints = sorted(
        checkpoints,
        key=lambda p: (
            _checkpoint_number(p) is None,
            _checkpoint_number(p)
            if _checkpoint_number(p) is not None
            else float("inf"),
            p.name,
        ),
    )

    seeds = [seed_start + i for i in range(n_runs)]

    x_values = []
    ise_values = []
    checkpoint_names = []

    for i, checkpoint in enumerate(checkpoints):
        mean_ise = _evaluate_checkpoint(
            model_path=checkpoint,
            seeds=seeds,
            device=device,
        )

        checkpoint_names.append(checkpoint.name)
        ise_values.append(mean_ise)

        checkpoint_num = _checkpoint_number(checkpoint)
        x_values.append(
            checkpoint_num if checkpoint_num is not None else i
        )

    best_idx = int(np.argmin(ise_values))
    best_checkpoint = checkpoint_names[best_idx]

    plt.figure(figsize=(9, 5))
    plt.plot(x_values, ise_values, marker="o")
    plt.xlabel("Checkpoint")
    plt.ylabel("Mean ISE")
    plt.title("ISE vs Training Checkpoint")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plot_path = folder / "checkpoint_ise_curve.png"
    plt.savefig(plot_path, dpi=300)
    plt.show()
    plt.close()

    print(best_checkpoint)

    return best_checkpoint
