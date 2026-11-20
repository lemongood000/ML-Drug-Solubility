from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"

CONFIGS = [
    {
        "run_name": "mlp_desc_interaction_huber",
        "architecture": "interaction",
        "loss": "huber",
        "interaction_dim": 32,
        "hidden_dim": 64,
        "dropout": 0.2,
        "encoder_dropout": 0.0,
        "learning_rate": 5e-4,
        "weight_decay": 1e-3,
        "batch_size": 64,
        "drug_bits": 0,
        "solvent_bits": 0,
    },
    {
        "run_name": "mlp_d256_interaction_huber",
        "architecture": "interaction",
        "loss": "huber",
        "interaction_dim": 32,
        "hidden_dim": 64,
        "dropout": 0.2,
        "encoder_dropout": 0.0,
        "learning_rate": 5e-4,
        "weight_decay": 1e-3,
        "batch_size": 64,
        "drug_bits": 256,
        "solvent_bits": 64,
    },
    {
        "run_name": "mlp_full_interaction_huber",
        "architecture": "interaction",
        "loss": "huber",
        "interaction_dim": 32,
        "hidden_dim": 128,
        "dropout": 0.2,
        "encoder_dropout": 0.0,
        "learning_rate": 3e-4,
        "weight_decay": 1e-3,
        "batch_size": 64,
        "drug_bits": 1024,
        "solvent_bits": 256,
    },
    {
        "run_name": "mlp_full_concat_huber",
        "architecture": "concat",
        "loss": "huber",
        "interaction_dim": 0,
        "hidden_dim": 128,
        "dropout": 0.1,
        "encoder_dropout": 0.0,
        "learning_rate": 3e-4,
        "weight_decay": 1e-3,
        "batch_size": 64,
        "drug_bits": 1024,
        "solvent_bits": 256,
    },
]


def main() -> None:
    summaries: list[pd.DataFrame] = []
    for config in CONFIGS:
        command = [
            sys.executable,
            str(ROOT / "run_mlp.py"),
            "--seeds",
            "42",
            "--epochs",
            "120",
            "--patience",
            "15",
            *[
                argument
                for key, value in config.items()
                if key != "run_name"
                for argument in (f"--{key.replace('_', '-')}", str(value))
            ],
            "--run-name",
            config["run_name"],
        ]
        print("\nRunning", config["run_name"])
        subprocess.run(command, cwd=ROOT, check=True)
        summary = pd.read_csv(
            RESULTS_DIR / f"{config['run_name']}_summary.csv"
        )
        summary.insert(0, "run_name", config["run_name"])
        summaries.append(summary)

    combined = pd.concat(summaries, ignore_index=True).sort_values("rmse_mean")
    combined.to_csv(RESULTS_DIR / "mlp_experiment_summary.csv", index=False)
    print("\nExperiment summary")
    print(combined.to_string(index=False))


if __name__ == "__main__":
    main()
