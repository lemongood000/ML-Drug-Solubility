from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"

CONFIGS = [
    {
        "run_name": "mlp_h128_lr3e4_do03_wd1e3_b64_d512",
        "learning_rate": 3e-4,
        "dropout": 0.3,
        "weight_decay": 1e-3,
        "hidden_dim": 128,
        "batch_size": 64,
        "drug_bits": 512,
        "solvent_bits": 128,
    },
    {
        "run_name": "mlp_h64_lr1e4_do04_wd1e2_b32_d512",
        "learning_rate": 1e-4,
        "dropout": 0.4,
        "weight_decay": 1e-2,
        "hidden_dim": 64,
        "batch_size": 32,
        "drug_bits": 512,
        "solvent_bits": 128,
    },
    {
        "run_name": "mlp_h64_lr5e4_do02_wd1e2_b64_d256",
        "learning_rate": 5e-4,
        "dropout": 0.2,
        "weight_decay": 1e-2,
        "hidden_dim": 64,
        "batch_size": 64,
        "drug_bits": 256,
        "solvent_bits": 64,
    },
    {
        "run_name": "mlp_h64_lr3e4_do05_wd1e2_b32_d256",
        "learning_rate": 3e-4,
        "dropout": 0.5,
        "weight_decay": 1e-2,
        "hidden_dim": 64,
        "batch_size": 32,
        "drug_bits": 256,
        "solvent_bits": 64,
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
            "80",
            "--patience",
            "10",
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
    combined.to_csv(RESULTS_DIR / "mlp_sweep_summary.csv", index=False)
    print("\nSweep summary")
    print(combined.to_string(index=False))


if __name__ == "__main__":
    main()
