from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.features import build_feature_matrix
from src.metrics import regression_metrics
from src.splits import load_folds
from src.torch_models import TwoTowerRegressor


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
MODELS_DIR = RESULTS_DIR / "models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="mlp_v0")
    parser.add_argument("--seeds", default="42", help="Comma-separated seeds")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--encoder-dropout", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--interaction-dim", type=int, default=0)
    parser.add_argument(
        "--architecture",
        choices=("concat", "interaction"),
        default="concat",
    )
    parser.add_argument("--loss", choices=("mse", "huber"), default="mse")
    parser.add_argument("--drug-bits", type=int, default=1024)
    parser.add_argument("--solvent-bits", type=int, default=256)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def to_tensor(values: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(values, dtype=np.float32))


def split_columns(
    frame: pd.DataFrame,
    column_groups: dict[str, list[str]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        frame[column_groups["drug"]].to_numpy(dtype=np.float32),
        frame[column_groups["solvent"]].to_numpy(dtype=np.float32),
        frame[column_groups["temperature"]].to_numpy(dtype=np.float32),
    )


def scale_feature_blocks(
    train_blocks: tuple[np.ndarray, np.ndarray, np.ndarray],
    other_blocks: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray],
    list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    list[StandardScaler],
]:
    scalers = [
        StandardScaler(),
        StandardScaler(),
        StandardScaler(),
    ]
    scaled_train = tuple(
        scaler.fit_transform(block).astype(np.float32)
        for scaler, block in zip(scalers, train_blocks, strict=True)
    )
    scaled_others = [
        tuple(
            scaler.transform(block).astype(np.float32)
            for scaler, block in zip(scalers, blocks, strict=True)
        )
        for blocks in other_blocks
    ]
    return scaled_train, scaled_others, scalers


def train_one_fold(
    feature_frame: pd.DataFrame,
    column_groups: dict[str, list[str]],
    frame: pd.DataFrame,
    target: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    seed: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, float | int], list[dict[str, float]]]:
    set_seed(seed)
    train_indices = np.flatnonzero(train_mask)
    inner_split = GroupShuffleSplit(
        n_splits=1,
        test_size=args.validation_fraction,
        random_state=seed,
    )
    fit_positions, validation_positions = next(
        inner_split.split(
            train_indices,
            groups=frame.iloc[train_indices]["drug_smiles_canonical"],
        )
    )
    fit_indices = train_indices[fit_positions]
    validation_indices = train_indices[validation_positions]
    test_indices = np.flatnonzero(test_mask)

    fit_mask = np.zeros(len(frame), dtype=bool)
    validation_mask = np.zeros(len(frame), dtype=bool)
    fit_mask[fit_indices] = True
    validation_mask[validation_indices] = True

    fit_blocks = split_columns(feature_frame.loc[fit_mask], column_groups)
    validation_blocks = split_columns(
        feature_frame.loc[validation_mask], column_groups
    )
    test_blocks = split_columns(feature_frame.loc[test_mask], column_groups)
    (
        scaled_fit,
        scaled_others,
        _,
    ) = scale_feature_blocks(fit_blocks, [validation_blocks, test_blocks])
    scaled_validation, scaled_test = scaled_others

    y_fit = target[fit_mask]
    y_validation = target[validation_mask]
    target_mean = float(y_fit.mean())
    target_std = float(y_fit.std())
    if target_std <= 1e-8:
        target_std = 1.0
    y_fit_scaled = (y_fit - target_mean) / target_std
    y_validation_scaled = (y_validation - target_mean) / target_std

    fit_dataset = TensorDataset(
        *[to_tensor(block) for block in scaled_fit],
        to_tensor(y_fit_scaled),
    )
    generator = torch.Generator().manual_seed(seed)
    fit_loader = DataLoader(
        fit_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_tensors = [to_tensor(block) for block in scaled_validation]
    y_validation_tensor = to_tensor(y_validation_scaled)
    test_tensors = [to_tensor(block) for block in scaled_test]

    model = TwoTowerRegressor(
        drug_dim=scaled_fit[0].shape[1],
        solvent_dim=scaled_fit[1].shape[1],
        temperature_dim=scaled_fit[2].shape[1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        encoder_dropout=args.encoder_dropout,
        interaction_dim=(
            args.interaction_dim if args.architecture == "interaction" else 0
        ),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    loss_function = (
        nn.HuberLoss(delta=1.0) if args.loss == "huber" else nn.MSELoss()
    )
    best_state = {
        key: value.detach().clone() for key, value in model.state_dict().items()
    }
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        training_loss_sum = 0.0
        for batch in fit_loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(*batch[:3])
            loss = loss_function(prediction, batch[3])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            training_loss_sum += float(loss.item()) * len(batch[3])

        model.eval()
        with torch.no_grad():
            validation_prediction = model(*validation_tensors)
            validation_loss = float(
                loss_function(validation_prediction, y_validation_tensor).item()
            )
        training_loss = training_loss_sum / len(fit_dataset)
        history.append(
            {
                "epoch": epoch,
                "train_mse_scaled": training_loss,
                "validation_mse_scaled": validation_loss,
            }
        )
        if validation_loss < best_validation_loss - 1e-7:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prediction_scaled = model(*test_tensors).cpu().numpy()
    prediction = prediction_scaled * target_std + target_mean
    metadata = {
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "best_validation_mse_scaled": best_validation_loss,
        "fit_rows": int(fit_mask.sum()),
        "validation_rows": int(validation_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "fit_drug_groups": int(
            frame.loc[fit_mask, "drug_smiles_canonical"].nunique()
        ),
        "validation_drug_groups": int(
            frame.loc[validation_mask, "drug_smiles_canonical"].nunique()
        ),
        "test_drug_groups": int(
            frame.loc[test_mask, "drug_smiles_canonical"].nunique()
        ),
    }
    return prediction, metadata, history


def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(1, min(8, (os.cpu_count() or 2) // 2)))
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    frame = pd.read_csv(RESULTS_DIR / "clean.csv", encoding="utf-8-sig")
    assignment = load_folds(frame, RESULTS_DIR / "folds.json")
    feature_frame, column_groups = build_feature_matrix(
        frame,
        drug_bits=args.drug_bits,
        solvent_bits=args.solvent_bits,
    )
    target = frame["Solubility_log10"].to_numpy(dtype=np.float32)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_rows: list[dict[str, float | int]] = []
    histories: list[dict[str, float | int]] = []
    for fold in sorted(np.unique(assignment)):
        for seed in seeds:
            fold_seed = seed + int(fold) * 1000
            started = time.perf_counter()
            prediction, metadata, history = train_one_fold(
                feature_frame=feature_frame,
                column_groups=column_groups,
                frame=frame,
                target=target,
                train_mask=assignment != fold,
                test_mask=assignment == fold,
                seed=fold_seed,
                args=args,
            )
            test_mask = assignment == fold
            metrics = regression_metrics(target[test_mask], prediction)
            elapsed = time.perf_counter() - started
            metrics_rows.append(
                {
                    "model": args.run_name,
                    "fold": int(fold),
                    **metadata,
                    **metrics,
                    "seconds": elapsed,
                }
            )
            histories.extend(
                {"fold": int(fold), **row} for row in history
            )
            pd.DataFrame(
                {
                    "row_id": frame.loc[test_mask, "row_id"],
                    "fold": int(fold),
                    "seed": seed,
                    "drug_smiles_canonical": frame.loc[
                        test_mask, "drug_smiles_canonical"
                    ],
                    "y_true": target[test_mask],
                    "y_pred": prediction,
                    "residual": target[test_mask] - prediction,
                }
            ).to_csv(
                PREDICTIONS_DIR / f"{args.run_name}_fold{fold}_seed{seed}.csv",
                index=False,
                encoding="utf-8-sig",
            )
            print(
                f"{args.run_name} fold {fold} seed {seed}: "
                f"RMSE={metrics['rmse']:.4f} "
                f"MAE={metrics['mae']:.4f} "
                f"R2={metrics['r2']:.4f} "
                f"best_epoch={metadata['best_epoch']} "
                f"({elapsed:.1f}s)"
            )

    metrics_frame = pd.DataFrame(metrics_rows)
    metrics_frame.to_csv(
        RESULTS_DIR / f"{args.run_name}_metrics.csv", index=False
    )
    pd.DataFrame(histories).to_csv(
        RESULTS_DIR / f"{args.run_name}_history.csv", index=False
    )
    summary = (
        metrics_frame.groupby("model")[["rmse", "mae", "r2"]]
        .agg(["mean", "std"])
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.reset_index().to_csv(
        RESULTS_DIR / f"{args.run_name}_summary.csv", index=False
    )
    (RESULTS_DIR / f"{args.run_name}_config.json").write_text(
        json.dumps(vars(args), indent=2), encoding="utf-8"
    )
    print(f"\n{args.run_name} summary")
    print(summary.to_string())


if __name__ == "__main__":
    main()
