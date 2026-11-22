from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from src.data import assert_no_group_leakage
from src.features import build_feature_matrix
from src.prediction import (
    load_model_bundle,
    predict_solvents,
    predict_temperature_curve,
)
from src.splits import load_folds


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
ARTIFACTS_DIR = ROOT / "artifacts"


def main() -> None:
    clean = pd.read_csv(RESULTS_DIR / "clean.csv", encoding="utf-8-sig")
    assignment = load_folds(clean, RESULTS_DIR / "folds.json")
    assert clean["row_id"].is_unique
    assert assignment.shape == (len(clean),)
    assert_no_group_leakage(clean, assignment, "drug_smiles_canonical")
    assert np.isfinite(clean["Solubility_log10"]).all()
    assert np.isfinite(clean["Temperature"]).all()

    bundle = load_model_bundle(ARTIFACTS_DIR / "solubility_models.joblib")
    features, groups = build_feature_matrix(
        clean,
        drug_bits=bundle["drug_bits"],
        solvent_bits=bundle["solvent_bits"],
        include_solvent_identity=bundle["include_solvent_identity"],
        include_physchem_interactions=bundle["include_physchem_interactions"],
        include_temperature_interactions=bundle["include_temperature_interactions"],
        include_maccs=bundle.get("include_maccs", False),
    )
    assert np.isfinite(features.to_numpy()).all()
    assert len(clean) == features.shape[0]
    assert features.shape[1] == bundle["feature_count"]
    assert groups["drug"] and groups["solvent"] and groups["temperature"]

    default_smiles = "CC(=O)NC1=CC=C(C=C1)O"
    ranking = predict_solvents(bundle, default_smiles, 25.0)
    assert len(ranking) == len(bundle["solvents"])
    assert np.isfinite(ranking["log10_solubility"]).all()
    assert (ranking["model_disagreement"] >= 0).all()
    assert ranking["log10_solubility"].is_monotonic_decreasing
    assert 0.0 <= bundle["ad_threshold"] <= 1.0
    assert "manifest" in bundle
    assert bundle["conformal"]["mean_coverage_80"] >= 0.75
    assert bundle["conformal"]["mean_coverage_90"] >= 0.85

    selected = ranking.iloc[0]
    temperatures = np.linspace(-20.0, 100.0, 25)
    curve = predict_temperature_curve(
        bundle,
        default_smiles,
        selected["solvent_smiles_canonical"],
        temperatures,
    )
    assert len(curve) == len(temperatures)
    assert np.isfinite(curve["log10_solubility"]).all()
    ensemble = pd.read_csv(RESULTS_DIR / "ensemble_metrics.csv")
    assert len(ensemble) == 5
    assert np.isfinite(ensemble[["rmse", "mae", "r2"]]).all().all()

    app = AppTest.from_file(str(ROOT / "app" / "app.py"), default_timeout=90).run()
    assert not app.exception, [item.value for item in app.exception]
    assert len(app.metric) >= 3

    print(
        "SELF_CHECK_OK "
        f"rows={len(clean)} "
        f"folds={assignment.max() + 1} "
        f"features={features.shape[1]} "
        f"solvents={len(ranking)}"
    )


if __name__ == "__main__":
    main()
