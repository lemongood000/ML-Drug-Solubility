from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import canonicalize_smiles
from src.batch import predict_batch
from src.prediction import (
    load_model_bundle,
    molecule_png,
    predict_solvents,
    predict_temperature_curve,
)


st.set_page_config(
    page_title="Drug Solubility Explorer",
    page_icon="🧪",
    layout="wide",
)


@st.cache_resource
def load_bundle():
    return load_model_bundle(ROOT / "artifacts" / "solubility_models.joblib")


bundle = load_bundle()


@st.cache_data(show_spinner=False, ttl=3600)
def cached_ranking(drug_smiles: str, temperature_c: float) -> pd.DataFrame:
    return predict_solvents(bundle, drug_smiles, temperature_c)


@st.cache_data(show_spinner=False, ttl=3600)
def cached_temperature_curve(
    drug_smiles: str,
    solvent_smiles: str,
    temperatures_c: tuple[float, ...],
) -> pd.DataFrame:
    return predict_temperature_curve(
        bundle,
        drug_smiles,
        solvent_smiles,
        np.asarray(temperatures_c),
    )


st.title("Drug Solubility Explorer")
st.caption(
    "Predict logarithmic molar solubility from drug SMILES, solvent, and temperature."
)

with st.sidebar:
    st.header("Input")
    default_smiles = "CC(=O)NC1=CC=C(C=C1)O"
    drug_smiles = st.text_area(
        "Drug SMILES",
        value=default_smiles,
        height=120,
        help="The default is paracetamol.",
    )
    temperature = st.slider(
        "Temperature (°C)",
        min_value=-60.0,
        max_value=130.0,
        value=25.0,
        step=0.5,
    )
    st.button("Predict solubility", width="stretch")
    st.divider()
    st.caption(
        f"Model bundle: {bundle['training_rows']} records · "
        f"{bundle['feature_count']} features"
    )
    cv_reference = bundle.get("cv_reference", {})
    if cv_reference:
        st.caption(
            f"Cross-validation: RMSE {cv_reference['rmse_mean']:.3f} · "
            f"MAE {cv_reference['mae_mean']:.3f}"
        )
    st.caption("Predictions combine HistGradientBoosting, XGBoost, and Random Forest.")
    st.caption(
        f"Applicability-domain threshold: "
        f"{bundle['ad_threshold']:.2f} nearest-drug Tanimoto."
    )

canonical = canonicalize_smiles(drug_smiles.strip())
if canonical is None:
    st.error("RDKit could not parse this SMILES. Please check the molecular string.")
    st.stop()

left, right = st.columns([1, 2])
with left:
    st.subheader("Molecule")
    st.image(molecule_png(canonical), width="stretch")
    st.code(canonical, language=None)

try:
    ranking = cached_ranking(canonical, float(temperature))
except Exception as exc:
    st.error(f"Prediction failed: {exc}")
    st.stop()

selected_solvent = st.selectbox(
    "Solvent for detailed prediction", ranking["Solvent"].tolist()
)
selected = ranking.loc[ranking["Solvent"] == selected_solvent].iloc[0]

with right:
    st.subheader("Prediction")
    metric_columns = st.columns(3)
    metric_columns[0].metric("log10 solubility", f"{selected['log10_solubility']:.3f}")
    metric_columns[1].metric("Solubility", f"{selected['solubility_mol_l']:.4g} mol/L")
    metric_columns[2].metric(
        "Model disagreement", f"{selected['model_disagreement']:.3f}"
    )
    if "lower_90" in selected.index:
        st.metric(
            "90% prediction interval",
            f"{selected['lower_90']:.2f} to {selected['upper_90']:.2f}",
        )

    if selected["applicability_status"] == "Within applicability domain":
        st.success(
            f"Supported by known chemistry (nearest-drug Tanimoto "
            f"{selected['nearest_drug_similarity']:.2f}); temperature is "
            "inside the training range."
        )
    else:
        st.warning(
            "Possible extrapolation: nearest-drug similarity is "
            f"{selected['nearest_drug_similarity']:.2f}; temperature status "
            f"is {selected['temperature_applicability_status'].lower()}."
        )

st.subheader("Temperature profile")
temperatures = np.linspace(-20.0, 100.0, 49)
curve = cached_temperature_curve(
    canonical,
    selected["solvent_smiles_canonical"],
    tuple(float(value) for value in temperatures),
).set_index("Temperature")
curve = curve[["log10_solubility"]].rename(
    columns={"log10_solubility": "log10 solubility"}
)
st.line_chart(curve, height=300)

st.subheader("Solvent ranking")
st.caption("Predictions cover the 49 solvents present in the training data.")
display_columns = [
    "Solvent",
    "log10_solubility",
    "solubility_mol_l",
    "model_disagreement",
    "nearest_drug_similarity",
    "temperature_applicability_status",
    "applicability_status",
]
st.dataframe(
    ranking[display_columns].rename(
        columns={
            "log10_solubility": "log10 solubility",
            "solubility_mol_l": "mol/L",
            "model_disagreement": "model disagreement",
            "nearest_drug_similarity": "nearest-drug similarity",
            "temperature_applicability_status": "temperature range",
            "applicability_status": "status",
        }
    ),
    width="stretch",
    hide_index=True,
)

csv_buffer = io.StringIO()
ranking[display_columns].to_csv(csv_buffer, index=False)
st.download_button(
    "Download solvent ranking",
    data=csv_buffer.getvalue().encode("utf-8"),
    file_name="solvent_ranking.csv",
    mime="text/csv",
)

st.divider()
st.subheader("Batch prediction")
st.caption(
    "Upload a CSV with Drug SMILES, Solvent, and Temperature columns. "
    "Only the 49 training solvents are supported."
)
template_buffer = io.StringIO()
pd.DataFrame(
    {
        "Drug SMILES": [default_smiles],
        "Solvent": [bundle["solvents"]["Solvent"].iloc[0]],
        "Temperature": [25.0],
    }
).to_csv(template_buffer, index=False)
st.download_button(
    "Download CSV template",
    data=template_buffer.getvalue().encode("utf-8"),
    file_name="solubility_batch_template.csv",
    mime="text/csv",
)
uploaded = st.file_uploader("Upload batch CSV", type=["csv"])
if uploaded is not None:
    try:
        batch_input = pd.read_csv(uploaded, encoding="utf-8-sig")
        batch_output = predict_batch(bundle, batch_input)
        st.dataframe(batch_output, width="stretch", hide_index=True)
        batch_buffer = io.StringIO()
        batch_output.to_csv(batch_buffer, index=False)
        st.download_button(
            "Download batch predictions",
            data=batch_buffer.getvalue().encode("utf-8"),
            file_name="solubility_batch_predictions.csv",
            mime="text/csv",
        )
    except Exception as exc:
        st.error(f"Batch prediction failed: {exc}")
