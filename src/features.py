from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator


RDLogger.DisableLog("rdApp.warning")

DESCRIPTORS: dict[str, Callable[[Chem.Mol], float]] = {
    "mol_wt": Descriptors.MolWt,
    "mol_logp": Descriptors.MolLogP,
    "tpsa": Descriptors.TPSA,
    "h_donors": Descriptors.NumHDonors,
    "h_acceptors": Descriptors.NumHAcceptors,
    "rotatable_bonds": Descriptors.NumRotatableBonds,
    "ring_count": Descriptors.RingCount,
    "fraction_csp3": Descriptors.FractionCSP3,
    "heavy_atoms": Descriptors.HeavyAtomCount,
    "aromatic_rings": Descriptors.NumAromaticRings,
    "heteroatoms": Descriptors.NumHeteroatoms,
    "labute_asa": Descriptors.LabuteASA,
}


def _molecule_features(
    smiles: str,
    n_bits: int,
    radius: int,
    prefix: str,
) -> tuple[list[str], np.ndarray]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES reached feature generation: {smiles}")

    if n_bits > 0:
        generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=n_bits
        )
        fingerprint = generator.GetFingerprintAsNumPy(molecule).astype(np.float32)
    else:
        fingerprint = np.empty(0, dtype=np.float32)
    descriptor_values = np.asarray(
        [function(molecule) for function in DESCRIPTORS.values()],
        dtype=np.float32,
    )
    descriptor_values = np.nan_to_num(
        descriptor_values, nan=0.0, posinf=0.0, neginf=0.0
    )

    names = [f"{prefix}_morgan_{i:04d}" for i in range(n_bits)]
    names.extend(f"{prefix}_{name}" for name in DESCRIPTORS)
    return names, np.concatenate([fingerprint, descriptor_values])


def build_feature_matrix(
    frame: pd.DataFrame,
    drug_bits: int = 1024,
    solvent_bits: int = 256,
    radius: int = 2,
    include_solvent_identity: bool = False,
    include_interactions: bool = False,
    include_physchem_interactions: bool = False,
    include_temperature_interactions: bool = False,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    feature_rows: dict[tuple[str, str], tuple[list[str], np.ndarray]] = {}

    for smiles in frame["drug_smiles_canonical"].unique():
        feature_rows[("drug", smiles)] = _molecule_features(
            smiles, n_bits=drug_bits, radius=radius, prefix="drug"
        )
    for smiles in frame["solvent_smiles_canonical"].unique():
        feature_rows[("solvent", smiles)] = _molecule_features(
            smiles, n_bits=solvent_bits, radius=radius, prefix="solvent"
        )

    drug_features = pd.DataFrame(
        [
            feature_rows[("drug", smiles)][1]
            for smiles in frame["drug_smiles_canonical"]
        ],
        columns=feature_rows[("drug", frame["drug_smiles_canonical"].iloc[0])][0],
        index=frame.index,
        dtype=np.float32,
    )
    solvent_features = pd.DataFrame(
        [
            feature_rows[("solvent", smiles)][1]
            for smiles in frame["solvent_smiles_canonical"]
        ],
        columns=feature_rows[
            ("solvent", frame["solvent_smiles_canonical"].iloc[0])
        ][0],
        index=frame.index,
        dtype=np.float32,
    )

    temperature = frame["Temperature"].to_numpy(dtype=np.float32)
    temperature_k = temperature + np.float32(273.15)
    temperature_features = pd.DataFrame(
        {
            "temperature_c": temperature,
            "temperature_k": temperature_k,
            "inverse_temperature_k": 1.0 / temperature_k,
        },
        index=frame.index,
        dtype=np.float32,
    )

    extra_features: list[pd.DataFrame] = []
    solvent_identity_columns: list[str] = []
    physchem_interaction_columns: list[str] = []
    temperature_interaction_columns: list[str] = []

    if include_solvent_identity:
        solvent_identity = pd.get_dummies(
            frame["solvent_smiles_canonical"],
            prefix="solvent_identity",
            dtype=np.float32,
        )
        solvent_identity.index = frame.index
        solvent_identity_columns = solvent_identity.columns.tolist()
        extra_features.append(solvent_identity)

    if include_interactions:
        include_physchem_interactions = True
        include_temperature_interactions = True

    if include_physchem_interactions:
        temperature_c = temperature_features["temperature_c"]
        inverse_temperature = temperature_features["inverse_temperature_k"]
        physchem_interactions = pd.DataFrame(
            {
                "interaction_logp_difference": np.abs(
                    drug_features["drug_mol_logp"]
                    - solvent_features["solvent_mol_logp"]
                ),
                "interaction_tpsa_difference": np.abs(
                    drug_features["drug_tpsa"]
                    - solvent_features["solvent_tpsa"]
                ),
                "interaction_logp_product": (
                    drug_features["drug_mol_logp"]
                    * solvent_features["solvent_mol_logp"]
                ),
                "interaction_tpsa_product": (
                    drug_features["drug_tpsa"]
                    * solvent_features["solvent_tpsa"]
                ),
            },
            index=frame.index,
            dtype=np.float32,
        )
        physchem_interactions = physchem_interactions.replace(
            [np.inf, -np.inf], 0.0
        ).fillna(0.0)
        physchem_interaction_columns = (
            physchem_interactions.columns.tolist()
        )
        extra_features.append(physchem_interactions)

    if include_temperature_interactions:
        temperature_c = temperature_features["temperature_c"]
        inverse_temperature = temperature_features["inverse_temperature_k"]
        temperature_interactions = pd.DataFrame(
            {
                "interaction_temperature_drug_logp": (
                    temperature_c * drug_features["drug_mol_logp"]
                ),
                "interaction_temperature_solvent_logp": (
                    temperature_c * solvent_features["solvent_mol_logp"]
                ),
                "interaction_inverse_temperature_drug_tpsa": (
                    inverse_temperature * drug_features["drug_tpsa"]
                ),
                "interaction_inverse_temperature_solvent_tpsa": (
                    inverse_temperature * solvent_features["solvent_tpsa"]
                ),
            },
            index=frame.index,
            dtype=np.float32,
        )
        temperature_interactions = (
            temperature_interactions.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        )
        temperature_interaction_columns = (
            temperature_interactions.columns.tolist()
        )
        extra_features.append(temperature_interactions)

    features = pd.concat(
        [drug_features, solvent_features, temperature_features, *extra_features],
        axis=1,
    )
    column_groups = {
        "drug": drug_features.columns.tolist(),
        "solvent": solvent_features.columns.tolist(),
        "temperature": temperature_features.columns.tolist(),
        "solvent_identity": solvent_identity_columns,
        "physchem_interaction": physchem_interaction_columns,
        "temperature_interaction": temperature_interaction_columns,
    }
    return features, column_groups
