"""
Unit and regression tests for chem_utils.py.

These specifically guard against the class of bug this project is named
for: a v1 feature-engineering function silently returned None (or wrong
values) for every real molecule because of two misspelled/nonexistent
RDKit API calls, hidden behind a broad `except Exception: return None`.
That silently dropped 40 of 71 features and left the deployed app's
predictions completely unresponsive to the actual substrate SMILES a
user typed in. See 05_model_v2_fixed_descriptors.ipynb and
06_model_v3_fingerprints.ipynb for the full history.
"""
import os
import pickle
import warnings

import pandas as pd
import pytest
from rdkit import Chem

import chem_utils as cu

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "trained_models")


@pytest.fixture(scope="module")
def scaler():
    with open(os.path.join(MODELS_DIR, "scaler_v2.pkl"), "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def feature_names_v2():
    with open(os.path.join(MODELS_DIR, "feature_names_v2.pkl"), "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def feature_names_v3():
    with open(os.path.join(MODELS_DIR, "feature_names_v3.pkl"), "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def model_v3():
    with open(os.path.join(MODELS_DIR, "best_model_v3.pkl"), "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------
# validate_smiles
# ---------------------------------------------------------------------

def test_validate_smiles_accepts_valid():
    assert cu.validate_smiles("c1ccccc1") is True


def test_validate_smiles_rejects_garbage():
    assert cu.validate_smiles("not a smiles!!") is False


def test_validate_smiles_rejects_empty():
    assert cu.validate_smiles("") is False
    assert cu.validate_smiles(None) is False


# ---------------------------------------------------------------------
# get_molecular_descriptors
# ---------------------------------------------------------------------

def test_descriptors_returns_all_expected_keys_for_valid_molecule():
    desc = cu.get_molecular_descriptors("c1ccccc1")  # benzene
    assert desc is not None
    assert set(desc.keys()) == set(cu.DESCRIPTOR_KEYS)


def test_descriptors_none_for_invalid_smiles():
    assert cu.get_molecular_descriptors("not a smiles!!") is None
    assert cu.get_molecular_descriptors("") is None
    assert cu.get_molecular_descriptors(None) is None


@pytest.mark.parametrize("smiles", [
    "ClC1=CC=C(C=C1)Br",
    "Nc1ccccc1",
    "Brc1ccc(F)cc1",
    "CNC",
    "Clc1ccc(Nc2ccccc2)cc1",
])
def test_descriptors_no_none_values_regression(smiles):
    """Core regression guard: every value must be a real number, never
    None, for a valid molecule — this is exactly what silently broke in
    v1."""
    desc = cu.get_molecular_descriptors(smiles)
    assert desc is not None
    for key, value in desc.items():
        assert value is not None, f"{key} was None for {smiles!r}"
        assert isinstance(value, (int, float)), f"{key} is not numeric for {smiles!r}: {value!r}"


def test_no_warnings_for_valid_molecules():
    """Spot-check real dataset-style SMILES to make sure descriptor
    computation doesn't silently warn/fail on any of them (the v1 bug
    silently failed on every single one)."""
    smiles_list = [
        "ClC1=CC=C(C=C1)Br",
        "COc1ccc(Br)cc1",
        "Ic1ccccc1",
        "Nc1ccccc1",
        "CNc1ccccc1",
    ]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for s in smiles_list:
            desc = cu.get_molecular_descriptors(s)
            assert desc is not None
    assert len(caught) == 0, f"unexpected warnings: {[str(w.message) for w in caught]}"


# ---------------------------------------------------------------------
# get_fingerprint_bits
# ---------------------------------------------------------------------

def test_fingerprint_bits_length_and_prefix():
    fp = cu.get_fingerprint_bits("c1ccccc1", "aryl")
    assert len(fp) == cu.FP_BITS
    assert all(k.startswith("aryl_fp_") for k in fp)
    assert set(fp.values()) <= {0, 1}


def test_fingerprint_bits_nonzero_for_real_molecule():
    fp = cu.get_fingerprint_bits("c1ccccc1", "aryl")
    assert sum(fp.values()) > 0


def test_fingerprint_bits_all_zero_for_invalid_smiles():
    fp = cu.get_fingerprint_bits("not a smiles!!", "aryl")
    assert sum(fp.values()) == 0


def test_fingerprint_differs_between_different_molecules():
    fp1 = cu.get_fingerprint_bits("c1ccccc1", "x")        # benzene
    fp2 = cu.get_fingerprint_bits("c1ccc2ccccc2c1", "x")  # naphthalene
    assert fp1 != fp2


# ---------------------------------------------------------------------
# generate_product_smiles
# ---------------------------------------------------------------------

def test_product_generation_simple_case():
    product = cu.generate_product_smiles("Brc1ccccc1", "Nc1ccccc1")
    assert product is not None
    expected = Chem.CanonSmiles("c1ccc(Nc2ccccc2)cc1")
    assert Chem.CanonSmiles(product) == expected


def test_product_generation_prefers_reactive_halide_over_chlorine():
    """4-bromochlorobenzene + aniline should react at C-Br (more
    reactive under Pd catalysis), leaving the C-Cl bond intact."""
    product = cu.generate_product_smiles("ClC1=CC=C(C=C1)Br", "Nc1ccccc1")
    assert product is not None
    mol = Chem.MolFromSmiles(product)
    atoms = {a.GetSymbol() for a in mol.GetAtoms()}
    assert "Cl" in atoms
    assert "Br" not in atoms


def test_product_generation_invalid_smiles_returns_none():
    assert cu.generate_product_smiles("garbage", "Nc1ccccc1") is None
    assert cu.generate_product_smiles("Brc1ccccc1", "garbage") is None
    assert cu.generate_product_smiles("", "") is None


def test_product_generation_no_reactive_groups_returns_none():
    # benzene has no halide; ethanol has no N-H
    assert cu.generate_product_smiles("c1ccccc1", "CCO") is None


# ---------------------------------------------------------------------
# build_scaled_descriptor_row
# ---------------------------------------------------------------------

def test_scaled_row_shape_and_columns(scaler, feature_names_v2):
    aryl_desc = cu.get_molecular_descriptors("ClC1=CC=C(C=C1)Br")
    product_desc = cu.get_molecular_descriptors("Clc1ccc(Nc2ccccc2)cc1")
    row = cu.build_scaled_descriptor_row(
        aryl_desc, product_desc, "MTBD", "XPhos", "No_Additive", scaler, feature_names_v2
    )
    assert row.shape == (1, len(feature_names_v2))
    assert list(row.columns) == list(feature_names_v2)
    assert not row.isnull().any().any()


def test_scaled_row_changes_with_base_ligand_additive(scaler, feature_names_v2):
    aryl_desc = cu.get_molecular_descriptors("ClC1=CC=C(C=C1)Br")
    product_desc = cu.get_molecular_descriptors("Clc1ccc(Nc2ccccc2)cc1")
    row_a = cu.build_scaled_descriptor_row(
        aryl_desc, product_desc, "MTBD", "XPhos", "No_Additive", scaler, feature_names_v2
    )
    row_b = cu.build_scaled_descriptor_row(
        aryl_desc, product_desc, "P2Et", "AdBrettPhos", "3-methylisoxazole", scaler, feature_names_v2
    )
    assert not row_a.equals(row_b)


# ---------------------------------------------------------------------
# End-to-end regression: the actual historical bug
# ---------------------------------------------------------------------

def test_end_to_end_different_substrates_give_different_predictions(
    scaler, feature_names_v2, feature_names_v3, model_v3
):
    """Core regression test for the bug this whole project is named for:
    the deployed app used to return IDENTICAL results for every
    substrate because it never actually used the input SMILES. Two
    different aryl halides, same reaction conditions, must now give
    different predicted yields."""

    def predict_for(aryl_smiles, amine_smiles, base, ligand, additive):
        aryl_desc = cu.get_molecular_descriptors(aryl_smiles)
        product_smiles = cu.generate_product_smiles(aryl_smiles, amine_smiles)
        product_desc = cu.get_molecular_descriptors(product_smiles)
        fp = {}
        fp.update(cu.get_fingerprint_bits(aryl_smiles, "aryl"))
        fp.update(cu.get_fingerprint_bits(product_smiles, "product"))
        fp_row = pd.DataFrame([fp])
        scaled = cu.build_scaled_descriptor_row(
            aryl_desc, product_desc, base, ligand, additive, scaler, feature_names_v2
        )
        X = pd.concat([scaled.reset_index(drop=True), fp_row.reset_index(drop=True)], axis=1)
        X = X.reindex(columns=feature_names_v3, fill_value=0)
        return float(model_v3.predict(X)[0])

    y1 = predict_for("ClC1=CC=C(C=C1)Br", "Nc1ccccc1", "MTBD", "XPhos", "No_Additive")
    y2 = predict_for("Brc1ccc(F)cc1", "CNC", "MTBD", "XPhos", "No_Additive")
    assert abs(y1 - y2) > 0.01, "different substrates produced identical predictions"
