"""
Chemistry / feature-engineering utilities for the Buchwald-Hartwig
Optimizer, extracted from app.py into their own module so they can be
unit-tested without a Streamlit runtime.

This module has no Streamlit dependency on purpose: app.py imports these
functions and handles user-facing error display itself. A genuine
descriptor-computation failure here is surfaced via warnings.warn rather
than swallowed — this is the exact class of bug (a broad
`except Exception: return None` hiding an AttributeError from a
misspelled/nonexistent RDKit API) that caused the v1 model's Test R² to
be 0.30 instead of the ~0.72-0.93 it should have been. See
05_model_v2_fixed_descriptors.ipynb and 06_model_v3_fingerprints.ipynb
for the full history.
"""
import warnings

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors, rdFingerprintGenerator

BASES = ['P2Et', 'BTMG', 'MTBD']
LIGANDS = ['XPhos', 't-BuXPhos', 't-BuBrettPhos', 'AdBrettPhos']
ADDITIVES = [
    '3,5-dimethylisoxazole', '3-methyl-5-phenylisoxazole',
    '3-methylisoxazole', '3-phenylisoxazole', '4-phenylisoxazole',
    '5-(2,6-difluorophenyl)isoxazole', '5-Phenyl-1,2,4-oxadiazole',
    '5-methyl-3-(1H-pyrrol-1-yl)isoxazole', '5-methylisoxazole',
    '5-phenylisoxazole', 'N,N-dibenzylisoxazol-3-amine',
    'N,N-dibenzylisoxazol-5-amine', 'No_Additive',
    'benzo[c]isoxazole', 'benzo[d]isoxazole',
    'ethyl-3-methoxyisoxazole-5-carboxylate',
    'ethyl-3-methylisoxazole-5-carboxylate',
    'ethyl-5-methylisoxazole-3-carboxylate',
    'ethyl-5-methylisoxazole-4-carboxylate',
    'ethyl-isoxazole-3-carboxylate',
    'ethyl-isoxazole-4-carboxylate',
    'methyl-5-(furan-2-yl)isoxazole-3-carboxylate',
    'methyl-5-(thiophen-2-yl)isoxazole-3-carboxylate',
    'methyl-isoxazole-5-carboxylate'
]

# Number of descriptor keys get_molecular_descriptors() must always return
# for a valid molecule. A regression that silently drops a key (or the
# whole dict) is exactly the class of bug this project is named for.
DESCRIPTOR_KEYS = [
    'mw', 'logp', 'hbd', 'hba', 'rotatable_bonds', 'aromatic_rings',
    'num_atoms', 'num_heavy_atoms', 'tpsa', 'molar_refractivity', 'fsp3',
    'num_aromatic_atoms', 'num_heteroatoms', 'num_heterocycles',
    'num_saturated_rings', 'num_aliphatic_rings', 'num_valence_electrons',
    'formal_charge', 'num_explicit_hs', 'num_radical_electrons',
]

FP_BITS = 128
FP_RADIUS = 2
_MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)

# Reaction templates for generating the coupling product from the two
# user-supplied substrates. Halide reactivity in Pd-catalyzed amination
# is I/Br >> Cl, so we try the more reactive halide first and only fall
# back to Cl if the molecule has no Br/I.
_RXN_PRIORITY = AllChem.ReactionFromSmarts(
    '[c:1][Br,I:2].[NX3;H1,H2;!$(N=*);!$(N-C(=O)):3]>>[c:1][N:3]'
)
_RXN_FALLBACK_CL = AllChem.ReactionFromSmarts(
    '[c:1][Cl:2].[NX3;H1,H2;!$(N=*);!$(N-C(=O)):3]>>[c:1][N:3]'
)


def validate_smiles(smiles):
    """Return True iff `smiles` parses to a non-empty molecule."""
    if not smiles or smiles.strip() == '':
        return False
    mol = Chem.MolFromSmiles(smiles.strip())
    return mol is not None and mol.GetNumAtoms() > 0


def get_molecular_descriptors(smiles):
    """Extract molecular descriptors from a SMILES string.

    Returns a dict with exactly the keys in DESCRIPTOR_KEYS, or None if
    `smiles` doesn't parse. A genuine computation failure (e.g. a typo'd
    or nonexistent RDKit API — the v1 bug this project is named for) is
    raised as a warnings.warn call, not silently swallowed, so it shows
    up in logs and can be asserted on in tests.
    """
    if smiles is None or smiles == '':
        return None
    try:
        mol = Chem.MolFromSmiles(str(smiles).strip())
    except Exception:
        return None
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    try:
        descriptors = {
            'mw': Descriptors.MolWt(mol),
            'logp': Crippen.MolLogP(mol),
            'hbd': Descriptors.NumHDonors(mol),
            'hba': Descriptors.NumHAcceptors(mol),
            'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
            'aromatic_rings': Descriptors.NumAromaticRings(mol),
            'num_atoms': mol.GetNumAtoms(),
            'num_heavy_atoms': Descriptors.HeavyAtomCount(mol),
            'tpsa': Descriptors.TPSA(mol),
            'molar_refractivity': Crippen.MolMR(mol),
            'fsp3': Descriptors.FractionCSP3(mol),
            'num_aromatic_atoms': sum(1 for a in mol.GetAtoms() if a.GetIsAromatic()),
            'num_heteroatoms': Descriptors.NumHeteroatoms(mol),
            'num_heterocycles': (Descriptors.NumAromaticHeterocycles(mol)
                                  + Descriptors.NumSaturatedHeterocycles(mol)
                                  + Descriptors.NumAliphaticHeterocycles(mol)),
            'num_saturated_rings': Descriptors.NumSaturatedRings(mol),
            'num_aliphatic_rings': Descriptors.NumAliphaticRings(mol),
            'num_valence_electrons': Descriptors.NumValenceElectrons(mol),
            'formal_charge': Chem.GetFormalCharge(mol),
            'num_explicit_hs': sum(a.GetNumExplicitHs() for a in mol.GetAtoms()),
            'num_radical_electrons': Descriptors.NumRadicalElectrons(mol),
        }
    except Exception as e:
        warnings.warn(f"Descriptor computation failed for SMILES={smiles!r}: {e}")
        return None

    missing = set(DESCRIPTOR_KEYS) - set(descriptors)
    if missing:
        # Should be unreachable, but if it ever happens, fail loudly
        # instead of silently returning a partial dict.
        warnings.warn(f"Descriptor dict missing expected keys {missing} for SMILES={smiles!r}")
        return None

    return descriptors


def get_fingerprint_bits(smiles, prefix):
    """Morgan fingerprint (radius=2, 128 bits) as a named feature dict,
    e.g. {'aryl_fp_0': 0, 'aryl_fp_1': 1, ...}. Returns all-zero bits
    (not None) for an invalid SMILES, since a fingerprint of "no
    structure" is a valid, harmless input to the model."""
    arr = np.zeros(FP_BITS, dtype=int)
    mol = Chem.MolFromSmiles(str(smiles).strip()) if smiles else None
    if mol is not None:
        fp = _MORGAN_GEN.GetFingerprint(mol)
        for i in fp.GetOnBits():
            arr[i] = 1
    return {f'{prefix}_fp_{i}': int(v) for i, v in enumerate(arr)}


def generate_product_smiles(aryl_smiles, amine_smiles):
    """Generate the C-N coupling product from the aryl halide + amine.

    Tries the more reactive halide (Br/I) first; falls back to Cl only
    if the molecule has no Br/I. Returns None if no reaction template
    matches (invalid SMILES, or no aryl-halide/N-H found).

    This is a simplification (single reactive halide, single reactive
    N-H) — a fair assumption for the substrates this dataset/demo uses,
    but not guaranteed correct for every possible input structure.
    """
    if not aryl_smiles or not amine_smiles:
        return None
    aryl_mol = Chem.MolFromSmiles(aryl_smiles.strip())
    amine_mol = Chem.MolFromSmiles(amine_smiles.strip())
    if aryl_mol is None or amine_mol is None:
        return None

    products = _RXN_PRIORITY.RunReactants((aryl_mol, amine_mol))
    if not products:
        products = _RXN_FALLBACK_CL.RunReactants((aryl_mol, amine_mol))
    if not products:
        return None

    for p in products:
        mol = p[0]
        try:
            Chem.SanitizeMol(mol)
            return Chem.MolToSmiles(mol)
        except Exception:
            continue
    return None


def build_scaled_descriptor_row(aryl_desc, product_desc, base, ligand, additive, scaler, feature_names_v2):
    """Build the 71-column scaled descriptor+categorical block for one
    base/ligand/additive combination.

    Important subtlety: the StandardScaler (scaler_v2.pkl) was fit on
    the RAW (un-sanitized) column names — e.g. 'additive_benzo[c]isoxazole'
    with brackets — while the trained model's feature names are the
    sanitized versions (brackets replaced with '_', required by
    XGBoost). The column ORDER is identical between the two; only those
    two additive names differ in spelling. So we build the raw-named
    row, scale it using the scaler's own column order, and only then
    relabel the columns to the sanitized names the model expects.
    """
    raw = {}
    for k, v in aryl_desc.items():
        raw[f'aryl_{k}'] = v
    for k, v in product_desc.items():
        raw[f'product_{k}'] = v
    for b in BASES:
        raw[f'base_{b}'] = 1 if b == base else 0
    for l in LIGANDS:
        raw[f'ligand_{l}'] = 1 if l == ligand else 0
    for add in ADDITIVES:
        raw[f'additive_{add}'] = 1 if add == additive else 0

    raw_names = list(scaler.feature_names_in_)
    row = pd.DataFrame([raw]).reindex(columns=raw_names, fill_value=0)
    scaled = scaler.transform(row)
    return pd.DataFrame(scaled, columns=feature_names_v2)
