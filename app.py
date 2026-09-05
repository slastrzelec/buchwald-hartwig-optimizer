import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, AllChem
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Buchwald-Hartwig Optimizer",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    [data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #1f77b4;
    }
    [data-testid="stMetricLabel"] {
        color: #333333;
    }
    [data-testid="stMetricValue"] {
        color: #1f77b4;
        font-size: 28px;
    }
    .main {
        padding-top: 2rem;
    }
    [data-testid="stDataFrame"] {
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Model / feature-engineering artifacts (v2 — see notebook
# 05_model_v2_fixed_descriptors.ipynb for the full bug-fix story)
# ============================================================

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

# Reaction templates for generating the coupling product from the two
# user-supplied substrates. Halide reactivity in Pd-catalyzed amination
# is I/Br >> Cl, so we try the more reactive halide first and only fall
# back to Cl if there is no Br/I in the molecule.
_RXN_PRIORITY = AllChem.ReactionFromSmarts(
    '[c:1][Br,I:2].[NX3;H1,H2;!$(N=*);!$(N-C(=O)):3]>>[c:1][N:3]'
)
_RXN_FALLBACK_CL = AllChem.ReactionFromSmarts(
    '[c:1][Cl:2].[NX3;H1,H2;!$(N=*);!$(N-C(=O)):3]>>[c:1][N:3]'
)


@st.cache_resource
def load_artifacts():
    models_dir = os.path.join("data", "trained_models")
    model_path = os.path.join(models_dir, "best_model_v2.pkl")
    scaler_path = os.path.join(models_dir, "scaler_v2.pkl")
    features_path = os.path.join(models_dir, "feature_names_v2.pkl")

    for p in (model_path, scaler_path, features_path):
        if not os.path.exists(p):
            st.error(f"Required artifact not found: {p}")
            st.stop()

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(features_path, 'rb') as f:
        feature_names = pickle.load(f)

    return model, scaler, feature_names


def validate_smiles(smiles):
    """Validate SMILES string"""
    if not smiles or smiles.strip() == '':
        return False
    mol = Chem.MolFromSmiles(smiles.strip())
    return mol is not None and mol.GetNumAtoms() > 0


def get_molecular_descriptors(smiles):
    """Extract molecular descriptors from a SMILES string.

    This mirrors the fixed function from
    05_model_v2_fixed_descriptors.ipynb — v1 silently produced None for
    every molecule here because of two RDKit API mistakes
    (FractionCsp3 -> FractionCSP3, NumAromaticAtoms doesn't exist),
    which meant the deployed model never used any substrate structure
    information at all.
    """
    try:
        if smiles is None or smiles == '':
            return None
        mol = Chem.MolFromSmiles(str(smiles).strip())
        if mol is None or mol.GetNumAtoms() == 0:
            return None
        return {
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
        st.warning(f"Descriptor computation failed for SMILES={smiles!r}: {e}")
        return None


def generate_product_smiles(aryl_smiles, amine_smiles):
    """Generate the C-N coupling product from the aryl halide + amine.

    Tries the more reactive halide (Br/I) first; falls back to Cl only
    if the molecule has no Br/I. This is a simplification (it assumes a
    single reactive halide and a single reactive N-H), which is a fair
    assumption for the simple substrates this dataset/demo uses, but
    won't be correct for every possible input structure.
    """
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


def build_feature_row(aryl_desc, product_desc, base, ligand, additive, scaler, feature_names):
    """Build one model-ready (scaled) feature row for a given
    base/ligand/additive combination and a fixed pair of substrate
    descriptors.

    Important subtlety: the StandardScaler (scaler_v2.pkl) was fit on
    the RAW (un-sanitized) column names — e.g. 'additive_benzo[c]isoxazole'
    with brackets — while the trained model's feature names
    (feature_names_v2.pkl) are the sanitized versions (brackets replaced
    with '_', required by XGBoost during the model-selection step). The
    column ORDER is identical between the two; only those two additive
    names differ in spelling. So we build the raw-named row, scale it
    using the scaler's own column order, and only then relabel the
    columns to the sanitized names the model expects.
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
    scaled_row = pd.DataFrame(scaled, columns=feature_names)
    return scaled_row


def generate_pdf_report(results_df, aryl_smiles, amine_smiles, top_n=10):
    """Generate PDF report with molecular structures"""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        import io
        import requests
        from PIL import Image as PILImage

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        story = []
        styles = getSampleStyleSheet()

        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=30
        )
        story.append(Paragraph("Buchwald-Hartwig C-N Coupling Optimizer", title_style))
        story.append(Spacer(1, 0.3*inch))

        # Input info
        story.append(Paragraph("<b>Input Substrates:</b>", styles['Heading2']))
        story.append(Paragraph(f"Aryl Halide SMILES: <font color='blue'>{aryl_smiles}</font>", styles['Normal']))
        story.append(Paragraph(f"Nucleophile SMILES: <font color='blue'>{amine_smiles}</font>", styles['Normal']))
        story.append(Spacer(1, 0.2*inch))

        # Add molecular structures
        story.append(Paragraph("<b>Molecular Structures:</b>", styles['Heading2']))

        struct_table_data = []

        # Get images from PubChem
        try:
            aryl_img = requests.get(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{aryl_smiles}/PNG", timeout=5)
            if aryl_img.status_code == 200:
                aryl_pil = PILImage.open(io.BytesIO(aryl_img.content))
                aryl_buf = io.BytesIO()
                aryl_pil.save(aryl_buf, format="PNG")
                aryl_buf.seek(0)
                aryl_img_obj = Image(aryl_buf, width=1.5*inch, height=1.5*inch)
            else:
                aryl_img_obj = Paragraph("Aryl Halide Structure", styles['Normal'])
        except:
            aryl_img_obj = Paragraph("Aryl Halide Structure", styles['Normal'])

        try:
            amine_img = requests.get(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{amine_smiles}/PNG", timeout=5)
            if amine_img.status_code == 200:
                amine_pil = PILImage.open(io.BytesIO(amine_img.content))
                amine_buf = io.BytesIO()
                amine_pil.save(amine_buf, format="PNG")
                amine_buf.seek(0)
                amine_img_obj = Image(amine_buf, width=1.5*inch, height=1.5*inch)
            else:
                amine_img_obj = Paragraph("Nucleophile Structure", styles['Normal'])
        except:
            amine_img_obj = Paragraph("Nucleophile Structure", styles['Normal'])

        struct_table_data = [
            [Paragraph("<b>Aryl Halide</b>", styles['Normal']), Paragraph("<b>Nucleophile</b>", styles['Normal'])],
            [aryl_img_obj, amine_img_obj]
        ]

        struct_table = Table(struct_table_data, colWidths=[2.5*inch, 2.5*inch])
        struct_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e6f2ff')),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey)
        ]))
        story.append(struct_table)
        story.append(Spacer(1, 0.3*inch))

        # Results table
        story.append(Paragraph("<b>Top 10 Recommended Conditions:</b>", styles['Heading2']))
        story.append(Spacer(1, 0.1*inch))

        top_results = results_df.head(top_n).reset_index(drop=True)
        top_results['Rank'] = range(1, len(top_results) + 1)
        top_results['Yield %'] = top_results['Predicted Yield'].apply(lambda x: f"{x:.1f}%")

        table_data = [['Rank', 'Base', 'Ligand', 'Additive', 'Yield %']]
        for _, row in top_results.iterrows():
            table_data.append([
                str(int(row['Rank'])),
                row['Base'],
                row['Ligand'],
                row['Additive'][:30],
                row['Yield %']
            ])

        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(table)
        story.append(Spacer(1, 0.3*inch))

        # Footer
        story.append(Paragraph("<i>⚠️ This is a preliminary model for research purposes only. Always validate results experimentally.</i>", styles['Normal']))

        doc.build(story)
        buffer.seek(0)
        return buffer
    except ImportError:
        st.warning("ReportLab not available. Install with: pip install reportlab")
        return None
    except Exception as e:
        st.warning(f"PDF generation error: {str(e)[:100]}")
        return None

st.title("🧪 Buchwald-Hartwig C-N Coupling Optimizer")
st.markdown("**Optimize reaction conditions for C-N cross-coupling.**")
st.markdown("---")

try:
    model, scaler, feature_names = load_artifacts()
    st.sidebar.success("✓ Model loaded (v2)")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Initialize session state for history
if 'history' not in st.session_state:
    st.session_state.history = []

st.sidebar.header("📝 Input Substrates")

example_aryl = "ClC1=CC=C(C=C1)Br"
example_amine = "Nc1ccccc1"

aryl_smiles = st.sidebar.text_input("Aryl Halide SMILES", value=example_aryl)
amine_smiles = st.sidebar.text_input("Nucleophile (Amine) SMILES", value=example_amine)
predict_button = st.sidebar.button("🔮 Predict", use_container_width=True)

# Display structures using PubChem
col1, col2 = st.columns(2)

with col1:
    if aryl_smiles.strip() and validate_smiles(aryl_smiles):
        st.subheader("Aryl Halide")
        st.image(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{aryl_smiles}/PNG", width=300)

with col2:
    if amine_smiles.strip() and validate_smiles(amine_smiles):
        st.subheader("Nucleophile")
        st.image(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{amine_smiles}/PNG", width=300)

if predict_button:
    if not validate_smiles(aryl_smiles) or not validate_smiles(amine_smiles):
        st.error("❌ Invalid SMILES")
    else:
        with st.spinner("Predicting..."):
            # 1. Compute real substrate descriptors from the SMILES the
            #    user actually typed in (this is the part v1 skipped).
            aryl_desc = get_molecular_descriptors(aryl_smiles)
            if aryl_desc is None:
                st.error("❌ Could not compute descriptors for the aryl halide SMILES.")
                st.stop()

            product_smiles = generate_product_smiles(aryl_smiles, amine_smiles)
            if product_smiles is None:
                st.error(
                    "❌ Could not generate a C-N coupling product from these two SMILES "
                    "(no matching aryl halide + N-H found). Try a simpler aryl halide "
                    "(one Cl/Br/I on an aromatic ring) and a primary/secondary amine."
                )
                st.stop()

            product_desc = get_molecular_descriptors(product_smiles)
            if product_desc is None:
                st.error("❌ Could not compute descriptors for the predicted product.")
                st.stop()

            st.caption(f"Predicted coupling product: `{product_smiles}`")

            results = []
            for base in BASES:
                for ligand in LIGANDS:
                    for additive in ADDITIVES:
                        X_scaled = build_feature_row(
                            aryl_desc, product_desc, base, ligand, additive, scaler, feature_names
                        )

                        try:
                            yield_pred = model.predict(X_scaled)[0]
                            yield_pred = np.clip(yield_pred, 0, 100)
                        except Exception as e:
                            st.error(f"Error: {e}")
                            st.stop()

                        results.append({
                            'Base': base,
                            'Ligand': ligand,
                            'Additive': additive,
                            'Predicted Yield': yield_pred
                        })

            results_df = pd.DataFrame(results).sort_values('Predicted Yield', ascending=False)

            # Add to history
            st.session_state.history.append({
                'Aryl SMILES': aryl_smiles,
                'Amine SMILES': amine_smiles,
                'Top Result': results_df.iloc[0]['Base'],
                'Yield': f"{results_df.iloc[0]['Predicted Yield']:.1f}%",
                'Timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
            })

            st.markdown("---")
            st.subheader("🎯 Top 10 Recommendations")
            st.markdown("**Predicted reaction yields sorted by performance:**")

            top_10 = results_df.head(10).reset_index(drop=True)
            top_10['Rank'] = range(1, 11)
            top_10['Yield %'] = top_10['Predicted Yield'].apply(lambda x: f"{x:.1f}")

            st.dataframe(
                top_10[['Rank', 'Base', 'Ligand', 'Additive', 'Yield %']],
                use_container_width=True,
                hide_index=True
            )

            # Detailed information
            with st.expander("📊 Detailed Statistics"):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Best Yield", f"{results_df['Predicted Yield'].max():.1f}%")
                with col2:
                    st.metric("Average Yield", f"{results_df['Predicted Yield'].mean():.1f}%")
                with col3:
                    st.metric("Total Combinations", len(results_df))
                with col4:
                    st.metric("Std Dev", f"{results_df['Predicted Yield'].std():.1f}%")

                # Base performance
                st.markdown("**Base Performance:**")
                base_stats = results_df.groupby('Base')['Predicted Yield'].agg(['mean', 'max', 'min'])
                st.dataframe(base_stats.round(1), use_container_width=True)

                # Ligand performance
                st.markdown("**Ligand Performance:**")
                ligand_stats = results_df.groupby('Ligand')['Predicted Yield'].agg(['mean', 'max', 'min'])
                st.dataframe(ligand_stats.round(1), use_container_width=True)

            # Export options
            st.markdown("---")
            st.subheader("💾 Export Results")

            col1, col2, col3 = st.columns(3)

            with col1:
                csv = top_10.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"buchwald_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

            with col2:
                pdf_buffer = generate_pdf_report(results_df, aryl_smiles, amine_smiles)
                if pdf_buffer:
                    st.download_button(
                        label="📄 Download PDF",
                        data=pdf_buffer,
                        file_name=f"buchwald_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf"
                    )

# History sidebar
st.sidebar.markdown("---")
st.sidebar.header("📜 History")

if st.session_state.history:
    history_df = pd.DataFrame(st.session_state.history)
    st.sidebar.dataframe(history_df, use_container_width=True, hide_index=True)

    if st.sidebar.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()
else:
    st.sidebar.info("No predictions yet")

st.markdown("---")
st.markdown("<p style='text-align: center'><small>🧬 Buchwald-Hartwig Optimizer v2.0 | Research Tool</small></p>", unsafe_allow_html=True)

# Status notice at bottom
st.info("""
### 🔧 v2: Bug fixed — substrate structure now actually used

An earlier version of this model reported Test R² = 0.30. The cause: two RDKit descriptor
calls in the feature-engineering step were silently failing (`Descriptors.FractionCsp3` — wrong
capitalization — and `Descriptors.NumAromaticAtoms`, which doesn't exist), so **every substrate
descriptor was dropped** and the model was only ever trained on the identity of the base/ligand/additive,
never on the actual chemical structure entered here.

**After fixing the descriptor computation and wiring it into this app:**
- Test R² improved from 0.30 → **0.72** (Gradient Boosting)
- This app now computes real descriptors from the SMILES you enter (aryl halide + the predicted
  coupling product) instead of ignoring them

**Known limitation:** the coupling product is generated with a simplified single-step reaction
template (most-reactive-halide + first available N-H). It works well for typical substrates like
the examples above, but may not find a product for more complex/unusual structures.

*This tool is for research and educational purposes. Always validate predictions experimentally.*
""")
