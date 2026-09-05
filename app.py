import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from chem_utils import (
    BASES,
    LIGANDS,
    ADDITIVES,
    validate_smiles,
    get_molecular_descriptors,
    get_fingerprint_bits,
    generate_product_smiles,
    build_scaled_descriptor_row,
)

st.set_page_config(
    page_title="Buchwald-Hartwig Optimizer",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# Visual styling — header banner, card-style metrics, refined
# spacing/typography. Paired with .streamlit/config.toml, which
# sets the overall color theme.
# ============================================================
st.markdown("""
<style>
    .main {
        padding-top: 1.5rem;
    }

    /* Header banner */
    .bh-header {
        background: linear-gradient(120deg, #4f46e5 0%, #7c3aed 100%);
        border-radius: 14px;
        padding: 1.75rem 2rem;
        margin-bottom: 1.5rem;
        color: #ffffff;
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.25);
    }
    .bh-header h1 {
        margin: 0 0 0.35rem 0;
        font-size: 1.9rem;
        font-weight: 700;
        color: #ffffff;
    }
    .bh-header p {
        margin: 0;
        font-size: 1rem;
        opacity: 0.92;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 16px 18px;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(30, 41, 59, 0.08);
        border-left: 4px solid #4f46e5;
    }
    [data-testid="stMetricLabel"] {
        color: #475569;
        font-weight: 600;
    }
    [data-testid="stMetricValue"] {
        color: #4f46e5;
        font-size: 26px;
    }

    /* Dataframes / tables */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 2px 12px rgba(30, 41, 59, 0.08);
    }

    /* Section headers */
    h2, h3 {
        color: #1e1b3a;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #f8f7fc;
        border-right: 1px solid #ece9f8;
    }
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
        color: #4f46e5;
    }

    /* Primary buttons */
    .stButton > button, .stDownloadButton > button {
        border-radius: 8px;
        font-weight: 600;
        border: none;
    }
    section[data-testid="stSidebar"] .stButton > button {
        background-color: #4f46e5;
        color: #ffffff;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background-color: #4338ca;
        color: #ffffff;
    }

    /* Caption / product SMILES chip */
    .bh-product-chip {
        display: inline-block;
        background-color: #f1f0fd;
        color: #4f46e5;
        padding: 4px 12px;
        border-radius: 999px;
        font-family: monospace;
        font-size: 0.85rem;
        margin: 0.25rem 0 1rem 0;
    }

    footer {visibility: hidden;}
</style>

<div class="bh-header">
    <h1>🧪 Buchwald-Hartwig C-N Coupling Optimizer</h1>
    <p>Predict optimal base / ligand / additive combinations for Pd-catalyzed C–N cross-coupling — from your own substrate SMILES.</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# Model / feature-engineering artifacts — v3: descriptors + Morgan
# fingerprints (see notebooks 05_model_v2_fixed_descriptors.ipynb
# and 06_model_v3_fingerprints.ipynb for the full history:
# v1 R²=0.30 -> v2 R²=0.72 -> v3 R²=0.93). The actual chemistry/
# feature-engineering functions live in chem_utils.py, unit-tested in
# tests/test_chem_utils.py.
# ============================================================


@st.cache_resource
def load_artifacts():
    models_dir = os.path.join("data", "trained_models")
    model_path = os.path.join(models_dir, "best_model_v3.pkl")
    scaler_path = os.path.join(models_dir, "scaler_v2.pkl")
    v2_features_path = os.path.join(models_dir, "feature_names_v2.pkl")
    v3_features_path = os.path.join(models_dir, "feature_names_v3.pkl")

    for p in (model_path, scaler_path, v2_features_path, v3_features_path):
        if not os.path.exists(p):
            st.error(f"Required artifact not found: {p}")
            st.stop()

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(v2_features_path, 'rb') as f:
        feature_names_v2 = pickle.load(f)
    with open(v3_features_path, 'rb') as f:
        feature_names_v3 = pickle.load(f)

    return model, scaler, feature_names_v2, feature_names_v3


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

        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#4f46e5'),
            spaceAfter=30
        )
        story.append(Paragraph("Buchwald-Hartwig C-N Coupling Optimizer", title_style))
        story.append(Spacer(1, 0.3*inch))

        story.append(Paragraph("<b>Input Substrates:</b>", styles['Heading2']))
        story.append(Paragraph(f"Aryl Halide SMILES: <font color='blue'>{aryl_smiles}</font>", styles['Normal']))
        story.append(Paragraph(f"Nucleophile SMILES: <font color='blue'>{amine_smiles}</font>", styles['Normal']))
        story.append(Spacer(1, 0.2*inch))

        story.append(Paragraph("<b>Molecular Structures:</b>", styles['Heading2']))

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
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4f46e5')),
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

try:
    model, scaler, feature_names_v2, feature_names_v3 = load_artifacts()
    st.sidebar.success("✓ Model loaded (v3 — descriptors + fingerprints)")
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

st.sidebar.markdown("---")
st.sidebar.caption("Model: XGBoost · Test R² = 0.93 · 5-fold CV = 0.933 ± 0.005")

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

            st.markdown(f'<span class="bh-product-chip">Predicted product: {product_smiles}</span>', unsafe_allow_html=True)

            # Fingerprints only depend on the substrates, not on the
            # base/ligand/additive grid — compute them once and reuse.
            fp_features = {}
            fp_features.update(get_fingerprint_bits(aryl_smiles, 'aryl'))
            fp_features.update(get_fingerprint_bits(product_smiles, 'product'))
            fp_row = pd.DataFrame([fp_features])

            results = []
            for base in BASES:
                for ligand in LIGANDS:
                    for additive in ADDITIVES:
                        scaled_desc_row = build_scaled_descriptor_row(
                            aryl_desc, product_desc, base, ligand, additive, scaler, feature_names_v2
                        )
                        X_full = pd.concat(
                            [scaled_desc_row.reset_index(drop=True), fp_row.reset_index(drop=True)],
                            axis=1
                        ).reindex(columns=feature_names_v3, fill_value=0)

                        try:
                            yield_pred = model.predict(X_full)[0]
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

                st.markdown("**Base Performance:**")
                base_stats = results_df.groupby('Base')['Predicted Yield'].agg(['mean', 'max', 'min'])
                st.dataframe(base_stats.round(1), use_container_width=True)

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

with st.expander("ℹ️ About this model & its development history"):
    st.markdown("""
An earlier version of this model reported **Test R² = 0.30**. The cause: two RDKit descriptor
calls in the feature-engineering step were silently failing (`Descriptors.FractionCsp3` — wrong
capitalization — and `Descriptors.NumAromaticAtoms`, which doesn't exist), so every substrate
descriptor was dropped and the model was only ever trained on the identity of the base, ligand
and additive — never on the actual chemical structure.

**v2 — bug fixed:** correcting the descriptor computation and wiring it into this app
brought Test R² to **0.72** (Gradient Boosting).

**v3 — Morgan fingerprints added:** adding 128-bit Morgan fingerprints (radius 2) for both the
aryl halide and the predicted coupling product, on top of the physicochemical descriptors,
brought Test R² to **0.93** (XGBoost) — confirmed with 5-fold cross-validation
(mean R² = 0.933, std = 0.005), not just a single lucky train/test split.

The coupling product itself is generated automatically from your two SMILES using an RDKit
reaction template (most-reactive-halide priority: I/Br over Cl). This works well for typical
substrates like the examples above, but may not resolve a product for more complex or unusual
structures.

*This tool is for research and educational purposes. Always validate predictions experimentally.*
    """)

st.markdown("<p style='text-align: center; color: #94a3b8;'><small>🧬 Buchwald-Hartwig Optimizer v3.0 — Research Tool</small></p>", unsafe_allow_html=True)
