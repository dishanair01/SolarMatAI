
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="SolarMatAI | Materials Intelligence",
    page_icon="☀️",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

MODEL_PATH = ARTIFACTS_DIR / "bandgap_model.joblib"
ELEMENTS_PATH = ARTIFACTS_DIR / "element_symbols.npy"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
PREDICTIONS_PATH = ARTIFACTS_DIR / "test_predictions.csv"

# ---------------- DESIGN ----------------

st.markdown("""
<style>
/* Main background and typography */
.stApp {
    background: linear-gradient(180deg, #FAF8FF 0%, #F5F3FC 100%);
    color: #292541;
}

html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

/* Main content spacing */
.block-container {
    max-width: 1400px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #EEE8FF 0%, #F7F2FF 100%);
    border-right: 1px solid #E3DAF7;
}

[data-testid="stSidebar"] p {
    color: #51476F;
}

/* Headings */
h1 {
    color: #302653 !important;
    font-weight: 800 !important;
    letter-spacing: -1px;
}

h2 {
    color: #433567 !important;
    font-weight: 750 !important;
}

h3 {
    color: #554477 !important;
    font-weight: 700 !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.95);
    border: 1px solid #E8E0F5;
    border-radius: 18px;
    padding: 20px 18px;
    box-shadow: 0 5px 18px rgba(91, 70, 140, 0.06);
}

[data-testid="stMetricLabel"] {
    color: #77708F !important;
    font-size: 0.88rem !important;
}

[data-testid="stMetricValue"] {
    color: #453477 !important;
    font-weight: 800 !important;
}

/* Buttons */
.stButton > button,
.stDownloadButton > button {
    background: linear-gradient(135deg, #7965C8, #9A83DB);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.65rem 1.2rem;
    font-weight: 650;
    transition: 0.2s ease;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background: linear-gradient(135deg, #6550B4, #836BCB);
    color: white;
    border: none;
    transform: translateY(-1px);
}

/* Input fields */
div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div {
    background: #FFFFFF;
    border-radius: 10px;
    border-color: #DED5F0;
}

[data-testid="stNumberInput"] button {
    color: #6550B4;
}

/* Tables and dataframes */
[data-testid="stDataFrame"] {
    border: 1px solid #E8E0F5;
    border-radius: 14px;
    overflow: hidden;
}

/* Informational messages */
[data-testid="stAlert"] {
    border-radius: 12px;
}

/* Tabs and radio navigation */
button[data-baseweb="tab"] {
    color: #62567E;
    font-weight: 600;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #6550B4;
}

/* Hero section */
.solar-hero {
    padding: 34px;
    border-radius: 24px;
    background: linear-gradient(
        120deg,
        #E8DEFF 0%,
        #F1E8FF 48%,
        #E1EEFF 100%
    );
    border: 1px solid #E2D8F7;
    box-shadow: 0 10px 30px rgba(91, 70, 140, 0.08);
    margin-bottom: 26px;
}

.solar-hero h1 {
    margin: 0 0 12px 0;
    font-size: 2.7rem;
}

.solar-hero p {
    color: #5E5578;
    font-size: 1.05rem;
    line-height: 1.75;
    max-width: 760px;
}

/* Small section label */
.section-label {
    color: #8373B5;
    text-transform: uppercase;
    font-size: 0.76rem;
    letter-spacing: 1.8px;
    font-weight: 800;
}

/* Footer */
.solar-footer {
    margin-top: 38px;
    padding-top: 18px;
    border-top: 1px solid #E6DDF3;
    color: #817A96;
    font-size: 0.82rem;
    text-align: center;
}

@media (max-width: 700px) {
    .solar-hero {
        padding: 23px;
    }

    .solar-hero h1 {
        font-size: 2rem;
    }
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    elements = np.load(ELEMENTS_PATH, allow_pickle=True).tolist()
    return model, elements


st.sidebar.title("☀️ SolarMatAI")
st.sidebar.caption("AI-assisted materials research")
page = st.sidebar.radio(
    "NAVIGATION",
    ["Overview", "Material Research Workspace", "Intelligent Band-Gap Analysis", "Candidate Screening & Comparison", "Band Gap Predictor", "Model Evaluation", "About"]
)

st.sidebar.divider()
st.sidebar.caption("Research prototype · Version 1.0")

if not MODEL_PATH.exists() or not ELEMENTS_PATH.exists():
    st.error(
        "The trained model files could not be found. "
        "Please run `python train_model.py` in your project terminal first."
    )
    st.stop()

try:
    model, elements = load_model()
except Exception as exc:
    st.error(f"Could not load the trained model: {exc}")
    st.stop()

metrics = {}
if METRICS_PATH.exists():
    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        metrics = json.load(file)

# ---------------- OVERVIEW ----------------

if page == "Overview":
    st.markdown("""
<div class="solar-hero">
    <div class="section-label">AI × MATERIALS SCIENCE</div>
    <h1>SolarMatAI</h1>
    <p>
        Discover how artificial intelligence can help explore
        materials for next-generation solar energy research.
    </p>
    <div style="
        display: inline-block;
        margin-top: 10px;
        padding: 8px 14px;
        background: rgba(255,255,255,0.65);
        border: 1px solid #DDD1F5;
        border-radius: 20px;
        color: #62518A;
        font-size: 0.85rem;
        font-weight: 600;
    ">
        MATERIAL INTELLIGENCE · BAND GAP PREDICTION
    </div>
</div>
""", unsafe_allow_html=True)

    st.subheader("Research overview")
    st.write(
        "SolarMatAI is a machine-learning prototype that estimates "
        "material band gaps from elemental composition. It is designed "
        "to help explore candidate materials, not to certify solar-cell performance."
    )

    st.markdown("""
<div class="section-label">PROJECT AT A GLANCE</div>
<h2 style="margin-top: 4px;">Research snapshot</h2>
<p style="color: #77708F; margin-bottom: 20px;">
    A quick overview of the dataset and current model performance.
</p>
""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            label="Dataset records",
            value=f"{int(metrics.get('records_used', 106113)):,}"
        )

    with c2:
        st.metric(
            label="Element features",
            value=f"{len(elements)}"
        )

    with c3:
        st.metric(
            label="Test MAE",
            value=f"{metrics.get('mae_ev', 0.8505):.4f} eV",
            help="Mean absolute error on the held-out test set. Lower is better."
        )

    with c4:
        st.metric(
            label="Test R²",
            value=f"{metrics.get('r2', 0.3984):.4f}",
            help="Coefficient of determination on the held-out test set."
        )

    st.divider()
    st.subheader("How the platform works")

    a, b, c = st.columns(3)

    with a:
        st.markdown("### 01 · Explore")
        st.write("Review the scientific dataset and elemental features.")

    with b:
        st.markdown("### 02 · Predict")
        st.write("Estimate band gap using the trained regression model.")

    with c:
        st.markdown("### 03 · Evaluate")
        st.write("Inspect held-out test metrics and prediction errors.")

    st.info(
        "Scientific note: the target values are PBE-calculated band gaps. "
        "They are not automatically equivalent to experimental measurements "
        "or proof of solar-cell efficiency."
    )

# ---------------- MATERIAL RESEARCH WORKSPACE ----------------

elif page == "Material Research Workspace":
    st.markdown("""
<div class="solar-hero">
    <div class="section-label">MATERIALS DISCOVERY WORKSPACE</div>
    <h1>Research Workspace</h1>
    <p>Build a composition, evaluate it with the current model, and collect
    candidate estimates for side-by-side research review.</p>
</div>
""", unsafe_allow_html=True)

    st.caption("Workspace tools use the existing trained model and saved artifacts. No retraining is performed.")

    if "workspace_candidates" not in st.session_state:
        st.session_state.workspace_candidates = []
    if "workspace_current_candidate" not in st.session_state:
        st.session_state.workspace_current_candidate = None

    st.subheader("01 · Define a material composition")
    st.write("Choose up to four elements and enter their fractions. The fractions must add up to 1.0.")

    with st.form("workspace_composition_form"):
        candidate_label = st.text_input(
            "Candidate name (optional)",
            placeholder="e.g. Candidate A or Silicon oxide composition"
        )
        selected_elements = st.multiselect(
            "Select elements",
            options=elements,
            default=[],
            max_selections=4,
            placeholder="Search element symbols, e.g. Si, O, Ti",
            help="Choose up to four elements supported by the existing model."
        )

        fractions = {}
        if selected_elements:
            fraction_columns = st.columns(min(3, len(selected_elements)))
            for i, element in enumerate(selected_elements):
                with fraction_columns[i % len(fraction_columns)]:
                    fractions[element] = st.number_input(
                        f"{element} fraction",
                        min_value=0.0,
                        max_value=1.0,
                        value=1.0 / len(selected_elements),
                        step=0.001,
                        format="%.4f",
                        key=f"workspace_fraction_{element}"
                    )

        analyze = st.form_submit_button("Analyze composition", type="primary", use_container_width=True)

    if analyze:
        if not selected_elements:
            st.error("Select at least one element before analyzing the composition.")
        elif not np.isclose(sum(fractions.values()), 1.0, atol=0.01):
            st.error(f"The fractions sum to {sum(fractions.values()):.3f}. Please adjust them so the total is 1.000.")
        else:
            feature_row = {element: 0.0 for element in elements}
            feature_row.update(fractions)
            input_frame = pd.DataFrame([feature_row], columns=elements)
            try:
                predicted_gap = float(max(0, np.expm1(model.predict(input_frame)[0])))
                composition_text = " · ".join(f"{symbol} {fraction:.3f}" for symbol, fraction in fractions.items())
                st.session_state.workspace_current_candidate = {
                    "Candidate": candidate_label.strip() or f"Candidate {len(st.session_state.workspace_candidates) + 1}",
                    "Composition": composition_text,
                    "Predicted band gap (eV)": round(predicted_gap, 4),
                    "Fractions sum": round(sum(fractions.values()), 4),
                    "Element count": len(fractions),
                    "Fractions": dict(fractions),
                }
            except Exception as exc:
                st.error(f"The composition could not be analyzed: {exc}")

    current_candidate = st.session_state.workspace_current_candidate
    if current_candidate:
        st.divider()
        st.subheader("02 · Candidate analysis")
        result_left, result_right = st.columns([1.4, 1])
        with result_left:
            st.markdown("#### " + current_candidate["Candidate"])
            st.write(current_candidate["Composition"])
            st.caption("Composition fractions are model inputs, not a confirmation that this exact material structure exists or is stable.")
        with result_right:
            st.metric("Estimated band gap", f"{current_candidate['Predicted band gap (eV)']:.3f} eV")
        st.info("Interpretation: this is a composition-based model estimate of a PBE-calculated band gap. It is not an experimental measurement, stability assessment, or prediction of solar-cell efficiency.")

        if st.button("Add candidate to research collection", type="primary"):
            candidate_to_save = {key: value for key, value in current_candidate.items() if key != "Fractions"}
            existing = st.session_state.workspace_candidates
            is_duplicate = any(
                item.get("Composition") == candidate_to_save["Composition"]
                and item.get("Predicted band gap (eV)") == candidate_to_save["Predicted band gap (eV)"]
                for item in existing
            )
            if is_duplicate:
                st.warning("This composition and prediction are already in your collection.")
            else:
                existing.append(candidate_to_save)
                st.success("Candidate added to this session's research collection.")

    st.divider()
    st.subheader("03 · Research candidate collection")
    if st.session_state.workspace_candidates:
        candidates_df = pd.DataFrame(st.session_state.workspace_candidates)
        st.dataframe(candidates_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Export candidate collection (CSV)",
            data=candidates_df.to_csv(index=False),
            file_name="solarmatai_research_candidates.csv",
            mime="text/csv",
        )
        st.caption("The collection is held in the current Streamlit session. Export the CSV to keep a copy after the session ends.")
        if st.button("Clear candidate collection"):
            st.session_state.workspace_candidates = []
            st.session_state.workspace_current_candidate = None
            st.rerun()
    else:
        st.write("Your collection is empty. Analyze a composition and add it here to start comparing research candidates.")

    st.markdown("#### Research note")
    st.write("A lower or higher band gap alone does not determine whether a material is suitable for a solar cell. Crystal structure, stability, absorption, defects, and experimental validation also matter.")


elif page == "Intelligent Band-Gap Analysis":
    st.markdown("""
<div class="solar-hero">
    <div class="section-label">INTERPRETABLE MATERIALS INTELLIGENCE</div>
    <h1>Intelligent Band-Gap Analysis</h1>
    <p>Explore a model-estimated band gap, visualize the energy separation, and understand what the result can—and cannot—tell you.</p>
</div>
""", unsafe_allow_html=True)

    st.caption("This analysis uses the existing composition-based model. It does not retrain the model or verify a real crystal structure.")
    st.subheader("01 · Enter a composition")
    st.write("Choose up to four supported elements and enter their fractions. The fractions must sum to 1.0.")

    with st.form("intelligent_gap_form"):
        analysis_name = st.text_input("Candidate label (optional)", placeholder="e.g. Silicon dioxide composition")
        analysis_elements = st.multiselect(
            "Elements", options=elements, max_selections=4,
            placeholder="Search element symbols, e.g. Si, O, Ti"
        )
        analysis_fractions = {}
        if analysis_elements:
            fraction_columns = st.columns(min(3, len(analysis_elements)))
            for i, symbol in enumerate(analysis_elements):
                with fraction_columns[i % len(fraction_columns)]:
                    analysis_fractions[symbol] = st.number_input(
                        f"{symbol} fraction", min_value=0.0, max_value=1.0,
                        value=1.0 / len(analysis_elements), step=0.001,
                        format="%.4f", key=f"intelligent_fraction_{symbol}"
                    )
        run_analysis = st.form_submit_button("Run band-gap analysis", type="primary", use_container_width=True)

    if run_analysis:
        if not analysis_elements:
            st.error("Choose at least one element first.")
        elif not np.isclose(sum(analysis_fractions.values()), 1.0, atol=0.01):
            st.error(f"Your fractions total {sum(analysis_fractions.values()):.3f}. Adjust them to total 1.000.")
        else:
            feature_row = {symbol: 0.0 for symbol in elements}
            feature_row.update(analysis_fractions)
            try:
                input_frame = pd.DataFrame([feature_row], columns=elements)
                gap = float(max(0, np.expm1(model.predict(input_frame)[0])))
                st.session_state["intelligent_gap_result"] = {
                    "name": analysis_name.strip() or "Composition under analysis",
                    "fractions": dict(analysis_fractions),
                    "gap": gap,
                }
            except Exception as exc:
                st.error(f"The model could not analyze this composition: {exc}")

    result = st.session_state.get("intelligent_gap_result")
    if result:
        st.divider()
        st.subheader("02 · Analysis result")
        st.markdown(f"#### {result['name']}")
        st.write(" · ".join(f"{symbol} {fraction:.3f}" for symbol, fraction in result["fractions"].items()))
        metric_col, context_col = st.columns([1, 1.5])
        with metric_col:
            st.metric("Estimated PBE band gap", f"{result['gap']:.3f} eV")
        with context_col:
            st.write("**What this number represents**")
            st.write("An estimate from a machine-learning model trained on elemental composition features and computational PBE band-gap labels.")

        st.subheader("03 · Simplified energy-band diagram")
        fig, ax = plt.subplots(figsize=(8, 3.6))
        ax.set_xlim(0, 10)
        upper = max(result["gap"], 0.1)
        ax.set_ylim(-0.8, upper + 1.0)
        ax.axhspan(-0.55, 0.0, xmin=0.12, xmax=0.88, alpha=0.22)
        ax.axhspan(upper, upper + 0.45, xmin=0.12, xmax=0.88, alpha=0.22)
        ax.hlines(0, 1.2, 8.8, linewidth=3)
        ax.hlines(upper, 1.2, 8.8, linewidth=3)
        ax.annotate("Valence-band edge (schematic)", (5, -0.22), ha="center", va="top", fontsize=10)
        ax.annotate("Conduction-band edge (schematic)", (5, upper + 0.18), ha="center", va="bottom", fontsize=10)
        ax.annotate(f"Estimated gap: {result['gap']:.3f} eV", (8.95, upper / 2 if upper > 0.2 else 0.12),
                    ha="left", va="center", fontsize=10)
        ax.set_xticks([])
        ax.set_ylabel("Energy (schematic; eV scale not to scale)")
        ax.set_title("Conceptual band-gap illustration — not a calculated band structure")
        ax.spines[["top", "right", "bottom"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.subheader("04 · Interpretation and research cautions")
        if result["gap"] < 0.1:
            st.info("The model estimate is near zero. Treat this as a screening signal only; this model cannot establish metallic behavior or electronic structure.")
        elif result["gap"] < 1.0:
            st.info("The estimate is below 1 eV. Band gaps in this range can be relevant in some electronic or infrared-related contexts, but suitability depends on many other properties.")
        elif result["gap"] <= 2.0:
            st.info("The estimate is between 1 and 2 eV. This range can motivate further semiconductor or photovoltaic research, but the value alone does not establish solar-cell suitability.")
        else:
            st.info("The estimate is above 2 eV. Wider-gap materials may be relevant to some optoelectronic or insulating applications; actual use depends on structure and measured properties.")

        st.warning("Important limitations: this is not an experimental measurement, stability test, full crystal-structure calculation, or prediction of solar-cell efficiency. The model uses composition features only, so treat the result as an initial screening estimate.")
        report = pd.DataFrame([{
            "Candidate": result["name"],
            "Composition fractions": " · ".join(f"{k} {v:.4f}" for k, v in result["fractions"].items()),
            "Estimated PBE band gap (eV)": round(result["gap"], 5),
            "Method": "Composition-based ML estimate; not experimental",
        }])
        st.download_button("Export analysis summary (CSV)", data=report.to_csv(index=False),
                           file_name="solarmatai_bandgap_analysis.csv", mime="text/csv")


# ---------------- CANDIDATE SCREENING & COMPARISON ----------------

elif page == "Candidate Screening & Comparison":
    st.markdown("""
<div class="solar-hero">
    <div class="section-label">COMPOSITION-BASED CANDIDATE REVIEW</div>
    <h1>Candidate Screening & Comparison</h1>
    <p>Build a small shortlist of material compositions, compare their model-estimated PBE band gaps, and export the results for further research.</p>
</div>
""", unsafe_allow_html=True)
    st.caption("Screening ranks only the model-estimated band gap. It does not rank solar-cell efficiency, stability, safety, cost, or overall material quality.")

    if "screening_candidates" not in st.session_state:
        st.session_state.screening_candidates = []

    st.subheader("01 · Add a candidate")
    with st.form("screening_candidate_form"):
        screen_name = st.text_input("Candidate name", placeholder="e.g. Gallium oxide composition")
        screen_elements = st.multiselect(
            "Select elements", options=elements, max_selections=4,
            placeholder="Search symbols, e.g. Ga, O, Ti",
            help="Choose up to four elements supported by this model."
        )
        screen_fractions = {}
        if screen_elements:
            cols = st.columns(min(3, len(screen_elements)))
            for i, symbol in enumerate(screen_elements):
                with cols[i % len(cols)]:
                    screen_fractions[symbol] = st.number_input(
                        f"{symbol} fraction", min_value=0.0, max_value=1.0,
                        value=1.0 / len(screen_elements), step=0.001,
                        format="%.4f", key=f"screen_fraction_{symbol}"
                    )
        add_candidate = st.form_submit_button("Predict and add candidate", type="primary", use_container_width=True)

    if add_candidate:
        if not screen_elements:
            st.error("Select at least one element.")
        elif any(v <= 0 for v in screen_fractions.values()):
            st.error("Each selected element must have a fraction greater than zero.")
        elif not np.isclose(sum(screen_fractions.values()), 1.0, atol=0.001):
            st.error(f"Fractions total {sum(screen_fractions.values()):.4f}. Adjust them so the total is 1.0000.")
        else:
            feature_row = {symbol: 0.0 for symbol in elements}
            feature_row.update(screen_fractions)
            try:
                frame = pd.DataFrame([feature_row], columns=elements)
                gap = float(max(0, np.expm1(model.predict(frame)[0])))
                composition = " · ".join(f"{symbol} {screen_fractions[symbol]:.4f}" for symbol in screen_elements)
                record = {
                    "Candidate": screen_name.strip() or f"Candidate {len(st.session_state.screening_candidates) + 1}",
                    "Composition": composition,
                    "Estimated PBE band gap (eV)": round(gap, 4),
                    "Fraction total": round(sum(screen_fractions.values()), 4),
                    "Element count": len(screen_elements),
                }
                duplicate = any(x["Composition"] == composition for x in st.session_state.screening_candidates)
                if duplicate:
                    st.warning("This exact composition is already in the comparison table.")
                else:
                    st.session_state.screening_candidates.append(record)
                    st.success(f"Added {record['Candidate']} to the comparison table.")
            except Exception as exc:
                st.error(f"Could not predict this composition: {exc}")

    st.divider()
    st.subheader("02 · Compare candidates")
    workspace_items = st.session_state.get("workspace_candidates", [])
    if workspace_items and st.button("Import candidates from Material Research Workspace"):
        imported = 0
        for item in workspace_items:
            composition = item.get("Composition", "")
            if not any(x.get("Composition") == composition for x in st.session_state.screening_candidates):
                st.session_state.screening_candidates.append({
                    "Candidate": item.get("Candidate", "Workspace candidate"),
                    "Composition": composition,
                    "Estimated PBE band gap (eV)": item.get("Predicted band gap (eV)", np.nan),
                    "Fraction total": item.get("Fractions sum", np.nan),
                    "Element count": item.get("Element count", np.nan),
                })
                imported += 1
        if imported:
            st.success(f"Imported {imported} candidate(s) from the workspace.")
        else:
            st.info("No new candidates to import.")

    if st.session_state.screening_candidates:
        comparison_df = pd.DataFrame(st.session_state.screening_candidates)
        sort_order = st.selectbox("Sort candidates by", ["Lowest estimated band gap", "Highest estimated band gap", "Candidate name"])
        if sort_order == "Lowest estimated band gap":
            comparison_df = comparison_df.sort_values("Estimated PBE band gap (eV)", ascending=True, na_position="last")
        elif sort_order == "Highest estimated band gap":
            comparison_df = comparison_df.sort_values("Estimated PBE band gap (eV)", ascending=False, na_position="last")
        else:
            comparison_df = comparison_df.sort_values("Candidate", ascending=True)
        comparison_df = comparison_df.reset_index(drop=True)
        comparison_df.insert(0, "Comparison order", np.arange(1, len(comparison_df) + 1))
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

        chart_df = comparison_df.dropna(subset=["Estimated PBE band gap (eV)"])
        if not chart_df.empty:
            st.subheader("03 · Estimated band-gap comparison")
            fig, ax = plt.subplots(figsize=(9, max(3.2, 0.55 * len(chart_df))))
            ax.barh(chart_df["Candidate"], chart_df["Estimated PBE band gap (eV)"])
            ax.set_xlabel("Estimated PBE band gap (eV)")
            ax.set_ylabel("Candidate")
            ax.set_title("Model estimates only — not solar-efficiency ranking")
            ax.invert_yaxis()
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        st.download_button(
            "Export comparison (CSV)",
            data=comparison_df.to_csv(index=False),
            file_name="solarmatai_candidate_comparison.csv",
            mime="text/csv",
        )
        st.caption("This list is held in the current Streamlit session. Export the CSV to save a copy.")
        if st.button("Clear comparison list"):
            st.session_state.screening_candidates = []
            st.rerun()
    else:
        st.info("No candidates yet. Add a composition above, or import candidates from your Material Research Workspace.")

    st.warning("Scientific caution: band-gap value alone cannot establish whether a material is suitable for solar cells. Crystal structure, phase stability, absorption, defects, synthesis feasibility, and experimental validation are not evaluated here.")

# ---------------- BAND GAP PREDICTOR ----------------

elif page == "Band Gap Predictor":
    st.title("Band Gap Predictor")
    st.write(
        "Enter a composition as elemental fractions. "
        "Fractions must be non-negative and sum to 1."
    )

    st.warning(
        "This first version predicts from composition only. "
        "It does not account for crystal structure, defects, stability, "
        "or full solar-cell performance."
    )

    with st.form("composition_form"):
        st.subheader("Material composition")
        st.caption(
            "Example: for a two-element material with equal fractions, "
            "enter 0.5 for each element."
        )

        selected_elements = st.multiselect(
    "Choose elements",
    options=elements,
    default=[],
    max_selections=4,
    placeholder="Search for elements, e.g. Si, O, Ti",
    help="Select up to four elements for this prototype.",
)

        fractions = {}
        if selected_elements:
            columns = st.columns(min(3, len(selected_elements)))
            for i, element in enumerate(selected_elements):
                with columns[i % len(columns)]:
                    fractions[element] = st.number_input(
                        f"{element} fraction",
                        min_value=0.0,
                        max_value=1.0,
                        value=1.0 / len(selected_elements),
                        step=0.001,
                        format="%.4f",
                        key=f"fraction_{element}",
                    )

        submitted = st.form_submit_button(
            "Predict band gap",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        if not selected_elements:
            st.error("Please select at least one element.")
        else:
            total = sum(fractions.values())
            if not np.isclose(total, 1.0, atol=0.01):
                st.error(
                    f"The fractions currently sum to {total:.3f}. "
                    "Adjust them so the total is 1.0."
                )
            else:
                row = {element: 0.0 for element in elements}
                row.update(fractions)
                input_frame = pd.DataFrame([row], columns=elements)

                try:
                    prediction = float(max(0, np.expm1(model.predict(input_frame))[0]))
                    st.subheader("Prediction result")
                    st.metric("Estimated band gap", f"{prediction:.3f} eV")
                    st.caption(
                        "This is a model estimate for the entered composition, "
                        "not a verified material property. A composition-only "
                        "model may be unreliable for unseen or hypothetical materials."
                    )
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")

# ---------------- MODEL EVALUATION ----------------

elif page == "Model Evaluation":
    st.title("Model Intelligence & Evaluation")
    st.write(
        "Evaluate how the composition-based Ridge model performed on the held-out "
        "test set. These plots describe test-set errors; they do not validate "
        "experimental material properties."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Mean Absolute Error (MAE)", f"{metrics.get('mae_ev', 0.8505):.4f} eV")
    c2.metric("Root Mean Squared Error (RMSE)", f"{metrics.get('rmse_ev', 1.2392):.4f} eV")
    c3.metric("R² score", f"{metrics.get('r2', 0.3984):.4f}")

    with st.expander("What do these metrics mean?"):
        st.markdown(
            """
            - **MAE:** the average absolute difference between predicted and reference band gaps.
            - **RMSE:** an error measure that penalizes larger errors more strongly than MAE.
            - **R²:** how much of the variation in the reference values is explained by the model, relative to a mean baseline. It can be below zero on other test sets.
            """
        )

    baseline = metrics.get("baseline_mae_ev", 1.3264)
    model_mae = metrics.get("mae_ev", 0.8505)
    improvement = (baseline - model_mae) / baseline * 100 if baseline else 0

    st.subheader("Baseline comparison")
    compare = pd.DataFrame({
        "Method": ["Mean baseline", "Log-transformed Ridge"],
        "MAE (eV)": [baseline, model_mae],
    })
    st.bar_chart(compare.set_index("Method"))
    st.write(f"MAE reduction relative to baseline: **{improvement:.1f}%**")

    if PREDICTIONS_PATH.exists():
        predictions = pd.read_csv(PREDICTIONS_PATH)

        st.subheader("Held-out test predictions")
        st.caption("Preview of the saved predictions used for model evaluation.")
        st.dataframe(predictions.head(100), use_container_width=True)

        # Detect reference and prediction columns from common saved-file naming styles.
        normalized = {str(col).strip().lower().replace(" ", "_"): col for col in predictions.columns}
        actual_col = next(
            (original for norm, original in normalized.items()
             if any(term in norm for term in ("actual", "true", "target", "reference", "y_test", "gap_pbe"))
             and not any(term in norm for term in ("pred", "error", "absolute"))),
            None,
        )
        predicted_col = next(
            (original for norm, original in normalized.items()
             if "pred" in norm and any(term in norm for term in ("gap", "band", "value", "y", "pbe"))),
            None,
        )

        if actual_col is None or predicted_col is None:
            # Fallback: use the first two numeric columns if their names are not descriptive.
            numeric_cols = predictions.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                actual_col, predicted_col = numeric_cols[:2]

        if actual_col is not None and predicted_col is not None and actual_col != predicted_col:
            plot_data = predictions[[actual_col, predicted_col]].copy()
            plot_data[actual_col] = pd.to_numeric(plot_data[actual_col], errors="coerce")
            plot_data[predicted_col] = pd.to_numeric(plot_data[predicted_col], errors="coerce")
            plot_data = plot_data.replace([np.inf, -np.inf], np.nan).dropna()
            plot_data = plot_data.rename(columns={actual_col: "Reference", predicted_col: "Predicted"})

            if not plot_data.empty:
                residuals = plot_data["Predicted"] - plot_data["Reference"]

                st.subheader("Actual vs. predicted band gap")
                st.caption("Points closer to the diagonal indicate closer predictions. Axes are in eV.")
                fig, ax = plt.subplots(figsize=(7, 5))
                ax.scatter(plot_data["Reference"], plot_data["Predicted"], alpha=0.35, s=12)
                low = float(min(plot_data["Reference"].min(), plot_data["Predicted"].min()))
                high = float(max(plot_data["Reference"].max(), plot_data["Predicted"].max()))
                ax.plot([low, high], [low, high], linestyle="--")
                ax.set_xlabel("Reference PBE band gap (eV)")
                ax.set_ylabel("Predicted PBE band gap (eV)")
                ax.set_title("Test-set prediction parity")
                ax.grid(True, alpha=0.2)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                left, right = st.columns(2)
                with left:
                    st.subheader("Residual distribution")
                    st.caption("Residual = predicted − reference. Values near zero indicate smaller signed errors.")
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.hist(residuals, bins=35, edgecolor="white")
                    ax.axvline(0, linestyle="--")
                    ax.set_xlabel("Residual (eV)")
                    ax.set_ylabel("Number of test records")
                    ax.set_title("Distribution of prediction errors")
                    ax.grid(True, axis="y", alpha=0.2)
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                with right:
                    st.subheader("Residuals vs. predicted value")
                    st.caption("Patterns or widening spread can reveal systematic errors.")
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.scatter(plot_data["Predicted"], residuals, alpha=0.35, s=12)
                    ax.axhline(0, linestyle="--")
                    ax.set_xlabel("Predicted band gap (eV)")
                    ax.set_ylabel("Residual (eV)")
                    ax.set_title("Residual analysis")
                    ax.grid(True, alpha=0.2)
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
            else:
                st.warning("The saved predictions file has no usable numeric reference/prediction rows for plotting.")
        else:
            st.info(
                "The saved test-predictions file is available, but its columns could not be "
                "identified for the diagnostic plots. The metrics and CSV export are still available."
            )

        st.download_button(
            "Download test predictions CSV",
            data=predictions.to_csv(index=False),
            file_name="solarmatai_test_predictions.csv",
            mime="text/csv",
        )
    else:
        st.info("The saved test-predictions CSV is not available.")

    st.caption(
        "These results apply to the current random train/test split and the saved computational "
        "PBE reference labels. They do not establish experimental accuracy, material stability, "
        "or solar-cell efficiency."
    )

# ---------------- ABOUT ----------------

elif page == "About":
    st.title("About SolarMatAI")

    st.subheader("Project objective")
    st.write(
        "To demonstrate how artificial intelligence and chemistry can be "
        "combined to estimate a material property relevant to solar-energy research."
    )

    st.subheader("Machine-learning method")
    st.write(
        "The current model uses Ridge regression with elemental composition "
        "fractions as input features. It is evaluated against a mean-prediction baseline."
    )

    st.subheader("Dataset")
    st.write(
        "The experiment uses the MatBench MP gap dataset, with 106,113 records "
        "and the `gap pbe` target. These are computationally calculated reference values."
    )

    st.subheader("Limitations")
    st.markdown(
        """
        - The current model uses composition-only features.
        - Crystal structure and other important material properties are not included.
        - Predicted band gap alone cannot determine solar-cell efficiency.
        - Predictions for hypothetical or unfamiliar compositions need further validation.
        """
    )

    st.success("SolarMatAI dashboard loaded.")
st.markdown("""
<div class="solar-footer">
    SolarMatAI · AI-assisted materials research
    <br>
    Educational prototype · Predictions require scientific validation
</div>
""", unsafe_allow_html=True)   