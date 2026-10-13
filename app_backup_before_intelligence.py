
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
    ["Overview", "Material Research Workspace", "Band Gap Predictor", "Model Evaluation", "About"]
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
                        step=0.05,
                        format="%.3f",
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
                        step=0.05,
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
    st.title("Model Evaluation")
    st.write("Results from the held-out test set of the training experiment.")

    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{metrics.get('mae_ev', 0.8505):.4f} eV")
    c2.metric("RMSE", f"{metrics.get('rmse_ev', 1.2392):.4f} eV")
    c3.metric("R² score", f"{metrics.get('r2', 0.3984):.4f}")

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
        st.dataframe(predictions.head(100), use_container_width=True)
        st.download_button(
            "Download test predictions CSV",
            data=predictions.to_csv(index=False),
            file_name="solarmatai_test_predictions.csv",
            mime="text/csv",
        )
    else:
        st.info("The saved test-predictions CSV is not available.")

    st.caption(
        "The test metrics apply to the current random train/test split. "
        "They do not establish experimental accuracy or solar-cell efficiency."
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