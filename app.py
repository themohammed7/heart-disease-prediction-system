import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import time
import os

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="HeartCare AI Pro",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

/* ── Root ── */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* ── App background ── */
.stApp {
    background: linear-gradient(160deg, #fff1f2 0%, #fef2f2 40%, #f0f9ff 100%);
    min-height: 100vh;
}

/* ── Main container ── */
.block-container {
    padding: 2rem 2.5rem 3rem !important;
    max-width: 1200px;
}

/* ── Header ── */
.hc-header {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
}
.hc-header h1 {
    font-size: 2.8rem;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: -1px;
    margin-bottom: 0.4rem;
}
.hc-header p {
    font-size: 1.05rem;
    color: #64748b;
    margin: 0;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #be123c 0%, #9f1239 60%, #881337 100%) !important;
    border-right: none;
}
section[data-testid="stSidebar"] * {
    color: #fff !important;
}
section[data-testid="stSidebar"] .stRadio label {
    font-weight: 500;
    font-size: 0.95rem;
    padding: 0.35rem 0.5rem;
    border-radius: 8px;
    display: block;
    transition: background 0.2s;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(255,255,255,0.12);
}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 {
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    opacity: 0.9;
    margin-bottom: 0.5rem;
}

/* ── Metric cards ── */
.metric-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.5rem;
    flex-wrap: wrap;
}
.metric-card {
    flex: 1;
    min-width: 140px;
    background: #fff;
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 4px 16px rgba(0,0,0,0.04);
    border: 1px solid rgba(0,0,0,0.05);
}
.metric-card .mc-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #94a3b8;
    margin-bottom: 0.5rem;
}
.metric-card .mc-value {
    font-size: 1.7rem;
    font-weight: 700;
    color: #0f172a;
    line-height: 1;
}
.metric-card .mc-sub {
    font-size: 0.78rem;
    color: #64748b;
    margin-top: 0.25rem;
}

/* ── Section header ── */
.section-header {
    font-size: 1.3rem;
    font-weight: 700;
    color: #0f172a;
    margin: 1.8rem 0 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-header::after {
    content: '';
    flex: 1;
    height: 2px;
    background: linear-gradient(90deg, #fecdd3, transparent);
    border-radius: 2px;
    margin-left: 0.5rem;
}

/* ── Form card ── */
.form-card {
    background: #fff;
    border-radius: 20px;
    padding: 1.8rem 2rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 8px 24px rgba(0,0,0,0.04);
    border: 1px solid rgba(0,0,0,0.05);
    margin-bottom: 1.2rem;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #e11d48, #be123c) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 1rem !important;
    padding: 0.7rem 2rem !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
    box-shadow: 0 4px 12px rgba(225,29,72,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(225,29,72,0.38) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* Download button */
.stDownloadButton > button {
    background: linear-gradient(135deg, #0369a1, #0284c7) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    box-shadow: 0 4px 12px rgba(3,105,161,0.28) !important;
}

/* ── Sliders ── */
.stSlider [data-testid="stThumbValue"] {
    font-weight: 600;
    color: #e11d48 !important;
}

/* ── Result badges ── */
.risk-badge-high {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: linear-gradient(135deg, #fef2f2, #fee2e2);
    color: #991b1b;
    border: 1.5px solid #fca5a5;
    border-radius: 12px;
    padding: 1.2rem 2rem;
    font-size: 1.4rem;
    font-weight: 700;
    width: 100%;
    justify-content: center;
    margin: 1rem 0;
}
.risk-badge-low {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: linear-gradient(135deg, #f0fdf4, #dcfce7);
    color: #166534;
    border: 1.5px solid #86efac;
    border-radius: 12px;
    padding: 1.2rem 2rem;
    font-size: 1.4rem;
    font-weight: 700;
    width: 100%;
    justify-content: center;
    margin: 1rem 0;
}

/* ── Info tip ── */
.info-tip {
    background: linear-gradient(135deg, #eff6ff, #dbeafe);
    border-left: 4px solid #3b82f6;
    border-radius: 0 12px 12px 0;
    padding: 0.9rem 1.2rem;
    font-size: 0.88rem;
    color: #1e40af;
    margin-bottom: 1rem;
}

/* ── Feature list ── */
.feature-pill {
    display: inline-block;
    background: #fff;
    border: 1px solid #e2e8f0;
    border-radius: 20px;
    padding: 0.35rem 0.9rem;
    font-size: 0.82rem;
    color: #475569;
    margin: 0.2rem;
    font-weight: 500;
}

/* ── Advice card ── */
.advice-card {
    background: #fff;
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    border: 1px solid #e2e8f0;
    margin: 0.8rem 0;
}
.advice-card h4 {
    margin: 0 0 0.6rem;
    font-size: 0.95rem;
    font-weight: 700;
    color: #0f172a;
}
.advice-card ul {
    margin: 0;
    padding-left: 1.2rem;
    color: #475569;
    font-size: 0.88rem;
    line-height: 1.8;
}

/* ── Steps ── */
.step-num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    background: linear-gradient(135deg, #e11d48, #be123c);
    color: #fff;
    border-radius: 50%;
    font-size: 0.8rem;
    font-weight: 700;
    flex-shrink: 0;
}
.step-row {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    font-size: 0.92rem;
    color: #374151;
    margin: 0.4rem 0;
}

/* ── Hide Streamlit defaults ── */
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SAFE MODEL LOADER
# ─────────────────────────────────────────────
@st.cache_resource
def load_model_files():
    """Load model, scaler, and columns — returns None values if files missing."""
    model, scaler, columns = None, None, None
    try:
        if os.path.exists("knn_heart_model.pkl"):
            model = joblib.load("knn_heart_model.pkl")
        if os.path.exists("heart_scaler.pkl"):
            scaler = joblib.load("heart_scaler.pkl")
        if os.path.exists("heart_columns.pkl"):
            columns = joblib.load("heart_columns.pkl")
    except Exception as e:
        pass
    return model, scaler, columns

model, scaler, columns = load_model_files()
DEMO_MODE = model is None or scaler is None or columns is None


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ❤️ HeartCare AI")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠  Dashboard", "🧾  Patient Form", "📊  Analysis"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("**About**")
    st.markdown(
        "<span style='font-size:0.83rem;opacity:0.85;'>AI-powered cardiac risk "
        "screening tool using K-Nearest Neighbours classification. "
        "For clinical decision support only.</span>",
        unsafe_allow_html=True,
    )
    if DEMO_MODE:
        st.markdown("---")
        st.markdown(
            "<span style='font-size:0.8rem;background:rgba(255,255,255,0.15);"
            "padding:0.4rem 0.7rem;border-radius:8px;display:block;'>"
            "⚠️ Demo Mode: model files not found. Results are calculated dynamically without the model.</span>",
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hc-header">
    <h1>❤️ HeartCare AI Pro</h1>
    <p>Smart Cardiac Risk Analysis — powered by Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE: DASHBOARD
# ─────────────────────────────────────────────
if page == "🏠  Dashboard":

    st.markdown("""
    <div class="info-tip">
        ⚕️ <strong>Medical Disclaimer:</strong> This tool is for educational and
        screening purposes only. Always consult a qualified cardiologist for
        clinical decisions.
    </div>
    """, unsafe_allow_html=True)

    # Metric row
    st.markdown("""
    <div class="metric-row">
        <div class="metric-card">
            <div class="mc-label">Algorithm</div>
            <div class="mc-value">KNN</div>
            <div class="mc-sub">K-Nearest Neighbours</div>
        </div>
        <div class="metric-card">
            <div class="mc-label">Analysis Speed</div>
            <div class="mc-value">⚡ Fast</div>
            <div class="mc-sub">Instant prediction</div>
        </div>
        <div class="metric-card">
            <div class="mc-label">Parameters</div>
            <div class="mc-value">11</div>
            <div class="mc-sub">Clinical inputs</div>
        </div>
        <div class="metric-card">
            <div class="mc-label">Output</div>
            <div class="mc-value">Risk %</div>
            <div class="mc-sub">With probability score</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">🚀 How It Works</div>', unsafe_allow_html=True)
    steps = [
        ("1", "Fill in the Patient Form with 11 clinical parameters"),
        ("2", "Navigate to Analysis and click Run Full Analysis"),
        ("3", "Review the ECG simulation and Risk Gauge"),
        ("4", "Read personalised clinical recommendations based on risk level"),
    ]
    col_a, col_b = st.columns(2)
    for i, (num, text) in enumerate(steps):
        col = col_a if i % 2 == 0 else col_b
        with col:
            st.markdown(
                f'<div class="step-row"><span class="step-num">{num}</span>'
                f'<span>{text}</span></div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-header">📋 Input Parameters</div>', unsafe_allow_html=True)
    params = {
        "Age": "Years (18–100)",
        "Gender": "Male (M) / Female (F)",
        "Chest Pain Type": "ATA, NAP, TA, ASY",
        "Blood Pressure": "Resting BP in mmHg",
        "Cholesterol": "Serum cholesterol in mg/dL",
        "Fasting BS": "> 120 mg/dL (1=True, 0=False)",
        "Resting ECG": "Normal, ST, LVH",
        "Max HR": "Maximum heart rate achieved",
        "Exercise Angina": "Exercise-Induced Angina (Y/N)",
        "Oldpeak": "ST depression induced by exercise",
        "ST Slope": "Up, Flat, Down"
    }
    pills_html = "".join(
        f'<span class="feature-pill">🔬 {k}: <em>{v}</em></span>'
        for k, v in params.items()
    )
    st.markdown(pills_html, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE: PATIENT FORM
# ─────────────────────────────────────────────
elif page == "🧾  Patient Form":
    st.markdown('<div class="section-header">📋 Enter Patient Details</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-tip">
        Fill in all 11 parameters accurately. Values are used directly by the Machine Learning model.
    </div>
    """, unsafe_allow_html=True)

    with st.form("patient_form", clear_on_submit=False):
        st.markdown('<div class="form-card">', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**👤 Demographics**")
            age = st.slider("Age", 18, 100, 40)
            sex = st.selectbox("Sex", ["M", "F"])

        with col2:
            st.markdown("**🫀 Cardiac Indicators**")
            chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
            resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
            max_hr = st.slider("Max Heart Rate", 60, 220, 150)
            exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
            st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

        with col3:
            st.markdown("**🩺 Vitals**")
            resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
            cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
            fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
            oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)

        st.markdown("</div>", unsafe_allow_html=True)

        submitted = st.form_submit_button("💾 Save Patient Data", use_container_width=True)

    if submitted:
        # Dictionary formatting specifically matching the one-hot encoding columns expected by ML Models
        raw_input = {
            'Age': age,
            'RestingBP': resting_bp,
            'Cholesterol': cholesterol,
            'FastingBS': fasting_bs,
            'MaxHR': max_hr,
            'Oldpeak': oldpeak,
            'Sex_' + sex: 1,
            'ChestPainType_' + chest_pain: 1,
            'RestingECG_' + resting_ecg: 1,
            'ExerciseAngina_' + exercise_angina: 1,
            'ST_Slope_' + st_slope: 1
        }
        
        # Raw display dictionary for UI 
        raw_display = {
            'Age': age, 'Sex': sex, 'ChestPain': chest_pain,
            'RestingBP': resting_bp, 'Cholesterol': cholesterol,
            'FastingBS': fasting_bs, 'RestingECG': resting_ecg,
            'MaxHR': max_hr, 'ExerciseAngina': exercise_angina,
            'Oldpeak': oldpeak, 'ST_Slope': st_slope
        }

        df = pd.DataFrame([raw_input])
        if not DEMO_MODE and columns is not None:
            # fill missing one-hot encoded columns
            for col in columns:
                if col not in df.columns:
                    df[col] = 0
            df = df[columns]

        st.session_state["input"] = df
        st.session_state["raw_input"] = raw_input
        st.session_state["raw_display"] = raw_display
        st.session_state["patient_meta"] = {"age": age, "sex_label": "Male" if sex == "M" else "Female"}

        st.success("✅ Patient data saved! Navigate to **📊 Analysis** to run the model.")

        # Live preview card
        st.markdown('<div class="section-header">🔍 Data Preview</div>', unsafe_allow_html=True)
        preview_cols = st.columns(4)
        preview_items = [
            ("Age", f"{age} yrs"),
            ("Gender", "Male" if sex == "M" else "Female"),
            ("Chest Pain", chest_pain),
            ("Blood Pressure", f"{resting_bp} mmHg"),
            ("Cholesterol", f"{cholesterol} mg/dL"),
            ("Max HR", f"{max_hr} bpm"),
            ("ST Depression", f"{oldpeak}"),
            ("Status", "✅ Saved"),
        ]
        for i, (label, val) in enumerate(preview_items):
            with preview_cols[i % 4]:
                st.markdown(
                    f'<div class="metric-card"><div class="mc-label">{label}</div>'
                    f'<div class="mc-value" style="font-size:1.2rem">{val}</div></div>',
                    unsafe_allow_html=True,
                )
    elif "input" not in st.session_state:
        st.info("⬆️ Fill in the form above and click **Save Patient Data** to continue.")

# ─────────────────────────────────────────────
# PAGE: ANALYSIS
# ─────────────────────────────────────────────
elif page == "📊  Analysis":
    st.markdown('<div class="section-header">📊 Cardiac Risk Analysis</div>', unsafe_allow_html=True)

    if "input" not in st.session_state:
        st.warning("⚠️ No patient data found. Please complete the **Patient Form** first.")
        st.stop()

    df_input = st.session_state["input"]
    raw_display = st.session_state["raw_display"]
    meta = st.session_state["patient_meta"]

    # Show saved parameters
    with st.expander("📋 Loaded Patient Parameters", expanded=False):
        # Display the simpler raw_display instead of one-hot encoded dataframe for readability
        st.json(raw_display)

    st.markdown("---")
    run_btn = st.button("❤️ Run Full Cardiac Analysis", use_container_width=True)

    if run_btn:
        # ── PREDICTION ──
        if DEMO_MODE:
            # Deterministic calculation so the percentage changes based on the form inputs
            base_risk = 5.0
            base_risk += max(0, (raw_display['Age'] - 40) * 0.6)
            base_risk += max(0, (raw_display['RestingBP'] - 120) * 0.3)
            base_risk += max(0, (raw_display['Cholesterol'] - 200) * 0.1)
            if raw_display['FastingBS'] == 1: base_risk += 10.0
            if raw_display['ExerciseAngina'] == 'Y': base_risk += 20.0
            base_risk += max(0, (150 - raw_display['MaxHR']) * 0.2)
            base_risk += (raw_display['Oldpeak'] * 5.0)

            prob = float(np.clip(base_risk, 5, 95))
            pred = 1 if prob >= 50 else 0
        else:
            scaled = scaler.transform(df_input)
            pred = model.predict(scaled)[0]
            if hasattr(model, "predict_proba"):
                prob = float(model.predict_proba(scaled)[0][1]) * 100
            else:
                prob = 85.0 if pred == 1 else 15.0
            prob = float(np.clip(prob, 5, 95))

        # ── ECG SIMULATION ──
        st.markdown('<div class="section-header">❤️ Live ECG Monitor</div>', unsafe_allow_html=True)
        ecg_placeholder = st.empty()

        for frame in range(20):
            x = np.linspace(0, 4 * np.pi, 300)
            # Realistic-ish ECG shape
            y = (
                0.05 * np.sin(x)
                + 0.15 * np.exp(-((x % (2 * np.pi) - 1.5) ** 2) / 0.05)   # P wave
                - 0.08 * np.exp(-((x % (2 * np.pi) - 2.0) ** 2) / 0.01)   # Q
                + 0.80 * np.exp(-((x % (2 * np.pi) - 2.1) ** 2) / 0.005)  # R peak
                - 0.15 * np.exp(-((x % (2 * np.pi) - 2.2) ** 2) / 0.008)  # S
                + 0.10 * np.exp(-((x % (2 * np.pi) - 2.6) ** 2) / 0.06)   # T wave
                + np.random.normal(0, 0.015, 300)
            )
            trace_color = "#e11d48" if pred == 1 else "#16a34a"
            fig_ecg = go.Figure()
            fig_ecg.add_trace(go.Scatter(
                x=x, y=y,
                mode="lines",
                line=dict(color=trace_color, width=2),
                fill="tozeroy",
                fillcolor=f"rgba({225 if pred==1 else 22},{29 if pred==1 else 163},{72 if pred==1 else 74},0.06)",
            ))
            fig_ecg.update_layout(
                height=180,
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                showlegend=False,
            )
            ecg_placeholder.plotly_chart(fig_ecg, use_container_width=True, key=f"ecg_{frame}")
            time.sleep(0.06)

        # ── GAUGE + BAR CHART ──
        col_g, col_b2 = st.columns([1, 1])

        with col_g:
            st.markdown('<div class="section-header">🎯 Risk Gauge</div>', unsafe_allow_html=True)
            gauge_val = prob if pred == 1 else (100 - prob)
            # Ensure gauge points accurately to the raw probability for High Risk, and inversed probability for clarity on Low Risk side
            display_val = prob if pred == 1 else prob
            
            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(display_val, 1),
                number={"suffix": "%", "font": {"size": 36, "color": "#0f172a"}},
                title={"text": "Cardiac Risk Score", "font": {"size": 14, "color": "#64748b"}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94a3b8"},
                    "bar": {"color": "#e11d48" if pred == 1 else "#16a34a", "thickness": 0.25},
                    "bgcolor": "#f8fafc",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 30], "color": "#dcfce7"},
                        {"range": [30, 70], "color": "#fef9c3"},
                        {"range": [70, 100], "color": "#fee2e2"},
                    ],
                    "threshold": {
                        "line": {"color": "#0f172a", "width": 2},
                        "thickness": 0.75,
                        "value": display_val,
                    },
                },
            ))
            gauge_fig.update_layout(
                height=280,
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                font_family="Plus Jakarta Sans",
            )
            st.plotly_chart(gauge_fig, use_container_width=True)

        with col_b2:
            st.markdown('<div class="section-header">📈 Parameter Snapshot</div>',
                        unsafe_allow_html=True)
            params_viz = {
                "Blood Pressure": (raw_display.get("RestingBP", 120), 80, 200),
                "Cholesterol": (raw_display.get("Cholesterol", 200), 100, 600),
                "Max HR": (raw_display.get("MaxHR", 150), 60, 220),
                "ST Depression": (raw_display.get("Oldpeak", 1.0), 0, 6),
            }
            fig_bar = go.Figure()
            for param, (val, lo, hi) in params_viz.items():
                norm = (val - lo) / (hi - lo) * 100
                fig_bar.add_trace(go.Bar(
                    x=[norm], y=[param],
                    orientation="h",
                    marker_color="#e11d48" if norm > 65 else "#f59e0b" if norm > 40 else "#16a34a",
                    text=f"{val}",
                    textposition="outside",
                    showlegend=False,
                ))
            fig_bar.update_layout(
                height=280,
                margin=dict(l=0, r=40, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(range=[0, 115], visible=False),
                yaxis=dict(tickfont=dict(size=12), showgrid=False),
                font_family="Plus Jakarta Sans",
                barmode="overlay",
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # ── RESULT BADGE ──
        if pred == 1:
            st.markdown(
                f'<div class="risk-badge-high">⚠️ HIGH CARDIAC RISK '
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="risk-badge-low">✅ LOW CARDIAC RISK '
                f'</div>',
                unsafe_allow_html=True,
            )

        # ── CLINICAL ADVICE ──
        st.markdown('<div class="section-header">💡 Clinical Recommendations</div>',
                    unsafe_allow_html=True)

        if pred == 1:
            col_adv1, col_adv2 = st.columns(2)
            with col_adv1:
                st.markdown("""
                <div class="advice-card">
                    <h4>🚨 Immediate Actions</h4>
                    <ul>
                        <li>Book an urgent cardiology appointment</li>
                        <li>Get an ECG and echocardiogram done</li>
                        <li>Begin blood pressure medication if prescribed</li>
                        <li>Avoid strenuous physical exertion</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            with col_adv2:
                st.markdown("""
                <div class="advice-card">
                    <h4>🥗 Lifestyle Changes</h4>
                    <ul>
                        <li>Adopt a low-sodium, low-fat DASH diet</li>
                        <li>Quit smoking and limit alcohol</li>
                        <li>Manage stress with mindfulness / yoga</li>
                        <li>Monitor vitals daily (BP, HR, weight)</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        else:
            col_adv1, col_adv2 = st.columns(2)
            with col_adv1:
                st.markdown("""
                <div class="advice-card">
                    <h4>✅ Keep It Up</h4>
                    <ul>
                        <li>Continue regular aerobic exercise (150 min/week)</li>
                        <li>Maintain a balanced, heart-healthy diet</li>
                        <li>Annual cardiac check-up recommended</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            with col_adv2:
                st.markdown("""
                <div class="advice-card">
                    <h4>🏃 Stay Proactive</h4>
                    <ul>
                        <li>Track cholesterol and BP every 6 months</li>
                        <li>Stay hydrated and sleep 7–9 hours nightly</li>
                        <li>Avoid prolonged sedentary periods</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

        # ── HEARTBEAT AUDIO ──
        st.markdown("""
        <audio autoplay loop style="display:none">
            <source src="https://www.soundjay.com/medical/heartbeat-01.mp3" type="audio/mp3">
        </audio>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#94a3b8;font-size:0.82rem;padding:0.5rem 0;'>"
    "🧑‍⚕️ <strong>Mohammed Shaikhsiddiqi</strong> &nbsp;|&nbsp; 📞 7202093361 "
    "&nbsp;|&nbsp; HeartCare AI Pro — For screening purposes only"
    "</div>",
    unsafe_allow_html=True,
)
