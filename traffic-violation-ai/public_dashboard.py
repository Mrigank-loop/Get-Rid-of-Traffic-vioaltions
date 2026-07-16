import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="IntelliTraffic AI Dashboard",
    page_icon="🚦",
    layout="wide"
)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #020617, #0f172a, #111827);
    color: white;
}
.big-title {
    font-size: 44px;
    font-weight: 900;
    background: linear-gradient(90deg, #38bdf8, #22c55e, #facc15);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.card {
    background: rgba(15, 23, 42, 0.95);
    padding: 22px;
    border-radius: 18px;
    border: 1px solid rgba(56, 189, 248, 0.35);
    box-shadow: 0px 0px 20px rgba(56, 189, 248, 0.12);
}
.metric-value {
    font-size: 32px;
    font-weight: 900;
    color: #38bdf8;
}
.metric-label {
    color: #cbd5e1;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">🚦 IntelliTraffic AI</div>', unsafe_allow_html=True)
st.write("### Smart Traffic Violation Detection & Enforcement Intelligence Dashboard")
st.write("AI-powered prototype for metropolitan traffic monitoring, violation detection, evidence generation, severity scoring, demo fine analytics, and smart-city enforcement insights.")

st.divider()

csv_files = sorted(DATA_DIR.glob("*.csv")) if DATA_DIR.exists() else []

all_data = []

for file in csv_files:
    try:
        df = pd.read_csv(file)
        df["source_report"] = file.name
        all_data.append(df)
    except Exception:
        pass

if not all_data:
    st.error("No demo CSV reports found. Please push CSV files inside traffic-violation-ai/data folder.")
    st.stop()

df = pd.concat(all_data, ignore_index=True)

total_records = len(df)
total_reports = len(csv_files)

high_cases = 0
medium_cases = 0
total_fine = 0

if "severity_level" in df.columns:
    sev = df["severity_level"].astype(str).str.upper()
    high_cases = len(df[sev == "HIGH"])
    medium_cases = len(df[sev == "MEDIUM"])

if "demo_fine_amount" in df.columns:
    total_fine = pd.to_numeric(df["demo_fine_amount"], errors="coerce").fillna(0).sum()

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f'<div class="card"><div class="metric-value">{total_records}</div><div class="metric-label">Evidence Records</div></div>', unsafe_allow_html=True)

with c2:
    st.markdown(f'<div class="card"><div class="metric-value">{total_reports}</div><div class="metric-label">Reports Loaded</div></div>', unsafe_allow_html=True)

with c3:
    st.markdown(f'<div class="card"><div class="metric-value">{high_cases}</div><div class="metric-label">High Severity Cases</div></div>', unsafe_allow_html=True)

with c4:
    st.markdown(f'<div class="card"><div class="metric-value">₹{int(total_fine)}</div><div class="metric-label">Demo Fine Amount</div></div>', unsafe_allow_html=True)

st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Analytics",
    "📄 Evidence Reports",
    "🚓 Enforcement Intelligence",
    "🧠 About Prototype"
])

with tab1:
    st.subheader("📊 Report-wise Evidence Count")
    st.bar_chart(df["source_report"].value_counts())

    if "severity_level" in df.columns:
        st.subheader("⚠️ Severity Distribution")
        st.bar_chart(df["severity_level"].astype(str).value_counts())

    if "event_type" in df.columns:
        st.subheader("🚦 Event Type Distribution")
        st.bar_chart(df["event_type"].astype(str).value_counts())

    if "demo_fine_amount" in df.columns:
        st.subheader("💰 Demo Fine by Report")
        temp = df.copy()
        temp["demo_fine_amount"] = pd.to_numeric(temp["demo_fine_amount"], errors="coerce").fillna(0)
        st.bar_chart(temp.groupby("source_report")["demo_fine_amount"].sum())

with tab2:
    st.subheader("📄 All Evidence Reports")

    report_names = [file.name for file in csv_files]
    selected = st.selectbox("Select report", report_names)

    selected_path = DATA_DIR / selected
    selected_df = pd.read_csv(selected_path)

    st.write(f"Rows: {len(selected_df)} | Columns: {len(selected_df.columns)}")
    st.dataframe(selected_df, width="stretch", height=450)

    with open(selected_path, "rb") as f:
        st.download_button(
            label=f"⬇️ Download {selected}",
            data=f.read(),
            file_name=selected,
            mime="text/csv"
        )

    st.subheader("🧠 Combined Evidence Table")
    st.dataframe(df, width="stretch", height=500)

with tab3:
    st.subheader("🚓 Enforcement Intelligence Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### Detection Modules
        - Triple riding detection  
        - Possible helmet-risk detection  
        - Wrong-direction detection  
        - Rain / wet-road risk analysis  
        - License Plate OCR / ANPR prototype  
        - All-in-one multi-violation detection  
        """)

    with col2:
        st.markdown("""
        ### Intelligence Layer
        - Evidence ID generation  
        - Timestamped reports  
        - Severity scoring  
        - Demo fine calculation  
        - Action recommendation  
        - Dashboard-based analytics  
        """)

    st.success("IntelliTraffic AI converts raw traffic footage into actionable smart-city enforcement intelligence.")

with tab4:
    st.subheader("🧠 IntelliTraffic AI Prototype")

    st.write("""
    IntelliTraffic AI is an AI-powered traffic violation detection and enforcement intelligence prototype
    designed for metropolitan traffic challenges. It helps reduce manual CCTV monitoring by converting
    traffic footage into structured evidence, analytics, severity scores, demo fines, and action recommendations.
    """)

    st.markdown("""
    **Tech Stack:** Python, YOLOv8, OpenCV, EasyOCR, Streamlit, Pandas, FastAPI-ready architecture.
    """)

st.caption("Built as a working prototype for smart-city traffic enforcement intelligence.")
