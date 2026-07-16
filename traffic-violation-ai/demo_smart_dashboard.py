import streamlit as st
import pandas as pd
from pathlib import Path
import cv2
import os

st.set_page_config(
    page_title="IntelliTraffic AI Dashboard",
    page_icon="🚦",
    layout="wide"
)

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")

st.title("🚦 IntelliTraffic AI — Premium Demo Dashboard")
st.caption("AI-powered traffic violation detection and enforcement intelligence prototype")

csv_files = list(DATA_DIR.glob("*.csv"))

if not csv_files:
    st.error("No CSV reports found inside data folder.")
    st.stop()

all_data = []

for file in csv_files:
    try:
        df_temp = pd.read_csv(file)
        df_temp["source_report"] = file.name
        all_data.append(df_temp)
    except Exception:
        pass

if not all_data:
    st.error("CSV files found, but could not read them.")
    st.stop()

df = pd.concat(all_data, ignore_index=True)

severity_col = "severity_level" if "severity_level" in df.columns else None
fine_col = "demo_fine_amount" if "demo_fine_amount" in df.columns else None

high_risk = 0
if severity_col:
    high_risk = len(df[df[severity_col].astype(str).str.upper() == "HIGH"])

total_fine = 0
if fine_col:
    total_fine = pd.to_numeric(df[fine_col], errors="coerce").fillna(0).sum()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Evidence Records", len(df))
col2.metric("CSV Reports Loaded", len(csv_files))
col3.metric("High Severity Cases", high_risk)
col4.metric("Demo Fine Amount", f"₹{int(total_fine)}")

st.divider()

left, right = st.columns([1.2, 1])

with left:
    st.subheader("📊 Report-wise Evidence Count")
    st.bar_chart(df["source_report"].value_counts())

with right:
    st.subheader("⚠️ Severity Distribution")
    if severity_col:
        st.bar_chart(df[severity_col].astype(str).value_counts())
    else:
        st.info("Severity column not available.")

st.divider()

st.subheader("🎥 Processed Output Video Evidence")

video_files = list(OUTPUT_DIR.glob("*.mp4"))

if not video_files:
    st.warning("No processed videos found inside outputs folder.")
else:
    selected_video = st.selectbox(
        "Select output video",
        [video.name for video in video_files]
    )

    video_path = OUTPUT_DIR / selected_video
    size_mb = os.path.getsize(video_path) / (1024 * 1024)

    st.write(f"**Selected video:** `{selected_video}`")
    st.write(f"**Size:** `{size_mb:.2f} MB`")

    try:
        with open(video_path, "rb") as video_file:
            video_bytes = video_file.read()
            st.video(video_bytes)
    except Exception as e:
        st.warning("Browser preview failed, showing frame preview instead.")
        st.write(e)

    st.download_button(
        label="⬇️ Download / Open Video Evidence",
        data=open(video_path, "rb").read(),
        file_name=selected_video,
        mime="video/mp4"
    )

    st.subheader("🖼️ Video Frame Preview")

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames > 0:
        frame_positions = [
            int(total_frames * 0.15),
            int(total_frames * 0.35),
            int(total_frames * 0.55),
            int(total_frames * 0.75),
        ]

        cols = st.columns(4)

        for idx, frame_no in enumerate(frame_positions):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = cap.read()

            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cols[idx].image(
                    frame,
                    caption=f"Frame {frame_no}",
                    width="stretch"
                )

    cap.release()

st.divider()

st.subheader("🧠 Violation Intelligence Table")
st.dataframe(df, width="stretch")

st.divider()

st.subheader("🚓 AI Enforcement Summary")

st.write("""
IntelliTraffic AI converts raw traffic footage into structured enforcement intelligence.
The system supports multi-violation detection, wrong-direction detection, triple-riding detection,
helmet-risk monitoring, rain/wet-road risk analysis, license plate OCR evidence, severity scoring,
demo fine generation, and dashboard-based analytics.
""")

st.success("Dashboard is running successfully.")
