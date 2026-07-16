import streamlit as st
import pandas as pd
from pathlib import Path
import cv2
import os
import base64

st.set_page_config(
    page_title="IntelliTraffic AI Premium Dashboard",
    page_icon="🚦",
    layout="wide"
)

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #050816 0%, #0f172a 45%, #111827 100%);
    color: white;
}

.main-title {
    font-size: 44px;
    font-weight: 900;
    background: linear-gradient(90deg, #38bdf8, #22c55e, #facc15);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0px;
}

.sub-title {
    font-size: 18px;
    color: #cbd5e1;
    margin-bottom: 25px;
}

.metric-card {
    background: rgba(15, 23, 42, 0.95);
    padding: 22px;
    border-radius: 20px;
    border: 1px solid rgba(56, 189, 248, 0.35);
    box-shadow: 0px 0px 25px rgba(56, 189, 248, 0.12);
}

.metric-value {
    font-size: 34px;
    font-weight: 900;
    color: #38bdf8;
}

.metric-label {
    font-size: 14px;
    color: #cbd5e1;
}

.section-box {
    background: rgba(15, 23, 42, 0.78);
    border-radius: 20px;
    padding: 20px;
    border: 1px solid rgba(148, 163, 184, 0.25);
    margin-top: 12px;
    margin-bottom: 12px;
}

.video-card {
    background: rgba(2, 6, 23, 0.85);
    border-radius: 18px;
    padding: 18px;
    border: 1px solid rgba(34, 197, 94, 0.35);
    margin-bottom: 18px;
}

.badge {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 999px;
    background: rgba(34, 197, 94, 0.16);
    border: 1px solid rgba(34, 197, 94, 0.4);
    color: #86efac;
    font-size: 13px;
    margin-right: 8px;
}

.warning-badge {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 999px;
    background: rgba(250, 204, 21, 0.14);
    border: 1px solid rgba(250, 204, 21, 0.45);
    color: #fde68a;
    font-size: 13px;
    margin-right: 8px;
}

.danger-badge {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 999px;
    background: rgba(239, 68, 68, 0.14);
    border: 1px solid rgba(239, 68, 68, 0.45);
    color: #fca5a5;
    font-size: 13px;
    margin-right: 8px;
}
</style>
""", unsafe_allow_html=True)


def safe_read_csv(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception:
        return None


def get_video_info(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0, 0, 0, 0

    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    duration = frames / fps if fps and fps > 0 else 0
    return frames, fps, width, height, duration


def extract_preview_frames(video_path, count=4):
    frames_list = []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return frames_list

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return frames_list

    positions = []
    for ratio in [0.12, 0.32, 0.52, 0.72]:
        positions.append(int(total_frames * ratio))

    for pos in positions[:count]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()

        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_list.append((pos, frame))

    cap.release()
    return frames_list


def download_button_for_file(path, label, mime):
    try:
        with open(path, "rb") as f:
            st.download_button(
                label=label,
                data=f.read(),
                file_name=Path(path).name,
                mime=mime
            )
    except Exception:
        st.warning(f"Could not prepare download for {Path(path).name}")


st.markdown('<div class="main-title">🚦 IntelliTraffic AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Premium Traffic Violation Detection & Smart Enforcement Intelligence Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown("""
<span class="badge">YOLOv8 + OpenCV</span>
<span class="badge">License Plate OCR</span>
<span class="warning-badge">Severity Scoring</span>
<span class="danger-badge">Multi-Violation Detection</span>
<span class="badge">FastAPI Ready</span>
""", unsafe_allow_html=True)

st.divider()

csv_files = sorted(DATA_DIR.glob("*.csv")) if DATA_DIR.exists() else []
video_files = sorted(OUTPUT_DIR.glob("*.mp4")) if OUTPUT_DIR.exists() else []

all_frames = []

for csv_file in csv_files:
    df_temp = safe_read_csv(csv_file)
    if df_temp is not None:
        df_temp["source_report"] = csv_file.name
        all_frames.append(df_temp)

if all_frames:
    all_df = pd.concat(all_frames, ignore_index=True)
else:
    all_df = pd.DataFrame()

total_records = len(all_df)
total_reports = len(csv_files)
total_videos = len(video_files)

high_cases = 0
medium_cases = 0
total_fine = 0

if not all_df.empty:
    if "severity_level" in all_df.columns:
        sev = all_df["severity_level"].astype(str).str.upper()
        high_cases = len(all_df[sev == "HIGH"])
        medium_cases = len(all_df[sev == "MEDIUM"])

    if "demo_fine_amount" in all_df.columns:
        total_fine = pd.to_numeric(all_df["demo_fine_amount"], errors="coerce").fillna(0).sum()

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{total_records}</div>
        <div class="metric-label">Evidence Records</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{total_reports}</div>
        <div class="metric-label">Reports Loaded</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{total_videos}</div>
        <div class="metric-label">Output Videos</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{high_cases}</div>
        <div class="metric-label">High Severity Cases</div>
    </div>
    """, unsafe_allow_html=True)

with c5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">₹{int(total_fine)}</div>
        <div class="metric-label">Demo Fine Amount</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎥 All Video Evidence",
    "📄 All Reports",
    "📊 Analytics",
    "🔍 OCR / Plate Evidence",
    "🚓 Enforcement Summary"
])

with tab1:
    st.subheader("🎥 Processed Video Evidence Gallery")

    if not video_files:
        st.error("No output videos found inside outputs folder.")
    else:
        st.success(f"{len(video_files)} processed videos found.")

        for video_path in video_files:
            st.markdown('<div class="video-card">', unsafe_allow_html=True)

            frames, fps, width, height, duration = get_video_info(video_path)
            size_mb = os.path.getsize(video_path) / (1024 * 1024)

            st.markdown(f"### 🎬 {video_path.name}")
            st.write(
                f"**Resolution:** {width}x{height} | "
                f"**Duration:** {duration:.1f}s | "
                f"**FPS:** {fps:.1f} | "
                f"**Size:** {size_mb:.2f} MB"
            )

            preview_frames = extract_preview_frames(video_path, count=4)

            if preview_frames:
                st.write("**Preview Frames:**")
                frame_cols = st.columns(4)

                for idx, (frame_no, frame_img) in enumerate(preview_frames):
                    frame_cols[idx].image(
                        frame_img,
                        caption=f"Frame {frame_no}",
                        width="stretch"
                    )
            else:
                st.warning("Could not extract preview frames from this video.")

            with st.expander("▶ Try browser video player"):
                try:
                    with open(video_path, "rb") as vf:
                        st.video(vf.read())
                except Exception as e:
                    st.warning("Browser video preview failed.")
                    st.write(e)

            download_button_for_file(
                video_path,
                f"⬇️ Download {video_path.name}",
                "video/mp4"
            )

            st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.subheader("📄 All CSV Evidence Reports")

    if not csv_files:
        st.error("No CSV reports found inside data folder.")
    else:
        report_tabs = st.tabs([file.name for file in csv_files])

        for idx, csv_file in enumerate(csv_files):
            with report_tabs[idx]:
                df_report = safe_read_csv(csv_file)

                if df_report is None:
                    st.error(f"Could not read {csv_file.name}")
                    continue

                st.write(f"**Report:** `{csv_file.name}`")
                st.write(f"**Rows:** {len(df_report)} | **Columns:** {len(df_report.columns)}")

                st.dataframe(df_report, width="stretch", height=420)

                download_button_for_file(
                    csv_file,
                    f"⬇️ Download {csv_file.name}",
                    "text/csv"
                )

    st.divider()

    st.subheader("🧠 Combined Evidence Table")

    if all_df.empty:
        st.warning("No combined evidence available.")
    else:
        st.dataframe(all_df, width="stretch", height=520)

with tab3:
    st.subheader("📊 Violation Analytics")

    if all_df.empty:
        st.warning("No analytics available because CSV data is empty.")
    else:
        left, right = st.columns(2)

        with left:
            st.markdown("### Report-wise Evidence Count")
            st.bar_chart(all_df["source_report"].value_counts())

        with right:
            if "severity_level" in all_df.columns:
                st.markdown("### Severity Distribution")
                st.bar_chart(all_df["severity_level"].astype(str).value_counts())
            else:
                st.info("No severity_level column available.")

        left2, right2 = st.columns(2)

        with left2:
            if "event_type" in all_df.columns:
                st.markdown("### Event Type Distribution")
                st.bar_chart(all_df["event_type"].astype(str).value_counts())
            elif "status" in all_df.columns:
                st.markdown("### Status Distribution")
                st.bar_chart(all_df["status"].astype(str).value_counts())
            else:
                st.info("No event/status column available.")

        with right2:
            if "demo_fine_amount" in all_df.columns:
                st.markdown("### Demo Fine by Report")
                fine_df = all_df.copy()
                fine_df["demo_fine_amount"] = pd.to_numeric(
                    fine_df["demo_fine_amount"],
                    errors="coerce"
                ).fillna(0)
                st.bar_chart(fine_df.groupby("source_report")["demo_fine_amount"].sum())
            else:
                st.info("No fine column available.")

with tab4:
    st.subheader("🔍 License Plate / OCR Evidence")

    plate_dirs = [
        OUTPUT_DIR / "all_in_one_plate_evidence",
        OUTPUT_DIR / "plate_evidence",
        OUTPUT_DIR / "license_plate_evidence",
    ]

    image_files = []

    for plate_dir in plate_dirs:
        if plate_dir.exists():
            image_files.extend(list(plate_dir.glob("*.jpg")))
            image_files.extend(list(plate_dir.glob("*.png")))

    if not image_files:
        st.warning("No plate evidence images found yet.")
    else:
        st.success(f"{len(image_files)} plate evidence images found.")

        cols = st.columns(4)

        for idx, image_path in enumerate(image_files):
            cols[idx % 4].image(
                str(image_path),
                caption=image_path.name,
                width="stretch"
            )

    st.divider()

    if not all_df.empty and "plate_text" in all_df.columns:
        plate_df = all_df[all_df["plate_text"].astype(str).str.strip() != ""]
        st.subheader("Detected Plate Records")
        st.dataframe(plate_df, width="stretch")
    else:
        st.info("No plate_text column available in reports.")

with tab5:
    st.subheader("🚓 AI Enforcement Intelligence Summary")

    st.markdown("""
    <div class="section-box">
    <h3>What this dashboard demonstrates</h3>
    <p>
    IntelliTraffic AI converts raw traffic videos into structured smart-city enforcement intelligence.
    The system supports multi-violation detection, wrong-direction detection, triple-riding detection,
    helmet-risk monitoring, rain/wet-road risk analysis, license plate OCR evidence, severity scoring,
    demo fine generation, and dashboard-based analytics.
    </p>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
        ### ✅ Detection Modules
        - Vehicle and pedestrian detection  
        - Triple riding detection  
        - Possible no-helmet risk detection  
        - Wrong-direction movement detection  
        - Rain / wet road risk analysis  
        - License Plate OCR / ANPR prototype  
        - All-in-one multi-violation detector  
        """)

    with col_b:
        st.markdown("""
        ### ✅ Enforcement Intelligence
        - Evidence ID generation  
        - Timestamped CSV reports  
        - Severity scoring  
        - Demo fine calculation  
        - Action recommendation  
        - Video evidence preview  
        - FastAPI and MongoDB-ready backend  
        """)

    st.success(
        "Final Impact: IntelliTraffic AI does not only detect violations; "
        "it converts traffic footage into actionable enforcement intelligence."
    )

st.divider()
st.caption("Built with Python, YOLOv8, OpenCV, EasyOCR, Pandas, Streamlit, FastAPI-ready architecture.")
