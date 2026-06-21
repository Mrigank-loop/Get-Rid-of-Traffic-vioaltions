import streamlit as st
from pathlib import Path
import pandas as pd

st.set_page_config(
    page_title="IntelliTraffic AI Dashboard",
    page_icon="🚦",
    layout="wide"
)

st.title("🚦 IntelliTraffic AI Violation Detection Dashboard")

st.write(
    "AI-powered prototype for detecting traffic violations from CCTV and road video footage."
)

st.markdown("---")

# Metrics
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Violations", "2")
col2.metric("Wrong-Side Cases", "1")
col3.metric("Helmet Violations", "1")
col4.metric("Evidence Status", "Generated")

st.markdown("---")

# Project overview
st.subheader("📌 Project Overview")

st.write(
    """
    IntelliTraffic is an AI-based traffic violation detection prototype.
    It uses computer vision and YOLO-based object detection to identify road violations
    such as wrong-side driving and helmet non-compliance from video footage.
    """
)

st.markdown("---")

# Detection modules
st.subheader("🧠 Detection Modules")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 1. Wrong-Side Driving Detection")
    st.write(
        """
        - Detects vehicles on flyover/road footage  
        - Tracks vehicle movement direction  
        - Compares movement with valid traffic direction  
        - Marks violation if movement is opposite  
        - Generates evidence frame  
        """
    )

with col2:
    st.markdown("### 2. Helmet Violation Detection")
    st.write(
        """
        - Detects motorcycles and riders  
        - Identifies possible no-helmet cases  
        - Marks violation on video frame  
        - Saves evidence with timestamp  
        - Helps reduce manual monitoring  
        """
    )

st.markdown("---")

# Violation records
st.subheader("📊 Violation Records")

records = [
    {
        "Violation ID": "V001",
        "Violation Type": "Wrong-Side Driving",
        "Location": "Flyover Road",
        "Detection Status": "Violation Detected",
        "Evidence": "Generated"
    },
    {
        "Violation ID": "V002",
        "Violation Type": "No Helmet",
        "Location": "Traffic Road",
        "Detection Status": "Violation Detected",
        "Evidence": "Generated"
    }
]

df = pd.DataFrame(records)
st.dataframe(df, use_container_width=True)

st.markdown("---")

# Evidence screenshots
st.subheader("🖼️ Evidence Screenshots")

evidence_folder = Path("outputs/evidence")

if evidence_folder.exists():
    image_files = (
        list(evidence_folder.glob("*.jpg"))
        + list(evidence_folder.glob("*.jpeg"))
        + list(evidence_folder.glob("*.png"))
    )

    if image_files:
        cols = st.columns(2)

        for index, image_path in enumerate(image_files[:6]):
            with cols[index % 2]:
                st.image(
                    str(image_path),
                    caption=image_path.name,
                    use_container_width=True
                )
    else:
        st.info(
            "Evidence folder exists, but no screenshots are available yet. "
            "Add output screenshots inside outputs/evidence."
        )
else:
    st.warning(
        "Evidence folder not found. Create outputs/evidence and add output screenshots."
    )

st.markdown("---")

# Architecture
st.subheader("⚙️ System Workflow")

st.code(
    """
Traffic Video
     ↓
OpenCV Frame Processing
     ↓
YOLO Object Detection
     ↓
Vehicle / Rider Detection
     ↓
Violation Logic
     ↓
Evidence Frame Generation
     ↓
Dashboard / Report
    """,
    language="text"
)

st.markdown("---")

# Tech stack
st.subheader("🛠️ Tech Stack")

tech_col1, tech_col2, tech_col3 = st.columns(3)

with tech_col1:
    st.write("**AI / ML**")
    st.write("- YOLOv8")
    st.write("- Ultralytics")
    st.write("- Computer Vision")

with tech_col2:
    st.write("**Backend / Logic**")
    st.write("- Python")
    st.write("- OpenCV")
    st.write("- FastAPI")

with tech_col3:
    st.write("**Frontend / Demo**")
    st.write("- Streamlit")
    st.write("- GitHub")
    st.write("- Evidence Dashboard")

st.markdown("---")

# Local run instructions
st.subheader("🚀 Instructions to Run Locally")

st.code(
    """
git clone https://github.com/Mrigank-loop/Get-Rid-of-Traffic-vioaltions.git
cd Get-Rid-of-Traffic-vioaltions/traffic-violation-ai

pip install -r requirements.txt

python wrong_side_detector.py
python helmet_detector.py

python -m streamlit run demo_dashboard.py
    """,
    language="bash"
)

st.markdown("---")

# Impact
st.subheader("🌍 Impact")

st.write(
    """
    This prototype can help traffic authorities reduce manual monitoring effort,
    detect violations faster, and generate evidence-based reports for safer roads.
    """
)

st.success("Prototype dashboard ready for demo submission.")
