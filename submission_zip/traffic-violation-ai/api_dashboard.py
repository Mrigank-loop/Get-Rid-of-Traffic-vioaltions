import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------

st.set_page_config(
    page_title="IntelliTraffic AI Dashboard",
    page_icon="🚦",
    layout="wide"
)

# -------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------

st.markdown(
    """
    <style>
        .main {
            background: linear-gradient(135deg, #020617 0%, #0f172a 45%, #111827 100%);
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        .hero-card {
            padding: 35px;
            border-radius: 24px;
            background: linear-gradient(135deg, #1e3a8a 0%, #0f172a 55%, #020617 100%);
            border: 1px solid rgba(56, 189, 248, 0.35);
            box-shadow: 0 20px 60px rgba(0,0,0,0.35);
            margin-bottom: 25px;
        }

        .hero-title {
            font-size: 46px;
            font-weight: 900;
            color: white;
            margin-bottom: 8px;
        }

        .hero-subtitle {
            font-size: 18px;
            color: #cbd5e1;
            max-width: 900px;
        }

        .badge {
            display: inline-block;
            padding: 8px 14px;
            border-radius: 999px;
            background: rgba(34, 197, 94, 0.15);
            color: #4ade80;
            border: 1px solid rgba(34, 197, 94, 0.4);
            font-weight: 700;
            margin-bottom: 18px;
        }

        .metric-card {
            padding: 24px;
            border-radius: 20px;
            background: rgba(15, 23, 42, 0.95);
            border: 1px solid rgba(148, 163, 184, 0.25);
            box-shadow: 0 12px 30px rgba(0,0,0,0.28);
        }

        .metric-label {
            color: #94a3b8;
            font-size: 15px;
            font-weight: 700;
        }

        .metric-value {
            color: white;
            font-size: 38px;
            font-weight: 900;
            margin-top: 8px;
        }

        .metric-note {
            color: #38bdf8;
            font-size: 13px;
            margin-top: 4px;
        }

        .section-card {
            padding: 25px;
            border-radius: 20px;
            background: rgba(15, 23, 42, 0.92);
            border: 1px solid rgba(148, 163, 184, 0.22);
            box-shadow: 0 12px 30px rgba(0,0,0,0.20);
            margin-bottom: 20px;
        }

        .module-title {
            color: #38bdf8;
            font-size: 22px;
            font-weight: 800;
            margin-bottom: 8px;
        }

        .module-text {
            color: #cbd5e1;
            font-size: 15px;
            line-height: 1.7;
        }

        .small-muted {
            color: #94a3b8;
            font-size: 14px;
        }

        .status-pill {
            padding: 7px 12px;
            border-radius: 999px;
            background: rgba(34, 197, 94, 0.12);
            color: #4ade80;
            border: 1px solid rgba(34, 197, 94, 0.35);
            font-weight: 700;
            display: inline-block;
        }

        .danger-pill {
            padding: 7px 12px;
            border-radius: 999px;
            background: rgba(239, 68, 68, 0.12);
            color: #f87171;
            border: 1px solid rgba(239, 68, 68, 0.35);
            font-weight: 700;
            display: inline-block;
        }

        h1, h2, h3 {
            color: white;
        }

        div[data-testid="stDataFrame"] {
            border-radius: 16px;
            overflow: hidden;
        }

        .footer {
            text-align: center;
            padding: 25px;
            color: #94a3b8;
            font-size: 14px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------
# DEMO DATA
# -------------------------------------------------

records = [
    {
        "Violation ID": "V001",
        "Violation Type": "Wrong-Side Driving",
        "Location": "Flyover Road",
        "Status": "Violation Detected",
        "Confidence": "0.87",
        "Evidence": "Generated",
        "Timestamp": "2026-06-21 06:06:39"
    },
    {
        "Violation ID": "V002",
        "Violation Type": "No Helmet",
        "Location": "Traffic Road",
        "Status": "Violation Detected",
        "Confidence": "0.82",
        "Evidence": "Generated",
        "Timestamp": "2026-06-21 06:15:00"
    },
    {
        "Violation ID": "V003",
        "Violation Type": "Triple Riding",
        "Location": "Urban Road",
        "Status": "Monitoring",
        "Confidence": "0.78",
        "Evidence": "Prototype",
        "Timestamp": "2026-06-21 06:20:00"
    },
    {
        "Violation ID": "V004",
        "Violation Type": "Waterlogging Risk",
        "Location": "Rain-Affected Road",
        "Status": "Monitoring",
        "Confidence": "0.74",
        "Evidence": "Prototype",
        "Timestamp": "2026-06-21 06:25:00"
    }
]

df = pd.DataFrame(records)

violation_summary = pd.DataFrame(
    {
        "Violation Type": [
            "Wrong-Side Driving",
            "No Helmet",
            "Triple Riding",
            "Waterlogging Risk"
        ],
        "Count": [1, 1, 1, 1]
    }
)

timeline_data = pd.DataFrame(
    {
        "Time": ["06:00", "06:05", "06:10", "06:15", "06:20", "06:25"],
        "Detected Cases": [0, 1, 1, 2, 3, 4]
    }
)

# -------------------------------------------------
# HERO SECTION
# -------------------------------------------------

st.markdown(
    """
    <div class="hero-card">
        <div class="badge">● Live Demo Mode Active</div>
        <div class="hero-title">🚦 IntelliTraffic AI Dashboard</div>
        <div class="hero-subtitle">
            Smart traffic violation monitoring system using Computer Vision, YOLO-based detection,
            evidence generation, and API-style analytics dashboard for safer roads.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------
# METRIC CARDS
# -------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">Total Cases</div>
            <div class="metric-value">04</div>
            <div class="metric-note">Detected / monitored records</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">Active Violations</div>
            <div class="metric-value">02</div>
            <div class="metric-note">Evidence generated</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">AI Confidence</div>
            <div class="metric-value">87%</div>
            <div class="metric-note">Best detection confidence</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col4:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">System Status</div>
            <div class="metric-value">ON</div>
            <div class="metric-note">Prototype dashboard ready</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# -------------------------------------------------
# TABS
# -------------------------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📊 Overview",
        "🚨 Violation Records",
        "🖼 Evidence",
        "⚙️ Architecture",
        "🚀 Run Project"
    ]
)

# -------------------------------------------------
# TAB 1: OVERVIEW
# -------------------------------------------------

with tab1:
    left, right = st.columns([1.2, 1])

    with left:
        st.markdown(
            """
            <div class="section-card">
                <div class="module-title">Smart AI Monitoring</div>
                <div class="module-text">
                    IntelliTraffic analyzes road and flyover footage to detect traffic violations.
                    The system tracks vehicle movement, identifies violation patterns, and generates
                    evidence frames for review.
                    <br><br>
                    <span class="status-pill">API-style dashboard active</span>
                    &nbsp;
                    <span class="danger-pill">Violation evidence ready</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("### Detection Modules")

        module_col1, module_col2 = st.columns(2)

        with module_col1:
            st.markdown(
                """
                <div class="section-card">
                    <div class="module-title">Wrong-Side Detection</div>
                    <div class="module-text">
                        Tracks vehicles inside a flyover monitoring zone and compares their movement
                        direction with the valid road direction.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with module_col2:
            st.markdown(
                """
                <div class="section-card">
                    <div class="module-title">Helmet Violation</div>
                    <div class="module-text">
                        Detects motorcycles and riders, then marks possible no-helmet violations
                        with timestamp-based evidence.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    with right:
        st.markdown("### Violation Distribution")
        st.bar_chart(
            violation_summary.set_index("Violation Type"),
            use_container_width=True
        )

        st.markdown("### Detection Timeline")
        st.line_chart(
            timeline_data.set_index("Time"),
            use_container_width=True
        )

# -------------------------------------------------
# TAB 2: RECORDS
# -------------------------------------------------

with tab2:
    st.markdown("### Live Violation Records")

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### Quick Analytics")

    a, b, c = st.columns(3)

    with a:
        st.info("Most Critical: Wrong-Side Driving")
    with b:
        st.warning("Evidence generated for 2 records")
    with c:
        st.success("Dashboard ready for reviewers")

# -------------------------------------------------
# TAB 3: EVIDENCE
# -------------------------------------------------

with tab3:
    st.markdown("### Evidence Gallery")

    evidence_folder = Path("outputs/evidence")

    if evidence_folder.exists():
        images = (
            list(evidence_folder.glob("*.png"))
            + list(evidence_folder.glob("*.jpg"))
            + list(evidence_folder.glob("*.jpeg"))
        )

        if images:
            cols = st.columns(2)

            for index, image_path in enumerate(images[:6]):
                with cols[index % 2]:
                    st.image(
                        str(image_path),
                        caption=f"Evidence: {image_path.name}",
                        use_container_width=True
                    )
        else:
            st.markdown(
                """
                <div class="section-card">
                    <div class="module-title">Evidence Folder Found</div>
                    <div class="module-text">
                        Add your final output screenshots inside <b>outputs/evidence</b>
                        to display them here on the deployed dashboard.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.markdown(
            """
            <div class="section-card">
                <div class="module-title">Evidence Preview</div>
                <div class="module-text">
                    Evidence screenshots will appear here after adding images inside:
                    <br><br>
                    <b>traffic-violation-ai/outputs/evidence/</b>
                    <br><br>
                    Recommended images:
                    <br>
                    • wrong_side_detection.png<br>
                    • helmet_detection.png<br>
                    • terminal_report.png<br>
                    • dashboard_output.png
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# -------------------------------------------------
# TAB 4: ARCHITECTURE
# -------------------------------------------------

with tab4:
    st.markdown("### System Architecture")

    st.code(
        """
Traffic CCTV / Road Video
        ↓
Frame Extraction using OpenCV
        ↓
YOLOv8 Object Detection
        ↓
Vehicle / Rider Identification
        ↓
Violation Logic Engine
        ├── Wrong-Side Direction Tracking
        ├── Helmet Violation Detection
        ├── Triple Riding Prototype
        └── Waterlogging Prototype
        ↓
Evidence Frame Generation
        ↓
API Records + Analytics
        ↓
Streamlit Dashboard
        """,
        language="text"
    )

    st.markdown("### Tech Stack")

    t1, t2, t3 = st.columns(3)

    with t1:
        st.markdown(
            """
            <div class="section-card">
                <div class="module-title">AI / Vision</div>
                <div class="module-text">
                    YOLOv8<br>
                    OpenCV<br>
                    Object Tracking<br>
                    Direction Logic
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with t2:
        st.markdown(
            """
            <div class="section-card">
                <div class="module-title">Backend</div>
                <div class="module-text">
                    Python<br>
                    FastAPI<br>
                    Violation Records<br>
                    Evidence Metadata
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with t3:
        st.markdown(
            """
            <div class="section-card">
                <div class="module-title">Dashboard</div>
                <div class="module-text">
                    Streamlit<br>
                    Analytics<br>
                    Evidence Gallery<br>
                    GitHub Deployment
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# -------------------------------------------------
# TAB 5: RUN PROJECT
# -------------------------------------------------

with tab5:
    st.markdown("### Local Run Instructions")

    st.code(
        """
git clone https://github.com/Mrigank-loop/Get-Rid-of-Traffic-vioaltions.git

cd Get-Rid-of-Traffic-vioaltions/traffic-violation-ai

pip install -r requirements.txt

python wrong_side_detector.py

python helmet_detector.py

python -m uvicorn backend_api:app --reload --port 8000

python -m streamlit run api_dashboard.py
        """,
        language="bash"
    )

    st.markdown("### Repository")

    st.code(
        "https://github.com/Mrigank-loop/Get-Rid-of-Traffic-vioaltions",
        language="text"
    )

# -------------------------------------------------
# FOOTER
# -------------------------------------------------

st.markdown(
    """
    <div class="footer">
        IntelliTraffic AI • Smart Traffic Violation Detection Prototype • Built for road safety innovation
    </div>
    """,
    unsafe_allow_html=True
)