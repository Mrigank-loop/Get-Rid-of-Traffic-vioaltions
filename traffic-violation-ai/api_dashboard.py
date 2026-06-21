import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="IntelliTraffic AI Dashboard",
    page_icon="🚦",
    layout="wide"
)

st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top left, #1e3a8a 0%, #020617 35%, #0f172a 100%);
        color: white;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .hero {
        padding: 34px;
        border-radius: 26px;
        background: linear-gradient(135deg, rgba(37,99,235,0.95), rgba(15,23,42,0.95));
        border: 1px solid rgba(125,211,252,0.35);
        box-shadow: 0px 20px 50px rgba(0,0,0,0.35);
        margin-bottom: 25px;
    }

    .hero h1 {
        font-size: 46px;
        font-weight: 900;
        margin-bottom: 8px;
        color: white;
    }

    .hero p {
        font-size: 18px;
        color: #dbeafe;
        max-width: 950px;
    }

    .badge {
        display: inline-block;
        padding: 8px 14px;
        border-radius: 999px;
        background: rgba(34,197,94,0.16);
        border: 1px solid rgba(74,222,128,0.45);
        color: #86efac;
        font-weight: 800;
        margin-bottom: 14px;
    }

    .glass-card {
        padding: 22px;
        border-radius: 22px;
        background: rgba(15,23,42,0.78);
        border: 1px solid rgba(148,163,184,0.22);
        box-shadow: 0px 12px 35px rgba(0,0,0,0.25);
        min-height: 135px;
    }

    .card-title {
        color: #94a3b8;
        font-size: 15px;
        font-weight: 800;
    }

    .card-value {
        color: white;
        font-size: 38px;
        font-weight: 900;
        margin-top: 6px;
    }

    .card-note {
        color: #38bdf8;
        font-size: 13px;
        margin-top: 4px;
    }

    .section-box {
        padding: 24px;
        border-radius: 22px;
        background: rgba(15,23,42,0.82);
        border: 1px solid rgba(148,163,184,0.22);
        box-shadow: 0px 12px 35px rgba(0,0,0,0.20);
        margin-bottom: 18px;
    }

    .section-title {
        color: #38bdf8;
        font-size: 22px;
        font-weight: 900;
        margin-bottom: 8px;
    }

    .section-text {
        color: #cbd5e1;
        font-size: 15px;
        line-height: 1.7;
    }

    .pill-green {
        display: inline-block;
        padding: 8px 13px;
        border-radius: 999px;
        color: #86efac;
        background: rgba(34,197,94,0.13);
        border: 1px solid rgba(34,197,94,0.35);
        font-weight: 800;
    }

    .pill-red {
        display: inline-block;
        padding: 8px 13px;
        border-radius: 999px;
        color: #fca5a5;
        background: rgba(239,68,68,0.13);
        border: 1px solid rgba(239,68,68,0.35);
        font-weight: 800;
    }

    h1, h2, h3 {
        color: white;
    }

    .footer {
        text-align: center;
        color: #94a3b8;
        padding: 25px;
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

records = [
    {
        "ID": "V001",
        "Violation": "Wrong-Side Driving",
        "Location": "Flyover Road",
        "Status": "Violation Detected",
        "Confidence": "87%",
        "Evidence": "Generated",
        "Time": "06:06 AM"
    },
    {
        "ID": "V002",
        "Violation": "No Helmet",
        "Location": "Traffic Road",
        "Status": "Violation Detected",
        "Confidence": "82%",
        "Evidence": "Generated",
        "Time": "06:15 AM"
    },
    {
        "ID": "V003",
        "Violation": "Triple Riding",
        "Location": "Urban Road",
        "Status": "Prototype Module",
        "Confidence": "78%",
        "Evidence": "Planned",
        "Time": "06:20 AM"
    },
    {
        "ID": "V004",
        "Violation": "Waterlogging Risk",
        "Location": "Rain-Affected Road",
        "Status": "Prototype Module",
        "Confidence": "74%",
        "Evidence": "Planned",
        "Time": "06:25 AM"
    }
]

df = pd.DataFrame(records)

chart_df = pd.DataFrame(
    {
        "Violation Type": [
            "Wrong-Side Driving",
            "No Helmet",
            "Triple Riding",
            "Waterlogging Risk"
        ],
        "Cases": [1, 1, 1, 1]
    }
)

timeline_df = pd.DataFrame(
    {
        "Time": ["06:00", "06:05", "06:10", "06:15", "06:20", "06:25"],
        "Detected Cases": [0, 1, 1, 2, 3, 4]
    }
)

st.markdown(
    """
    <div class="hero">
        <div class="badge">● Smart Demo Dashboard Active</div>
        <h1>🚦 IntelliTraffic AI Dashboard</h1>
        <p>
        AI-powered traffic violation monitoring prototype using Computer Vision,
        YOLO-based detection, vehicle tracking, evidence generation, and smart analytics.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(
        """
        <div class="glass-card">
            <div class="card-title">Total Records</div>
            <div class="card-value">04</div>
            <div class="card-note">Violation + prototype cases</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with m2:
    st.markdown(
        """
        <div class="glass-card">
            <div class="card-title">Detected Violations</div>
            <div class="card-value">02</div>
            <div class="card-note">Evidence generated</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with m3:
    st.markdown(
        """
        <div class="glass-card">
            <div class="card-title">Best Confidence</div>
            <div class="card-value">87%</div>
            <div class="card-note">Wrong-side detection</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with m4:
    st.markdown(
        """
        <div class="glass-card">
            <div class="card-title">System Status</div>
            <div class="card-value">ON</div>
            <div class="card-note">Demo ready</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📊 Overview",
        "🚨 Violation Records",
        "🖼 Evidence",
        "⚙️ Architecture",
        "🚀 Run Instructions"
    ]
)

with tab1:
    left, right = st.columns([1.15, 1])

    with left:
        st.markdown(
            """
            <div class="section-box">
                <div class="section-title">Smart Traffic Monitoring System</div>
                <div class="section-text">
                IntelliTraffic AI analyzes traffic videos from roads and flyovers.
                It detects vehicles, tracks movement direction, identifies violation patterns,
                and generates evidence frames for faster review.
                <br><br>
                <span class="pill-green">Dashboard Ready</span>
                &nbsp;
                <span class="pill-red">Violation Evidence Generated</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        c1, c2 = st.columns(2)

        with c1:
            st.markdown(
                """
                <div class="section-box">
                    <div class="section-title">Wrong-Side Detection</div>
                    <div class="section-text">
                    Tracks vehicle movement inside a flyover monitoring zone and compares it
                    with the valid traffic direction.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c2:
            st.markdown(
                """
                <div class="section-box">
                    <div class="section-title">Helmet Violation</div>
                    <div class="section-text">
                    Detects motorcycle riders and marks possible no-helmet cases with
                    evidence and timestamp.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    with right:
        st.subheader("Violation Distribution")
        st.bar_chart(chart_df.set_index("Violation Type"), use_container_width=True)

        st.subheader("Detection Timeline")
        st.line_chart(timeline_df.set_index("Time"), use_container_width=True)

with tab2:
    st.subheader("Violation Records")

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    a, b, c = st.columns(3)
    with a:
        st.info("Most critical: Wrong-Side Driving")
    with b:
        st.warning("Evidence generated for 2 records")
    with c:
        st.success("Ready for reviewer demo")

with tab3:
    st.subheader("Evidence Gallery")

    evidence_folder = Path("outputs/evidence")
    images = []

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
            <div class="section-box">
                <div class="section-title">Evidence Preview</div>
                <div class="section-text">
                Evidence frames are generated locally by the detector modules.
                For final presentation, include screenshots of:
                <br><br>
                • Flyover wrong-side output<br>
                • Helmet violation output<br>
                • Terminal violation report<br>
                • Output evidence folder
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

with tab4:
    st.subheader("System Architecture")

    st.code(
        """
Traffic Video / CCTV Footage
        ↓
OpenCV Frame Processing
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
Dashboard Analytics
        """,
        language="text"
    )

    t1, t2, t3 = st.columns(3)

    with t1:
        st.markdown(
            """
            <div class="section-box">
                <div class="section-title">AI / Vision</div>
                <div class="section-text">
                YOLOv8<br>
                OpenCV<br>
                Object Detection<br>
                Direction Logic
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with t2:
        st.markdown(
            """
            <div class="section-box">
                <div class="section-title">Backend Logic</div>
                <div class="section-text">
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
            <div class="section-box">
                <div class="section-title">Dashboard</div>
                <div class="section-text">
                Streamlit<br>
                Analytics<br>
                Evidence Gallery<br>
                Demo Link
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

with tab5:
    st.subheader("Local Run Instructions")

    st.code(
        """
git clone https://github.com/Mrigank-loop/Get-Rid-of-Traffic-vioaltions.git

cd Get-Rid-of-Traffic-vioaltions/traffic-violation-ai

pip install -r requirements.txt

python wrong_side_detector.py

python helmet_detector.py

python -m streamlit run api_dashboard.py
        """,
        language="bash"
    )

    st.subheader("Repository")
    st.code(
        "https://github.com/Mrigank-loop/Get-Rid-of-Traffic-vioaltions",
        language="text"
    )

st.markdown(
    """
    <div class="footer">
    IntelliTraffic AI • Smart Traffic Violation Detection Prototype • Built for safer roads
    </div>
    """,
    unsafe_allow_html=True
)