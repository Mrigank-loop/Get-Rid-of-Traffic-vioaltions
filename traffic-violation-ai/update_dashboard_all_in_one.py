from pathlib import Path

def patch_file(path, replacements):
    p = Path(path)
    if not p.exists():
        print(f"Skipped: {path} not found")
        return

    text = p.read_text(encoding="utf-8")

    for marker, line in replacements:
        if line.strip() in text:
            continue

        idx = text.find(marker)
        if idx == -1:
            print(f"Marker not found in {path}: {marker}")
            continue

        end = text.find("\n", idx)
        text = text[:end + 1] + line + "\n" + text[end + 1:]

    p.write_text(text, encoding="utf-8")
    print(f"Updated: {path}")


# Main premium dashboard
patch_file(
    "dashboard.py",
    [
        ('REPORT_FILES = {', '    "All-in-One Detection": DATA_DIR / "all_in_one_report.csv",'),
        ('VIDEO_FILES = {', '    "All-in-One Detection": OUTPUT_DIR / "all_in_one_processed.mp4",'),
    ]
)

# Smart feature generator
patch_file(
    "smart_features.py",
    [
        ('REPORT_FILES = {', '    "All-in-One Detection": DATA_DIR / "all_in_one_report.csv",'),
        ('SEVERITY_MAP = {', '    "ALL_IN_ONE": 5,'),
        ('FINE_MAP = {', '    "ALL_IN_ONE": 3500,'),
    ]
)

# Backend API, only if it uses REPORT_FILES
patch_file(
    "backend_api.py",
    [
        ('REPORT_FILES = {', '    "All-in-One Detection": DATA_DIR / "all_in_one_report.csv",'),
    ]
)

# requirements update
req = Path("requirements.txt")
existing = req.read_text(encoding="utf-8") if req.exists() else ""

needed = [
    "ultralytics",
    "opencv-python",
    "pandas",
    "numpy",
    "streamlit",
    "plotly",
    "easyocr",
    "fastapi",
    "uvicorn",
    "pymongo",
    "requests",
]

for item in needed:
    if item not in existing:
        existing += "\n" + item

req.write_text(existing.strip() + "\n", encoding="utf-8")
print("Updated: requirements.txt")
