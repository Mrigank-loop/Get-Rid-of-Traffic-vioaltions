import cv2
import os
import csv
from datetime import datetime
from ultralytics import YOLO


# =========================
# CONFIG
# =========================

VIDEO_CANDIDATES = [
    "videos/no_helmet.mp4.mp4",
    "videos/no_helmet.mp4",
    "videos/day_traffic.mp4.mp4",
    "videos/day_traffic.mp4",
    "videos/hq_plate.mp4",
]

OUTPUT_VIDEO = "outputs/no_helmet_processed.mp4"
OUTPUT_CSV = "data/helmet_report.csv"

os.makedirs("outputs", exist_ok=True)
os.makedirs("data", exist_ok=True)


# =========================
# SELECT VIDEO
# =========================

VIDEO_PATH = None

for path in VIDEO_CANDIDATES:
    if os.path.exists(path):
        VIDEO_PATH = path
        break

if VIDEO_PATH is None:
    print("No helmet video found.")
    print("Put video inside videos folder as no_helmet.mp4")
    exit()

print(f"Using video: {VIDEO_PATH}")


# =========================
# LOAD MODEL
# =========================

model = YOLO("yolov8n.pt")


# =========================
# HELPER FUNCTIONS
# =========================

def box_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) // 2, (y1 + y2) // 2


def is_person_near_motorcycle(person_box, motorcycle_box):
    px1, py1, px2, py2 = person_box
    mx1, my1, mx2, my2 = motorcycle_box

    person_cx, person_cy = box_center(person_box)

    bike_width = mx2 - mx1
    bike_height = my2 - my1

    expanded_x1 = mx1 - int(bike_width * 0.8)
    expanded_y1 = my1 - int(bike_height * 1.8)
    expanded_x2 = mx2 + int(bike_width * 0.8)
    expanded_y2 = my2 + int(bike_height * 0.8)

    return (
        expanded_x1 <= person_cx <= expanded_x2
        and expanded_y1 <= person_cy <= expanded_y2
    )


def get_severity(rider_count):
    if rider_count >= 3:
        return 5, "HIGH"
    if rider_count == 2:
        return 3, "MEDIUM"
    if rider_count == 1:
        return 2, "LOW"

    return 1, "LOW"


def get_action(risk_status, rider_count):
    if risk_status == "POSSIBLE_HELMET_RISK" and rider_count >= 1:
        return "Verify helmet compliance manually or with custom helmet model"

    return "No action required"


# =========================
# VIDEO SETUP
# =========================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Could not open video.")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (width, height),
)

csv_file = open(OUTPUT_CSV, "w", newline="", encoding="utf-8")
writer = csv.writer(csv_file)

writer.writerow([
    "evidence_id",
    "timestamp",
    "frame_number",
    "motorcycle_count",
    "rider_count",
    "risk_status",
    "severity_score",
    "severity_level",
    "safety_score",
    "confidence",
    "action_recommendation",
    "note"
])


# =========================
# PROCESS VIDEO
# =========================

frame_count = 0
evidence_count = 0

print("Processing helmet compliance prototype...")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1
    display_frame = frame.copy()

    results = model(frame, verbose=False)[0]

    motorcycles = []
    persons = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        conf = float(box.conf[0])

        if conf < 0.35:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if label == "motorcycle":
            motorcycles.append((x1, y1, x2, y2, conf))

        elif label == "person":
            persons.append((x1, y1, x2, y2, conf))

    total_motorcycles = len(motorcycles)

    for motorcycle in motorcycles:
        mx1, my1, mx2, my2, bike_conf = motorcycle
        motorcycle_box = (mx1, my1, mx2, my2)

        nearby_riders = []

        for person in persons:
            px1, py1, px2, py2, person_conf = person
            person_box = (px1, py1, px2, py2)

            if is_person_near_motorcycle(person_box, motorcycle_box):
                nearby_riders.append(person)

        rider_count = len(nearby_riders)

        severity_score, severity_level = get_severity(rider_count)

        if rider_count >= 1:
            risk_status = "POSSIBLE_HELMET_RISK"
            safety_score = max(30, 100 - severity_score * 15)
            confidence = round(min(0.95, bike_conf + 0.10), 2)
            color = (0, 0, 255)
        else:
            risk_status = "NO_RIDER_ASSOCIATED"
            safety_score = 90
            confidence = round(bike_conf, 2)
            color = (0, 255, 0)

        action = get_action(risk_status, rider_count)

        # Draw motorcycle box
        cv2.rectangle(
            display_frame,
            (mx1, my1),
            (mx2, my2),
            color,
            3
        )

        label_text = f"{risk_status} | Riders: {rider_count}"

        cv2.putText(
            display_frame,
            label_text,
            (mx1, max(30, my1 - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2
        )

        # Draw nearby riders
        for rider in nearby_riders:
            px1, py1, px2, py2, person_conf = rider

            cv2.rectangle(
                display_frame,
                (px1, py1),
                (px2, py2),
                (255, 255, 0),
                2
            )

            cv2.putText(
                display_frame,
                "RIDER",
                (px1, max(30, py1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2
            )

        if risk_status == "POSSIBLE_HELMET_RISK":
            evidence_count += 1
            evidence_id = f"HELMET_{evidence_count:04d}"

            writer.writerow([
                evidence_id,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                frame_count,
                total_motorcycles,
                rider_count,
                risk_status,
                severity_score,
                severity_level,
                safety_score,
                confidence,
                action,
                "Prototype helmet-risk detection using motorcycle and rider proximity"
            ])

            cv2.putText(
                display_frame,
                f"Evidence: {evidence_id}",
                (mx1, min(height - 20, my2 + 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

    # Top dashboard overlay
    cv2.rectangle(
        display_frame,
        (0, 0),
        (width, 70),
        (10, 10, 10),
        -1
    )

    cv2.putText(
        display_frame,
        "Helmet Compliance Prototype | YOLOv8 + Rider Association",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        display_frame,
        f"Motorcycles: {total_motorcycles} | Persons: {len(persons)} | Evidence: {evidence_count}",
        (20, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )

    out.write(display_frame)

cap.release()
out.release()
csv_file.close()

print("Helmet detection prototype completed.")
print(f"Output video saved at: {OUTPUT_VIDEO}")
print(f"CSV report saved at: {OUTPUT_CSV}")
print(f"Total helmet-risk evidence records: {evidence_count}")
