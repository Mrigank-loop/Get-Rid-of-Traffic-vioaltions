import cv2
import os
import csv
import numpy as np
from datetime import datetime
from ultralytics import YOLO


# =========================
# CONFIG
# =========================

VIDEO_CANDIDATES = [
    "videos/multi_violation.mp4",
    "videos/multi_violation.mp4.mp4",
    "videos/no_helmet.mp4",
    "videos/no_helmet.mp4.mp4",
    "videos/day_traffic.mp4",
    "videos/day_traffic.mp4.mp4",
]

OUTPUT_VIDEO = "outputs/multi_violation_processed.mp4"
OUTPUT_CSV = "data/multi_violation_report.csv"

VEHICLE_CLASSES = ["motorcycle", "car", "bus", "truck"]
CONFIDENCE_THRESHOLD = 0.35

PROCESS_EVERY_N_FRAMES_FOR_CSV = 8

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
    print("Multi-violation video not found.")
    print("Put your video inside videos folder as multi_violation.mp4")
    exit()

print(f"Using video: {VIDEO_PATH}")


# =========================
# LOAD YOLO
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

    expanded_x1 = mx1 - int(bike_width * 0.90)
    expanded_y1 = my1 - int(bike_height * 2.00)
    expanded_x2 = mx2 + int(bike_width * 0.90)
    expanded_y2 = my2 + int(bike_height * 0.90)

    return (
        expanded_x1 <= person_cx <= expanded_x2
        and expanded_y1 <= person_cy <= expanded_y2
    )


def analyze_rain_and_wet_road(frame):
    """
    Prototype rain/wet-road detection using bottom road region.
    It checks road brightness, saturation, reflection-like areas, and water-like coverage.
    """

    height, width = frame.shape[:2]

    road_roi = frame[int(height * 0.52):height, :]

    if road_roi.size == 0:
        return "NORMAL_ROAD", 0.0, 95

    hsv = cv2.cvtColor(road_roi, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(road_roi, cv2.COLOR_BGR2GRAY)

    h, s, v = cv2.split(hsv)

    # Wet road often has reflection patches: low saturation + medium/high brightness
    reflection_mask = cv2.inRange(
        hsv,
        np.array([0, 0, 80]),
        np.array([180, 80, 255])
    )

    # Dark wet road: low brightness + low saturation
    dark_wet_mask = cv2.inRange(
        hsv,
        np.array([0, 0, 20]),
        np.array([180, 90, 110])
    )

    combined_mask = cv2.bitwise_or(reflection_mask, dark_wet_mask)

    wet_pixels = cv2.countNonZero(combined_mask)
    total_pixels = combined_mask.shape[0] * combined_mask.shape[1]

    wet_percentage = (wet_pixels / total_pixels) * 100

    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Road safety score decreases when wet percentage increases
    road_safety_score = max(25, int(100 - wet_percentage * 2.2))

    if wet_percentage >= 22:
        return "RAIN_WET_ROAD_HIGH_RISK", round(wet_percentage, 2), road_safety_score

    if wet_percentage >= 12:
        return "RAIN_WET_ROAD_MEDIUM_RISK", round(wet_percentage, 2), road_safety_score

    if blur_score < 45:
        return "RAIN_OR_BLUR_LOW_VISIBILITY", round(wet_percentage, 2), road_safety_score

    return "NORMAL_ROAD", round(wet_percentage, 2), road_safety_score


def get_triple_riding_status(rider_count):
    if rider_count >= 3:
        return "TRIPLE_RIDING_VIOLATION", 5, "HIGH", 1000

    if rider_count == 2:
        return "DOUBLE_RIDING_MONITORING", 2, "LOW", 0

    if rider_count == 1:
        return "SINGLE_RIDER", 1, "LOW", 0

    return "NO_RIDER_ASSOCIATED", 1, "LOW", 0


def get_helmet_status(rider_count):
    """
    Prototype helmet-risk detection.
    Since YOLOv8n does not have helmet/no-helmet class,
    this marks possible helmet risk when riders are detected on motorcycle.
    """

    if rider_count >= 1:
        return "POSSIBLE_NO_HELMET_RISK", 3, "MEDIUM", 500

    return "NO_HELMET_RISK_NOT_CONFIRMED", 1, "LOW", 0


def get_action(triple_status, helmet_status, road_status):
    actions = []

    if triple_status == "TRIPLE_RIDING_VIOLATION":
        actions.append("Generate triple-riding fine and alert")

    if helmet_status == "POSSIBLE_NO_HELMET_RISK":
        actions.append("Manual helmet verification / no-helmet fine review")

    if "RAIN_WET_ROAD" in road_status or "LOW_VISIBILITY" in road_status:
        actions.append("Warn riders about rain risk and slippery road")

    if not actions:
        actions.append("No immediate action required")

    return " | ".join(actions)


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
    (width, height)
)

csv_file = open(OUTPUT_CSV, "w", newline="", encoding="utf-8")
writer = csv.writer(csv_file)

writer.writerow([
    "evidence_id",
    "timestamp",
    "frame_number",
    "motorcycle_count",
    "rider_count",
    "triple_riding_status",
    "helmet_status",
    "rain_road_status",
    "wet_road_percentage",
    "road_safety_score",
    "severity_score",
    "severity_level",
    "demo_fine_amount",
    "confidence",
    "action_recommendation",
    "note"
])


# =========================
# PROCESS VIDEO
# =========================

frame_count = 0
evidence_count = 0

print("Processing combined detector: triple riding + helmet risk + rain/wet road...")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1
    display_frame = frame.copy()

    rain_status, wet_percentage, road_safety_score = analyze_rain_and_wet_road(frame)

    # Draw rain / road condition area
    road_y = int(height * 0.52)

    if "HIGH" in rain_status:
        rain_color = (0, 0, 255)
    elif "MEDIUM" in rain_status or "LOW_VISIBILITY" in rain_status:
        rain_color = (0, 165, 255)
    else:
        rain_color = (0, 255, 0)

    cv2.rectangle(
        display_frame,
        (0, road_y),
        (width, height),
        rain_color,
        2
    )

    cv2.putText(
        display_frame,
        f"Road Condition: {rain_status} | Wet: {wet_percentage}%",
        (20, road_y + 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        rain_color,
        2
    )

    results = model(frame, verbose=False)[0]

    motorcycles = []
    persons = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        conf = float(box.conf[0])

        if conf < CONFIDENCE_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if label == "motorcycle":
            motorcycles.append((x1, y1, x2, y2, conf))

        elif label == "person":
            persons.append((x1, y1, x2, y2, conf))

    total_motorcycles = len(motorcycles)
    frame_violation_count = 0

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

        triple_status, triple_severity, triple_level, triple_fine = get_triple_riding_status(rider_count)
        helmet_status, helmet_severity, helmet_level, helmet_fine = get_helmet_status(rider_count)

        severity_score = max(triple_severity, helmet_severity)

        if "HIGH" in rain_status:
            severity_score = max(severity_score, 4)
        elif "MEDIUM" in rain_status or "LOW_VISIBILITY" in rain_status:
            severity_score = max(severity_score, 3)

        if severity_score >= 4:
            severity_level = "HIGH"
        elif severity_score == 3:
            severity_level = "MEDIUM"
        else:
            severity_level = "LOW"

        demo_fine = triple_fine + helmet_fine

        if triple_status == "TRIPLE_RIDING_VIOLATION":
            box_color = (0, 0, 255)
            frame_violation_count += 1
        elif helmet_status == "POSSIBLE_NO_HELMET_RISK":
            box_color = (0, 165, 255)
            frame_violation_count += 1
        else:
            box_color = (0, 255, 0)

        # Draw motorcycle box
        cv2.rectangle(
            display_frame,
            (mx1, my1),
            (mx2, my2),
            box_color,
            3
        )

        label_text = f"Riders:{rider_count} | {triple_status}"

        cv2.putText(
            display_frame,
            label_text,
            (mx1, max(35, my1 - 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            box_color,
            2
        )

        cv2.putText(
            display_frame,
            f"{helmet_status} | Fine: Rs {demo_fine}",
            (mx1, min(height - 20, my2 + 26)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            box_color,
            2
        )

        # Draw rider boxes
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
                0.5,
                (255, 255, 0),
                2
            )

        # Write evidence only if some violation/risk exists
        violation_exists = (
            triple_status == "TRIPLE_RIDING_VIOLATION"
            or helmet_status == "POSSIBLE_NO_HELMET_RISK"
            or rain_status != "NORMAL_ROAD"
        )

        if violation_exists and frame_count % PROCESS_EVERY_N_FRAMES_FOR_CSV == 0:
            evidence_count += 1
            evidence_id = f"MULTI_{evidence_count:04d}"

            action = get_action(
                triple_status,
                helmet_status,
                rain_status
            )

            writer.writerow([
                evidence_id,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                frame_count,
                total_motorcycles,
                rider_count,
                triple_status,
                helmet_status,
                rain_status,
                wet_percentage,
                road_safety_score,
                severity_score,
                severity_level,
                demo_fine,
                round(bike_conf, 2),
                action,
                "Combined prototype: triple riding + helmet risk + rain/wet road condition"
            ])

            cv2.putText(
                display_frame,
                f"Evidence: {evidence_id}",
                (mx1, min(height - 20, my2 + 52)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.56,
                (0, 255, 255),
                2
            )

    # Top overlay
    cv2.rectangle(
        display_frame,
        (0, 0),
        (width, 95),
        (8, 8, 8),
        -1
    )

    cv2.putText(
        display_frame,
        "Combined Violation Detection | Triple Riding + Helmet Risk + Rain/Wet Road",
        (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2
    )

    cv2.putText(
        display_frame,
        f"Motorcycles: {total_motorcycles} | Persons: {len(persons)} | Frame Risks: {frame_violation_count} | Evidence: {evidence_count}",
        (20, 63),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (0, 255, 255),
        2
    )

    cv2.putText(
        display_frame,
        f"Road Safety Score: {road_safety_score}/100 | {rain_status}",
        (20, 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        rain_color,
        2
    )

    out.write(display_frame)

cap.release()
out.release()
csv_file.close()

print("Combined detector completed.")
print(f"Output video saved at: {OUTPUT_VIDEO}")
print(f"CSV report saved at: {OUTPUT_CSV}")
print(f"Total evidence records: {evidence_count}")
