import cv2
import os
import csv
import math
import numpy as np
from datetime import datetime
from ultralytics import YOLO


# =========================
# CONFIG
# =========================

VIDEO_CANDIDATES = [
    "videos/wrong_direction.mp4.mp4",
    "videos/wrong_direction.mp4",
    "videos/day_traffic.mp4.mp4",
    "videos/day_traffic.mp4",
    "videos/hq_plate.mp4",
]

OUTPUT_VIDEO = "outputs/wrong_side_processed.mp4"
OUTPUT_CSV = "data/wrong_side_report.csv"

VEHICLE_CLASSES = ["car", "motorcycle", "bus", "truck"]

CONFIDENCE_THRESHOLD = 0.35

# For your screenshot:
# Valid traffic goes straight / away from camera.
# Wrong vehicle comes towards camera.
EXPECTED_VALID_FLOW = "AWAY"

MIN_Y_MOVEMENT = 14
MIN_AREA_GROWTH_RATIO = 0.18
MAX_TRACK_DISTANCE = 140
TRACK_MEMORY_FRAMES = 18

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
    print("Wrong-direction video not found.")
    print("Put your video inside videos folder as wrong_direction.mp4")
    exit()

print(f"Using video: {VIDEO_PATH}")


# =========================
# LOAD MODEL
# =========================

model = YOLO("yolov8n.pt")


# =========================
# HELPER FUNCTIONS
# =========================

def make_polygon(points_ratio, width, height):
    points = []

    for x_ratio, y_ratio in points_ratio:
        x = int(x_ratio * width)
        y = int(y_ratio * height)
        points.append([x, y])

    return np.array(points, dtype=np.int32)


def point_inside_polygon(point, polygon):
    return cv2.pointPolygonTest(polygon, point, False) >= 0


def point_inside_any_zone(point, zones):
    for zone_name, polygon in zones.items():
        if point_inside_polygon(point, polygon):
            return True, zone_name

    return False, "OUTSIDE_ROAD_ZONE"


def bottom_center(x1, y1, x2, y2):
    return (x1 + x2) // 2, y2


def box_area(x1, y1, x2, y2):
    return max(1, (x2 - x1) * (y2 - y1))


def distance(p1, p2):
    return math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2
    )


def assign_track(point, tracks, next_track_id):
    best_track_id = None
    best_distance = 10**9

    for track_id, track in tracks.items():
        last_point = track["points"][-1]
        d = distance(point, last_point)

        if d < best_distance:
            best_distance = d
            best_track_id = track_id

    if best_track_id is not None and best_distance <= MAX_TRACK_DISTANCE:
        return best_track_id, next_track_id

    new_track_id = next_track_id
    next_track_id += 1

    return new_track_id, next_track_id


def direction_decision(track):
    points = track["points"]
    areas = track["areas"]

    if len(points) < 4:
        return "TRACKING_DIRECTION", 0, 0

    old_x, old_y = points[0]
    new_x, new_y = points[-1]

    old_area = areas[0]
    new_area = areas[-1]

    movement_y = new_y - old_y
    area_growth_ratio = (new_area - old_area) / max(old_area, 1)

    # In image coordinates:
    # y increases = object moves down / towards camera
    # y decreases = object moves up / away from camera

    if EXPECTED_VALID_FLOW == "AWAY":
        if movement_y > MIN_Y_MOVEMENT or area_growth_ratio > MIN_AREA_GROWTH_RATIO:
            return "WRONG_DIRECTION_TOWARDS_CAMERA", movement_y, area_growth_ratio

        if movement_y < -MIN_Y_MOVEMENT or area_growth_ratio < -MIN_AREA_GROWTH_RATIO:
            return "VALID_DIRECTION_AWAY", movement_y, area_growth_ratio

    return "TRACKING_DIRECTION", movement_y, area_growth_ratio


def get_severity(vehicle_type):
    if vehicle_type in ["bus", "truck"]:
        return 5, "HIGH"

    if vehicle_type == "car":
        return 4, "HIGH"

    if vehicle_type == "motorcycle":
        return 3, "MEDIUM"

    return 2, "LOW"


def get_action(status):
    if status == "WRONG_DIRECTION_TOWARDS_CAMERA":
        return "Flag opposite-direction vehicle for police review"

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
    (width, height)
)

csv_file = open(OUTPUT_CSV, "w", newline="", encoding="utf-8")
writer = csv.writer(csv_file)

writer.writerow([
    "evidence_id",
    "timestamp",
    "frame_number",
    "track_id",
    "vehicle_type",
    "road_zone",
    "status",
    "movement_y",
    "area_growth_ratio",
    "severity_score",
    "severity_level",
    "confidence",
    "action_recommendation",
    "note"
])


# =========================
# ROAD ZONES BASED ON YOUR SCREENSHOT
# =========================
# These zones cover:
# 1. Flyover straight road
# 2. Road below/beside flyover
# 3. Middle approach area where the wrong car comes from opposite direction

ROAD_ZONES_RATIO = {
    "FLYOVER_STRAIGHT_ROAD": [
        (0.00, 0.30),
        (0.34, 0.24),
        (0.43, 0.52),
        (0.00, 0.55),
    ],

    "BELOW_FLYOVER_STRAIGHT_ROAD": [
        (0.00, 0.50),
        (0.40, 0.47),
        (0.76, 1.00),
        (0.00, 1.00),
    ],

    "CENTER_WRONG_DIRECTION_APPROACH_ZONE": [
        (0.28, 0.43),
        (0.70, 0.42),
        (0.88, 1.00),
        (0.22, 1.00),
    ],
}

ROAD_ZONES = {}

for zone_name, points in ROAD_ZONES_RATIO.items():
    ROAD_ZONES[zone_name] = make_polygon(points, width, height)


# =========================
# TRACKING STORAGE
# =========================

tracks = {}
next_track_id = 1
saved_wrong_tracks = set()
evidence_count = 0
frame_count = 0

print("Processing screenshot-based wrong-direction detector...")


# =========================
# PROCESS VIDEO
# =========================

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1
    display_frame = frame.copy()

    # Draw monitored zones
    overlay = display_frame.copy()

    zone_colors = {
        "FLYOVER_STRAIGHT_ROAD": (0, 255, 0),
        "BELOW_FLYOVER_STRAIGHT_ROAD": (255, 140, 0),
        "CENTER_WRONG_DIRECTION_APPROACH_ZONE": (255, 0, 0),
    }

    for zone_name, polygon in ROAD_ZONES.items():
        color = zone_colors.get(zone_name, (255, 255, 0))

        cv2.fillPoly(
            overlay,
            [polygon],
            color
        )

        cv2.polylines(
            display_frame,
            [polygon],
            True,
            color,
            3
        )

    display_frame = cv2.addWeighted(
        overlay,
        0.12,
        display_frame,
        0.88,
        0
    )

    # Direction arrows
    arrow_x = int(width * 0.73)

    cv2.arrowedLine(
        display_frame,
        (arrow_x, int(height * 0.82)),
        (arrow_x, int(height * 0.45)),
        (0, 255, 0),
        6,
        tipLength=0.18
    )

    cv2.putText(
        display_frame,
        "VALID FLOW: STRAIGHT / AWAY",
        (arrow_x - 240, int(height * 0.42)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (0, 255, 0),
        2
    )

    cv2.arrowedLine(
        display_frame,
        (int(width * 0.47), int(height * 0.46)),
        (int(width * 0.47), int(height * 0.82)),
        (0, 0, 255),
        6,
        tipLength=0.18
    )

    cv2.putText(
        display_frame,
        "WRONG FLOW: TOWARDS CAMERA",
        (int(width * 0.36), int(height * 0.88)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (0, 0, 255),
        2
    )

    results = model(frame, verbose=False)[0]

    current_track_ids = set()
    vehicles_detected = 0
    wrong_count = 0

    for box in results.boxes:
        cls_id = int(box.cls[0])
        vehicle_type = model.names[cls_id]
        confidence = float(box.conf[0])

        if vehicle_type not in VEHICLE_CLASSES:
            continue

        if confidence < CONFIDENCE_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cx, cy = bottom_center(x1, y1, x2, y2)
        area = box_area(x1, y1, x2, y2)

        inside_zone, road_zone = point_inside_any_zone(
            (cx, cy),
            ROAD_ZONES
        )

        if not inside_zone:
            continue

        vehicles_detected += 1

        track_id, next_track_id = assign_track(
            (cx, cy),
            tracks,
            next_track_id
        )

        current_track_ids.add(track_id)

        if track_id not in tracks:
            tracks[track_id] = {
                "points": [],
                "areas": [],
                "missing": 0,
                "vehicle_type": vehicle_type,
                "road_zone": road_zone,
                "confidence": confidence,
            }

        tracks[track_id]["points"].append((cx, cy))
        tracks[track_id]["points"] = tracks[track_id]["points"][-10:]

        tracks[track_id]["areas"].append(area)
        tracks[track_id]["areas"] = tracks[track_id]["areas"][-10:]

        tracks[track_id]["missing"] = 0
        tracks[track_id]["vehicle_type"] = vehicle_type
        tracks[track_id]["road_zone"] = road_zone
        tracks[track_id]["confidence"] = confidence

        status, movement_y, area_growth_ratio = direction_decision(
            tracks[track_id]
        )

        if status == "WRONG_DIRECTION_TOWARDS_CAMERA":
            wrong_count += 1
            severity_score, severity_level = get_severity(vehicle_type)
            action = get_action(status)
            color = (0, 0, 255)
        elif status == "VALID_DIRECTION_AWAY":
            severity_score, severity_level = 1, "LOW"
            action = "Vehicle moving with valid traffic flow"
            color = (0, 255, 0)
        else:
            severity_score, severity_level = 1, "LOW"
            action = "Tracking vehicle movement direction"
            color = (255, 255, 0)

        cv2.rectangle(
            display_frame,
            (x1, y1),
            (x2, y2),
            color,
            3
        )

        cv2.circle(
            display_frame,
            (cx, cy),
            7,
            color,
            -1
        )

        label_text = f"ID {track_id} | {vehicle_type.upper()} | {status}"

        cv2.putText(
            display_frame,
            label_text,
            (x1, max(35, y1 - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            2
        )

        cv2.putText(
            display_frame,
            f"dy={int(movement_y)} area={area_growth_ratio:.2f}",
            (x1, min(height - 20, y2 + 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2
        )

        if status == "WRONG_DIRECTION_TOWARDS_CAMERA":
            evidence_id = f"WRONG_DIR_{track_id:04d}"

            cv2.putText(
                display_frame,
                f"Evidence: {evidence_id}",
                (x1, min(height - 20, y2 + 50)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 255, 255),
                2
            )

            if track_id not in saved_wrong_tracks:
                saved_wrong_tracks.add(track_id)
                evidence_count += 1

                writer.writerow([
                    evidence_id,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    frame_count,
                    track_id,
                    vehicle_type,
                    road_zone,
                    status,
                    round(movement_y, 2),
                    round(area_growth_ratio, 3),
                    severity_score,
                    severity_level,
                    round(confidence, 2),
                    action,
                    "Wrong-direction detection using movement direction and bounding-box growth"
                ])

    # Remove missing old tracks
    for track_id in list(tracks.keys()):
        if track_id not in current_track_ids:
            tracks[track_id]["missing"] += 1

        if tracks[track_id]["missing"] > TRACK_MEMORY_FRAMES:
            del tracks[track_id]

    # Top overlay
    cv2.rectangle(
        display_frame,
        (0, 0),
        (width, 88),
        (8, 8, 8),
        -1
    )

    cv2.putText(
        display_frame,
        "Wrong-Direction Detection | Flyover + Below-Flyover Same Direction Road",
        (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2
    )

    cv2.putText(
        display_frame,
        f"Vehicles: {vehicles_detected} | Wrong Direction: {wrong_count} | Evidence Saved: {evidence_count}",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (0, 255, 255),
        2
    )

    out.write(display_frame)

cap.release()
out.release()
csv_file.close()

print("Wrong-direction detection completed.")
print(f"Output video saved at: {OUTPUT_VIDEO}")
print(f"CSV report saved at: {OUTPUT_CSV}")
print(f"Total wrong-direction evidence records: {evidence_count}")