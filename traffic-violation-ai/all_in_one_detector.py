import cv2
import os
import csv
import re
import math
import numpy as np
from datetime import datetime
from ultralytics import YOLO
import easyocr


# =========================
# CONFIG
# =========================

VIDEO_CANDIDATES = [
    "videos/multi_violation.mp4",
    "videos/multi_violation.mp4.mp4",
    "videos/wrong_direction.mp4",
    "videos/wrong_direction.mp4.mp4",
    "videos/hq_plate.mp4",
    "videos/day_traffic.mp4",
    "videos/day_traffic.mp4.mp4",
]

OUTPUT_VIDEO = "outputs/all_in_one_processed.mp4"
OUTPUT_CSV = "data/all_in_one_report.csv"
PLATE_EVIDENCE_DIR = "outputs/all_in_one_plate_evidence"

VEHICLE_CLASSES = ["car", "motorcycle", "bus", "truck"]

CONFIDENCE_THRESHOLD = 0.35

CSV_EVERY_N_FRAMES = 8
OCR_EVERY_N_FRAMES = 90
MAX_SECONDS_TO_PROCESS = 35

EXPECTED_VALID_FLOW = "AWAY"
MIN_Y_MOVEMENT = 14
MIN_AREA_GROWTH_RATIO = 0.18
MAX_TRACK_DISTANCE = 140
TRACK_MEMORY_FRAMES = 18

os.makedirs("outputs", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs(PLATE_EVIDENCE_DIR, exist_ok=True)


# =========================
# SELECT VIDEO
# =========================

VIDEO_PATH = None

for path in VIDEO_CANDIDATES:
    if os.path.exists(path):
        VIDEO_PATH = path
        break

if VIDEO_PATH is None:
    print("Video not found.")
    print("Put your video inside videos folder as multi_violation.mp4")
    exit()

print(f"Using video: {VIDEO_PATH}")


# =========================
# LOAD MODELS
# =========================

print("Loading YOLOv8...")
model = YOLO("yolov8n.pt")

print("Loading EasyOCR...")
reader = easyocr.Reader(["en"], gpu=False)


# =========================
# HELPERS
# =========================

def clean_plate_text(text):
    text = str(text).upper()
    text = re.sub(r"[^A-Z0-9]", "", text)
    return text


def box_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) // 2, (y1 + y2) // 2


def bottom_center(x1, y1, x2, y2):
    return (x1 + x2) // 2, y2


def box_area(x1, y1, x2, y2):
    return max(1, (x2 - x1) * (y2 - y1))


def distance(p1, p2):
    return math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2
    )


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
    height, width = frame.shape[:2]

    road_roi = frame[int(height * 0.52):height, :]

    if road_roi.size == 0:
        return "NORMAL_ROAD", 0.0, 95

    hsv = cv2.cvtColor(road_roi, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(road_roi, cv2.COLOR_BGR2GRAY)

    reflection_mask = cv2.inRange(
        hsv,
        np.array([0, 0, 80]),
        np.array([180, 85, 255])
    )

    dark_wet_mask = cv2.inRange(
        hsv,
        np.array([0, 0, 20]),
        np.array([180, 95, 115])
    )

    combined_mask = cv2.bitwise_or(reflection_mask, dark_wet_mask)

    wet_pixels = cv2.countNonZero(combined_mask)
    total_pixels = combined_mask.shape[0] * combined_mask.shape[1]

    wet_percentage = (wet_pixels / total_pixels) * 100

    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    road_safety_score = max(25, int(100 - wet_percentage * 2.2))

    if wet_percentage >= 22:
        return "RAIN_WET_ROAD_HIGH_RISK", round(wet_percentage, 2), road_safety_score

    if wet_percentage >= 12:
        return "RAIN_WET_ROAD_MEDIUM_RISK", round(wet_percentage, 2), road_safety_score

    if blur_score < 45:
        return "RAIN_OR_BLUR_LOW_VISIBILITY", round(wet_percentage, 2), road_safety_score

    return "NORMAL_ROAD", round(wet_percentage, 2), road_safety_score


def get_triple_status(rider_count):
    if rider_count >= 3:
        return "TRIPLE_RIDING_VIOLATION", 5, "HIGH", 1000

    if rider_count == 2:
        return "DOUBLE_RIDING_MONITORING", 2, "LOW", 0

    if rider_count == 1:
        return "SINGLE_RIDER", 1, "LOW", 0

    return "NO_RIDER_ASSOCIATED", 1, "LOW", 0


def get_helmet_status(rider_count):
    if rider_count >= 1:
        return "POSSIBLE_NO_HELMET_RISK", 3, "MEDIUM", 500

    return "NO_HELMET_RISK_NOT_CONFIRMED", 1, "LOW", 0


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

    if EXPECTED_VALID_FLOW == "AWAY":
        if movement_y > MIN_Y_MOVEMENT or area_growth_ratio > MIN_AREA_GROWTH_RATIO:
            return "WRONG_DIRECTION_TOWARDS_CAMERA", movement_y, area_growth_ratio

        if movement_y < -MIN_Y_MOVEMENT or area_growth_ratio < -MIN_AREA_GROWTH_RATIO:
            return "VALID_DIRECTION_AWAY", movement_y, area_growth_ratio

    return "TRACKING_DIRECTION", movement_y, area_growth_ratio


def read_plate_from_vehicle(vehicle_crop):
    if vehicle_crop is None or vehicle_crop.size == 0:
        return "", 0, None

    h, w = vehicle_crop.shape[:2]

    if h < 40 or w < 50:
        return "", 0, None

    px1 = int(w * 0.15)
    px2 = int(w * 0.85)
    py1 = int(h * 0.50)
    py2 = int(h * 0.92)

    plate_crop = vehicle_crop[py1:py2, px1:px2]

    if plate_crop.size == 0:
        return "", 0, None

    zoomed = cv2.resize(
        plate_crop,
        None,
        fx=4,
        fy=4,
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.cvtColor(zoomed, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=3.0,
        tileGridSize=(8, 8)
    )
    gray = clahe.apply(gray)

    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    best_text = ""
    best_conf = 0

    for img in [zoomed, gray, thresh]:
        try:
            results = reader.readtext(
                img,
                detail=1,
                paragraph=False,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            )

            for result in results:
                text = clean_plate_text(result[1])
                conf = float(result[2])

                if len(text) >= 5 and conf > best_conf:
                    best_text = text
                    best_conf = conf

        except Exception:
            continue

    return best_text, round(best_conf, 2), zoomed


def get_severity(vehicle_type):
    if vehicle_type in ["bus", "truck"]:
        return 5, "HIGH"

    if vehicle_type == "car":
        return 4, "HIGH"

    if vehicle_type == "motorcycle":
        return 3, "MEDIUM"

    return 2, "LOW"


def write_row(writer, evidence_id, frame_count, event_type, vehicle_type, track_id,
              rider_count, plate_text, triple_status, helmet_status,
              wrong_status, rain_status, wet_percentage, road_safety_score,
              severity_score, severity_level, fine, confidence, action, note):

    writer.writerow([
        evidence_id,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        frame_count,
        event_type,
        vehicle_type,
        track_id,
        rider_count,
        plate_text,
        triple_status,
        helmet_status,
        wrong_status,
        rain_status,
        wet_percentage,
        road_safety_score,
        severity_score,
        severity_level,
        fine,
        confidence,
        action,
        note
    ])


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
max_frames_to_process = fps * MAX_SECONDS_TO_PROCESS

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
    "event_type",
    "vehicle_type",
    "track_id",
    "rider_count",
    "plate_text",
    "triple_riding_status",
    "helmet_status",
    "wrong_direction_status",
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
# ROAD ZONES
# =========================

ROAD_ZONES_RATIO = {
    "FLYOVER_OR_MAIN_ROAD": [
        (0.00, 0.30),
        (1.00, 0.30),
        (1.00, 1.00),
        (0.00, 1.00),
    ],
}

ROAD_ZONES = {}

for zone_name, points in ROAD_ZONES_RATIO.items():
    ROAD_ZONES[zone_name] = make_polygon(points, width, height)


# =========================
# PROCESS VIDEO
# =========================

tracks = {}
next_track_id = 1
saved_wrong_tracks = set()
saved_plate_texts = set()

frame_count = 0
evidence_count = 0

print("Processing ALL-IN-ONE detector...")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    if frame_count > max_frames_to_process:
        break

    display_frame = frame.copy()

    rain_status, wet_percentage, road_safety_score = analyze_rain_and_wet_road(frame)

    if "HIGH" in rain_status:
        rain_color = (0, 0, 255)
    elif "MEDIUM" in rain_status or "LOW_VISIBILITY" in rain_status:
        rain_color = (0, 165, 255)
    else:
        rain_color = (0, 255, 0)

    # Draw road zone
    road_y = int(height * 0.52)

    cv2.rectangle(
        display_frame,
        (0, road_y),
        (width, height),
        rain_color,
        2
    )

    cv2.putText(
        display_frame,
        f"Rain/Road: {rain_status} | Wet: {wet_percentage}%",
        (20, road_y + 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        rain_color,
        2
    )

    # Direction arrows
    cv2.arrowedLine(
        display_frame,
        (int(width * 0.78), int(height * 0.82)),
        (int(width * 0.78), int(height * 0.45)),
        (0, 255, 0),
        5,
        tipLength=0.18
    )

    cv2.putText(
        display_frame,
        "VALID FLOW: AWAY",
        (int(width * 0.62), int(height * 0.42)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2
    )

    cv2.arrowedLine(
        display_frame,
        (int(width * 0.48), int(height * 0.46)),
        (int(width * 0.48), int(height * 0.82)),
        (0, 0, 255),
        5,
        tipLength=0.18
    )

    cv2.putText(
        display_frame,
        "WRONG FLOW: TOWARDS CAMERA",
        (int(width * 0.32), int(height * 0.88)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (0, 0, 255),
        2
    )

    results = model(frame, verbose=False)[0]

    motorcycles = []
    persons = []
    vehicles = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        conf = float(box.conf[0])

        if conf < CONFIDENCE_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if label == "person":
            persons.append((x1, y1, x2, y2, conf))

        if label == "motorcycle":
            motorcycles.append((x1, y1, x2, y2, conf))

        if label in VEHICLE_CLASSES:
            vehicles.append((x1, y1, x2, y2, label, conf))

    current_track_ids = set()
    wrong_count = 0
    triple_count = 0
    helmet_risk_count = 0
    plate_count = 0

    # =========================
    # VEHICLE TRACKING + WRONG DIRECTION + OCR
    # =========================

    for vehicle in vehicles:
        x1, y1, x2, y2, vehicle_type, confidence = vehicle

        cx, cy = bottom_center(x1, y1, x2, y2)
        area = box_area(x1, y1, x2, y2)

        inside_zone, road_zone = point_inside_any_zone((cx, cy), ROAD_ZONES)

        if not inside_zone:
            continue

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
            }

        tracks[track_id]["points"].append((cx, cy))
        tracks[track_id]["points"] = tracks[track_id]["points"][-10:]

        tracks[track_id]["areas"].append(area)
        tracks[track_id]["areas"] = tracks[track_id]["areas"][-10:]

        tracks[track_id]["missing"] = 0

        wrong_status, movement_y, area_growth_ratio = direction_decision(tracks[track_id])

        if wrong_status == "WRONG_DIRECTION_TOWARDS_CAMERA":
            wrong_count += 1
            severity_score, severity_level = get_severity(vehicle_type)
            color = (0, 0, 255)
        elif wrong_status == "VALID_DIRECTION_AWAY":
            severity_score, severity_level = 1, "LOW"
            color = (0, 255, 0)
        else:
            severity_score, severity_level = 1, "LOW"
            color = (255, 255, 0)

        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 3)
        cv2.circle(display_frame, (cx, cy), 6, color, -1)

        cv2.putText(
            display_frame,
            f"ID {track_id} | {vehicle_type.upper()} | {wrong_status}",
            (x1, max(35, y1 - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2
        )

        # Save wrong direction evidence once per track
        if wrong_status == "WRONG_DIRECTION_TOWARDS_CAMERA" and track_id not in saved_wrong_tracks:
            saved_wrong_tracks.add(track_id)
            evidence_count += 1
            evidence_id = f"ALL_WRONG_{evidence_count:04d}"

            write_row(
                writer,
                evidence_id,
                frame_count,
                "WRONG_DIRECTION",
                vehicle_type,
                track_id,
                "",
                "",
                "",
                "",
                wrong_status,
                rain_status,
                wet_percentage,
                road_safety_score,
                severity_score,
                severity_level,
                2000,
                round(confidence, 2),
                "Flag wrong-direction vehicle for police review",
                "All-in-one detector: wrong direction using movement and box growth"
            )

        # OCR only every few frames
        if frame_count % OCR_EVERY_N_FRAMES == 0:
            vehicle_crop = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
            plate_text, plate_conf, plate_crop = read_plate_from_vehicle(vehicle_crop)

            if plate_text:
                plate_count += 1

                if plate_text not in saved_plate_texts:
                    saved_plate_texts.add(plate_text)

                    evidence_count += 1
                    evidence_id = f"ALL_PLATE_{evidence_count:04d}"

                    plate_path = os.path.join(
                        PLATE_EVIDENCE_DIR,
                        f"{evidence_id}_{plate_text}.jpg"
                    )

                    if plate_crop is not None:
                        cv2.imwrite(plate_path, plate_crop)

                    write_row(
                        writer,
                        evidence_id,
                        frame_count,
                        "LICENSE_PLATE_OCR",
                        vehicle_type,
                        track_id,
                        "",
                        plate_text,
                        "",
                        "",
                        wrong_status,
                        rain_status,
                        wet_percentage,
                        road_safety_score,
                        1,
                        "LOW",
                        0,
                        plate_conf,
                        "Plate recorded for vehicle identity evidence",
                        "All-in-one detector: OCR plate evidence"
                    )

                cv2.putText(
                    display_frame,
                    f"PLATE: {plate_text}",
                    (x1, min(height - 20, y2 + 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2
                )

    # Remove old tracks
    for track_id in list(tracks.keys()):
        if track_id not in current_track_ids:
            tracks[track_id]["missing"] += 1

        if tracks[track_id]["missing"] > TRACK_MEMORY_FRAMES:
            del tracks[track_id]

    # =========================
    # MOTORCYCLE: TRIPLE + HELMET RISK
    # =========================

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

        triple_status, triple_severity, triple_level, triple_fine = get_triple_status(rider_count)
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

        violation_exists = (
            triple_status == "TRIPLE_RIDING_VIOLATION"
            or helmet_status == "POSSIBLE_NO_HELMET_RISK"
            or rain_status != "NORMAL_ROAD"
        )

        if triple_status == "TRIPLE_RIDING_VIOLATION":
            box_color = (0, 0, 255)
            triple_count += 1
        elif helmet_status == "POSSIBLE_NO_HELMET_RISK":
            box_color = (0, 165, 255)
            helmet_risk_count += 1
        else:
            box_color = (0, 255, 0)

        cv2.rectangle(display_frame, (mx1, my1), (mx2, my2), box_color, 3)

        cv2.putText(
            display_frame,
            f"Riders:{rider_count} | {triple_status}",
            (mx1, max(35, my1 - 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            box_color,
            2
        )

        cv2.putText(
            display_frame,
            f"{helmet_status} | Fine: Rs {demo_fine}",
            (mx1, min(height - 20, my2 + 26)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.53,
            box_color,
            2
        )

        for rider in nearby_riders:
            px1, py1, px2, py2, person_conf = rider
            cv2.rectangle(display_frame, (px1, py1), (px2, py2), (255, 255, 0), 2)
            cv2.putText(
                display_frame,
                "RIDER",
                (px1, max(30, py1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 0),
                2
            )

        if violation_exists and frame_count % CSV_EVERY_N_FRAMES == 0:
            evidence_count += 1
            evidence_id = f"ALL_MULTI_{evidence_count:04d}"

            actions = []

            if triple_status == "TRIPLE_RIDING_VIOLATION":
                actions.append("Generate triple-riding fine")

            if helmet_status == "POSSIBLE_NO_HELMET_RISK":
                actions.append("Manual no-helmet verification")

            if rain_status != "NORMAL_ROAD":
                actions.append("Rain/wet-road safety warning")

            action = " | ".join(actions) if actions else "No immediate action"

            write_row(
                writer,
                evidence_id,
                frame_count,
                "TRIPLE_HELMET_RAIN",
                "motorcycle",
                "",
                rider_count,
                "",
                triple_status,
                helmet_status,
                "",
                rain_status,
                wet_percentage,
                road_safety_score,
                severity_score,
                severity_level,
                demo_fine,
                round(bike_conf, 2),
                action,
                "All-in-one detector: triple riding + helmet risk + rain/wet road"
            )

            cv2.putText(
                display_frame,
                f"Evidence: {evidence_id}",
                (mx1, min(height - 20, my2 + 52)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.53,
                (0, 255, 255),
                2
            )

    # Rain evidence row occasionally
    if rain_status != "NORMAL_ROAD" and frame_count % 30 == 0:
        evidence_count += 1
        evidence_id = f"ALL_RAIN_{evidence_count:04d}"

        write_row(
            writer,
            evidence_id,
            frame_count,
            "RAIN_WET_ROAD",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            rain_status,
            wet_percentage,
            road_safety_score,
            3,
            "MEDIUM",
            0,
            0.85,
            "Alert road safety team / warn riders",
            "All-in-one detector: rain and wet-road risk"
        )

    # Top overlay
    cv2.rectangle(display_frame, (0, 0), (width, 105), (8, 8, 8), -1)

    cv2.putText(
        display_frame,
        "ALL-IN-ONE TRAFFIC AI | Triple + Helmet + Rain + Wrong Direction + OCR",
        (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (255, 255, 255),
        2
    )

    cv2.putText(
        display_frame,
        f"Vehicles:{len(vehicles)} | Bikes:{len(motorcycles)} | Persons:{len(persons)} | Wrong:{wrong_count} | Triple:{triple_count} | HelmetRisk:{helmet_risk_count}",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2
    )

    cv2.putText(
        display_frame,
        f"Road Safety:{road_safety_score}/100 | {rain_status} | OCR Plates Seen:{len(saved_plate_texts)} | Evidence:{evidence_count}",
        (20, 94),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        rain_color,
        2
    )

    out.write(display_frame)

cap.release()
out.release()
csv_file.close()

print("ALL-IN-ONE detector completed.")
print(f"Output video saved at: {OUTPUT_VIDEO}")
print(f"CSV report saved at: {OUTPUT_CSV}")
print(f"Plate evidence saved at: {PLATE_EVIDENCE_DIR}")
print(f"Total evidence records: {evidence_count}")
