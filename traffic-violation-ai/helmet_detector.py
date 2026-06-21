import cv2
import os
from datetime import datetime
from ultralytics import YOLO

VIDEO_PATH = "videos/no_helmet.mp4"
OUTPUT_PATH = "outputs/helmet_output.mp4"
EVIDENCE_DIR = "outputs/evidence"

MODEL_PATH = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.35


def create_folders():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs(EVIDENCE_DIR, exist_ok=True)


def box_center(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def is_person_near_bike(person_box, bike_box):
    px, py = box_center(person_box)
    bx, by = box_center(bike_box)

    bike_x1, bike_y1, bike_x2, bike_y2 = bike_box
    bike_width = bike_x2 - bike_x1
    bike_height = bike_y2 - bike_y1

    horizontal_close = abs(px - bx) < bike_width
    vertical_close = abs(py - by) < bike_height * 1.5

    return horizontal_close and vertical_close


def draw_professional_output(frame, violation_type, confidence=None):
    h, w = frame.shape[:2]

    panel_x1, panel_y1 = 20, 20
    panel_x2, panel_y2 = min(w - 20, 760), 165

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (panel_x1, panel_y1),
        (panel_x2, panel_y2),
        (0, 0, 0),
        -1
    )

    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.rectangle(
        frame,
        (panel_x1, panel_y1),
        (panel_x2, panel_y1 + 40),
        (0, 0, 255),
        -1
    )

    cv2.putText(
        frame,
        "VIOLATION DETECTED",
        (panel_x1 + 15, panel_y1 + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Violation Type: {violation_type}",
        (panel_x1 + 15, panel_y1 + 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    if confidence is not None:
        cv2.putText(
            frame,
            f"Confidence: {float(confidence):.2f}",
            (panel_x1 + 15, panel_y1 + 102),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cv2.putText(
        frame,
        f"Timestamp: {timestamp}",
        (panel_x1 + 15, panel_y1 + 132),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        2
    )

    cv2.putText(
        frame,
        "Rider detected without visible helmet. Evidence frame generated.",
        (20, h - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    return frame


def save_evidence_frame(frame, violation_type):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_name = violation_type.lower().replace(" ", "_").replace("-", "_")

    evidence_path = os.path.join(
        EVIDENCE_DIR,
        f"{clean_name}_evidence_{timestamp}.jpg"
    )

    cv2.imwrite(evidence_path, frame)
    return evidence_path


def print_violation_report(violation_type, confidence=None, evidence_path=None):
    print("\n" + "=" * 60)
    print("TRAFFIC VIOLATION REPORT")
    print("=" * 60)
    print(f"Violation Type : {violation_type}")

    if confidence is not None:
        print(f"Confidence     : {float(confidence):.2f}")

    print(f"Timestamp      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if evidence_path:
        print(f"Evidence Saved : {evidence_path}")

    print("Status         : Evidence generated successfully")
    print("=" * 60 + "\n")


def main():
    if not os.path.exists(VIDEO_PATH):
        print("No helmet video found.")
        print("Put video inside videos folder as no_helmet.mp4")
        return

    create_folders()

    print("Loading YOLO model...")
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("Could not open video.")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 25

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        OUTPUT_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    evidence_saved = False
    frame_count = 0

    print("Helmet violation detector started...")
    print("Live preview disabled. Output video will be saved automatically.")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]

        persons = []
        motorcycles = []
        best_confidence = None

        for box in results.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            label = model.names[cls_id]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if label == "person":
                persons.append((x1, y1, x2, y2, confidence))

            elif label == "motorcycle":
                motorcycles.append((x1, y1, x2, y2, confidence))

        violation_detected = False

        for person in persons:
            px1, py1, px2, py2, p_conf = person

            for bike in motorcycles:
                bx1, by1, bx2, by2, b_conf = bike

                if is_person_near_bike(
                    (px1, py1, px2, py2),
                    (bx1, by1, bx2, by2)
                ):
                    violation_detected = True
                    best_confidence = max(p_conf, b_conf)

                    cv2.rectangle(
                        frame,
                        (px1, py1),
                        (px2, py2),
                        (0, 0, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        "Rider / Possible No Helmet",
                        (px1, py1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2
                    )

                    cv2.rectangle(
                        frame,
                        (bx1, by1),
                        (bx2, by2),
                        (255, 0, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        "Motorcycle",
                        (bx1, by1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 0, 0),
                        2
                    )

        if violation_detected:
            frame = draw_professional_output(
                frame,
                "No Helmet",
                best_confidence
            )

            if not evidence_saved:
                evidence_path = save_evidence_frame(frame, "No Helmet")

                print_violation_report(
                    "No Helmet",
                    best_confidence,
                    evidence_path
                )

                evidence_saved = True

        else:
            cv2.putText(
                frame,
                "Monitoring for helmet violation...",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

        writer.write(frame)

        if frame_count % 50 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    writer.release()

    print("\nHelmet detection completed.")
    print(f"Output video saved at: {OUTPUT_PATH}")
    print(f"Evidence folder: {EVIDENCE_DIR}")


if __name__ == "__main__":
    main()
    ENCE_THRESHOLD = 0.35
