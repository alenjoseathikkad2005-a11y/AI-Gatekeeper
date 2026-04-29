import cv2
import os
import time
import numpy as np
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
FRONTEND_DIR = ROOT_DIR / "frontend"
DATA_DIR = BASE_DIR / "data"

# ================== CAPTURE CONFIG ==================
CAPTURE_DIR = FRONTEND_DIR / "static" / "captures"
os.makedirs(CAPTURE_DIR, exist_ok=True)

last_seen = {}   # prevent saving every frame

# ================== MODEL CONFIG ==================
MODEL_FILE = DATA_DIR / "trainer.yml"
LABELS_FILE = DATA_DIR / "label_ids.npy"

FACE_SIZE = (200, 200)
CONFIDENCE_THRESHOLD = 55   # lower = stricter

# ================== LOAD MODEL ==================
if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(f"[ERROR] {MODEL_FILE} not found. Run trainer.py first.")

if not os.path.exists(LABELS_FILE):
    raise FileNotFoundError(f"[ERROR] {LABELS_FILE} not found. Run trainer.py first.")

try:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
except AttributeError:
    raise RuntimeError(
        "cv2.face.LBPHFaceRecognizer_create not found.\n"
        "Install: pip install opencv-contrib-python"
    )

recognizer.read(str(MODEL_FILE))

# label_id -> student_id mapping
label_id_to_student_id = np.load(str(LABELS_FILE), allow_pickle=True).item()

print("[INFO] Face recognizer loaded")
print("[INFO] Label -> Student ID map:", label_id_to_student_id)

# ================== FACE CASCADE ==================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ================== MAIN RECOGNITION FUNCTION ==================
def recognize_frame(frame, present_students, seen_counter):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=6,
        minSize=(90, 90)
    )

    recognized_id = None
    saved_img_path = None

    # decay seen counter
    for sid in list(seen_counter.keys()):
        seen_counter[sid] = max(seen_counter[sid] - 1, 0)
        if seen_counter[sid] == 0:
            del seen_counter[sid]

    # ================== FACE LOOP ==================
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, FACE_SIZE)

        face_color = frame[y:y+h, x:x+w]

        try:
            label_id, confidence = recognizer.predict(roi_gray)
        except cv2.error:
            continue

        color = (0, 0, 255)
        text = "Guest"

        # Recognized face
        if label_id in label_id_to_student_id and confidence < CONFIDENCE_THRESHOLD:
            student_id = label_id_to_student_id[label_id]

            seen_counter[student_id] = seen_counter.get(student_id, 0) + 2

            if seen_counter[student_id] >= 8:
                recognized_id = student_id
                color = (0, 255, 0)
                text = f"ID {student_id} ({int(confidence)})"

                now = time.time()

                # capture safety (once per 5 seconds)
                if student_id not in last_seen or now - last_seen[student_id] > 5:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    img_name = f"{student_id}_{timestamp}.jpg"
                    img_path = CAPTURE_DIR / img_name

                    cv2.imwrite(str(img_path), face_color)
                    last_seen[student_id] = now
                    present_students.add(student_id)

                    saved_img_path = f"/static/captures/{img_name}"

        # draw face box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            frame,
            text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    # Return ONLY recognition result
    return frame, recognized_id, saved_img_path
