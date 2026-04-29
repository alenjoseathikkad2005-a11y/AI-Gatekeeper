import cv2
import os
import numpy as np
import time
from pathlib import Path

# ---------------- CONFIG ----------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATASET_DIR = DATA_DIR / "dataset"
MODEL_FILE = DATA_DIR / "trainer.yml"
LABELS_FILE = DATA_DIR / "label_ids.npy"
FACE_SIZE = (200, 200)
SAMPLES_PER_PERSON = 50  # change to 100 or 200 if you want
# ----------------------------------------

# Create dataset folder if not exists
os.makedirs(DATASET_DIR, exist_ok=True)

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
def run_trainer(student_name=None, student_id=None):

    # Create recognizer
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
    except AttributeError:
        print("[ERROR] opencv-contrib-python not installed")
        exit()

    # ---------------- USER INPUT ----------------
    if student_name is None:
        name = input("Enter Student Name: ").strip()
    else:
        name = student_name.strip()

    if student_id is None:
        student_id = input("Enter Student ID: ").strip()
    else:
        student_id = student_id.strip()


    folder_name = f"{name}_{student_id}"
    person_dir = DATASET_DIR / folder_name
    os.makedirs(person_dir, exist_ok=True)

    print(f"[INFO] Images will be saved in: {person_dir}")
    print("[INFO] Press SPACE to capture image")
    print("[INFO] Press Q to quit")
def train_from_frames(face_frames, name, student_id):
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    faces = []
    labels = []
    label_map = {}
    label_id = 0

    folder_name = f"{name}_{student_id}"
    person_dir = DATASET_DIR / folder_name
    os.makedirs(person_dir, exist_ok=True)

    # Save captured frames
    for i, img in enumerate(face_frames):
        cv2.imwrite(str(person_dir / f"{i}.jpg"), img)

    # Rebuild dataset
    for folder in os.listdir(DATASET_DIR):
        folder_path = DATASET_DIR / folder
        if not os.path.isdir(folder_path):
            continue

        label_map[label_id] = folder

        for img_name in os.listdir(folder_path):
            img_path = folder_path / img_name
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img = cv2.resize(img, FACE_SIZE)
            faces.append(img)
            labels.append(label_id)

        label_id += 1

    recognizer.train(faces, np.array(labels))
    recognizer.write(str(MODEL_FILE))
    np.save(str(LABELS_FILE), label_map)

    print("[INFO] Training complete (WEB MODE)")

if __name__ == "__main__":
    run_trainer()
