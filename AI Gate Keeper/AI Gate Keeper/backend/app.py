from flask import (Flask, render_template, Response, jsonify, request, session, redirect, url_for)
import cv2
import os
import csv
import time
import re
import shutil
from pathlib import Path
from datetime import datetime
from threading import Thread
from collections import OrderedDict

try:
    from . import trainer
    from .recognizer import recognize_frame
    from .compliance import compliance_detector
except ImportError:
    import trainer
    from recognizer import recognize_frame
    from compliance import compliance_detector

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
FRONTEND_DIR = ROOT_DIR / "frontend"
DATA_DIR = BASE_DIR / "data"

DATASET_DIR = DATA_DIR / "dataset"
STUDENTS_IMG_DIR = FRONTEND_DIR / "static" / "students"
STUDENTS_IMG_URL_PREFIX = "/static/students"
os.makedirs(STUDENTS_IMG_DIR, exist_ok=True)
# ================= HELPERS =================
def normalize_student_id(recognized_id):
    if not recognized_id:
        return None
    return str(recognized_id).split("_")[-1]
def student_id_exists_in_csv(student_id):
    for row in read_students_csv():
        if row.get("ID") == str(student_id):
            return True
    return False



# ================= APP =================
app = Flask(
    __name__,
    template_folder=str(FRONTEND_DIR / "templates"),
    static_folder=str(FRONTEND_DIR / "static"),
    static_url_path="/static",
)
app.secret_key = "super_secret_key_123"

# ================= CONFIG =================
FACE_SIZE = (200, 200)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ================= STUDENT DETAILS CSV =================
STD_DETAILS_CSV = DATA_DIR / "stddetails.csv"
STUDENT_CSV_FIELDS = ["ID", "Name", "Class", "Image"]


def normalize_image_url(path_value):
    raw = (path_value or "").strip().replace("\\", "/")
    if not raw:
        return ""
    if raw.startswith("/static/"):
        return raw
    if raw.startswith("static/"):
        return f"/{raw}"
    if raw.startswith("frontend/static/"):
        return "/" + raw.replace("frontend/", "")
    return raw


def read_students_csv():
    if not os.path.exists(STD_DETAILS_CSV):
        return []

    rows = []
    with open(STD_DETAILS_CSV, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clean = {k.strip().lower(): (v or "").strip() for k, v in row.items() if k}
            if not clean.get("id"):
                continue
            rows.append(
                {
                    "ID": clean.get("id", ""),
                    "Name": clean.get("name", ""),
                    "Class": clean.get("class", ""),
                    "Image": normalize_image_url(clean.get("image", "")),
                }
            )
    return rows


def get_student_by_id(student_id):
    for row in read_students_csv():
        if row.get("ID") == str(student_id):
            return {
                "id": row.get("ID", ""),
                "name": row.get("Name", ""),
                "class": row.get("Class", ""),
                "image": row.get("Image", ""),
            }
    return None


def save_student_to_csv(student_id, name, student_class, image_path):
    student_id = str(student_id).strip()
    student_name = (name or "").strip()
    student_class = (student_class or "").strip()
    image_url = normalize_image_url(image_path)

    rows = read_students_csv()
    updated = False
    for row in rows:
        if row.get("ID") == student_id:
            row["Name"] = student_name or row.get("Name", "")
            row["Class"] = student_class or row.get("Class", "")
            row["Image"] = image_url or row.get("Image", "")
            updated = True
            break

    if not updated:
        rows.append(
            {
                "ID": student_id,
                "Name": student_name,
                "Class": student_class,
                "Image": image_url,
            }
        )

    with open(STD_DETAILS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=STUDENT_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def list_students():
    students = []
    for row in read_students_csv():
        students.append(
            {
                "id": row.get("ID", ""),
                "name": row.get("Name", ""),
                "student_class": row.get("Class", ""),
                "image": row.get("Image", ""),
            }
        )
    students.sort(key=lambda x: (x.get("name", "").upper(), x.get("id", "")))
    return students


# ================= CAMERA =================
camera = None
# ================= TRAINING CAMERA (NO imshow) =================
train_camera = None
train_frames = []
train_face_count = 0
TRAIN_SAMPLES = 50
train_stream_active = False

current_train_name = None
current_train_id = None

# current training metadata (set by /api/train)
TRAIN_CURRENT_NAME = None
TRAIN_CURRENT_ID = None

class ThreadedCamera:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.frame = None
        self.running = True
        Thread(target=self.update, daemon=True).start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame
            else:
                time.sleep(0.02)

    def get_frame(self):
        return self.frame

    def stop(self):
        self.running = False
        time.sleep(0.1)
        self.cap.release()


class TrainingCamera:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.frame = None
        self.running = True
        Thread(target=self.update, daemon=True).start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame

    def get_frame(self):
        return self.frame

    def stop(self):
        self.running = False
        self.cap.release()


# ================= ATTENDANCE =================
ATTENDANCE_DIR = DATA_DIR / "attendance"
os.makedirs(ATTENDANCE_DIR, exist_ok=True)
FINES_CSV = DATA_DIR / "fines.csv"
FINE_AMOUNT_RS = 500

present_students = set()
seen_counter = {}

def _ensure_fines_csv():
    if os.path.exists(FINES_CSV):
        return
    with open(FINES_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Date",
                "ID",
                "Name",
                "FineAmount",
                "Reason",
                "Status",
                "IssuedTime",
                "PaidTime",
            ],
        )
        writer.writeheader()


def _attendance_file_for_date(date_str):
    return ATTENDANCE_DIR / f"attendance_{date_str}.csv"


def attendance_exists(date_str, student_id):
    file_path = _attendance_file_for_date(date_str)
    if not os.path.exists(file_path):
        return False
    with open(file_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("ID", "")).strip() == str(student_id):
                return True
    return False


def mark_attendance(date_str, student_id, student_name, decision, fine_amount="0", fine_paid="N/A"):
    if attendance_exists(date_str, student_id):
        return
    file_path = _attendance_file_for_date(date_str)
    new_file = not os.path.exists(file_path)
    with open(file_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(
                ["ID", "Name", "Time", "Decision", "FineAmount", "FinePaid"]
            )
        writer.writerow(
            [
                student_id,
                student_name,
                datetime.now().strftime("%H:%M:%S"),
                decision,
                fine_amount,
                fine_paid,
            ]
        )


def update_attendance_payment(date_str, student_id, decision=None, fine_paid=None):
    file_path = _attendance_file_for_date(date_str)
    if not os.path.exists(file_path):
        return False

    with open(file_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return False

    changed = False
    fieldnames = ["ID", "Name", "Time", "Decision", "FineAmount", "FinePaid"]
    for row in rows:
        row.setdefault("Decision", "ENTRY ALLOWED")
        row.setdefault("FineAmount", "0")
        row.setdefault("FinePaid", "N/A")
        if str(row.get("ID", "")).strip() == str(student_id):
            if decision is not None:
                row["Decision"] = decision
            if fine_paid is not None:
                row["FinePaid"] = fine_paid
            changed = True
            break

    if not changed:
        return False

    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return True


def _read_fines():
    _ensure_fines_csv()
    fines = []
    with open(FINES_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fines.append(row)
    return fines


def _write_fines(rows):
    _ensure_fines_csv()
    with open(FINES_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Date",
                "ID",
                "Name",
                "FineAmount",
                "Reason",
                "Status",
                "IssuedTime",
                "PaidTime",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def create_fine_if_missing(date_str, student_id, student_name, reason):
    fines = _read_fines()
    for row in fines:
        if (
            row.get("Date") == date_str
            and row.get("ID") == str(student_id)
            and row.get("Status", "").upper() == "UNPAID"
        ):
            return False
    fines.append(
        {
            "Date": date_str,
            "ID": str(student_id),
            "Name": student_name,
            "FineAmount": str(FINE_AMOUNT_RS),
            "Reason": reason,
            "Status": "UNPAID",
            "IssuedTime": datetime.now().strftime("%H:%M:%S"),
            "PaidTime": "",
        }
    )
    _write_fines(fines)
    return True


def mark_fine_paid_and_mark_attendance(date_str, student_id):
    fines = _read_fines()
    paid_row = None
    for row in fines:
        if (
            row.get("Date") == date_str
            and row.get("ID") == str(student_id)
            and row.get("Status", "").upper() == "UNPAID"
        ):
            row["Status"] = "PAID"
            row["PaidTime"] = datetime.now().strftime("%H:%M:%S")
            paid_row = row
            break

    if not paid_row:
        return False

    _write_fines(fines)

    sid = paid_row.get("ID", "")
    sname = paid_row.get("Name", "--")
    famount = paid_row.get("FineAmount", str(FINE_AMOUNT_RS))
    if attendance_exists(date_str, sid):
        update_attendance_payment(
            date_str=date_str,
            student_id=sid,
            decision="ENTRY_ALLOWED_FINE_APPROVED",
            fine_paid="YES",
        )
    else:
        mark_attendance(
            date_str=date_str,
            student_id=sid,
            student_name=sname,
            decision="ENTRY_ALLOWED_FINE_APPROVED",
            fine_amount=famount,
            fine_paid="YES",
        )
    return True


def load_fines_grouped_by_date():
    grouped = OrderedDict()
    fines = sorted(
        _read_fines(),
        key=lambda r: (r.get("Date", ""), r.get("IssuedTime", "")),
        reverse=True,
    )
    for row in fines:
        date = row.get("Date", "--")
        grouped.setdefault(date, [])
        grouped[date].append(
            {
                "id": row.get("ID", "--"),
                "name": row.get("Name", "--"),
                "amount": row.get("FineAmount", "0"),
                "reason": row.get("Reason", "--"),
                "status": row.get("Status", "UNPAID"),
                "issued_time": row.get("IssuedTime", "--"),
                "paid_time": row.get("PaidTime", ""),
            }
        )
    return grouped


def load_attendance_grouped_by_date():
    grouped = OrderedDict()

    if not os.path.exists(ATTENDANCE_DIR):
        return grouped

    for file in sorted(os.listdir(ATTENDANCE_DIR), reverse=True):
        if file.endswith(".csv"):
            date = file.replace("attendance_", "").replace(".csv", "")
            file_path = ATTENDANCE_DIR / file

            grouped[date] = []

            with open(file_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    grouped[date].append({
                        "id": row.get("ID"),
                        "name": row.get("Name"),
                        "time": row.get("Time"),
                        "decision": row.get("Decision", "ENTRY ALLOWED"),
                        "fine_amount": row.get("FineAmount", "0"),
                        "fine_paid": row.get("FinePaid", "N/A"),
                    })

    return grouped


# ================= UI STATE =================
last_result = {
    "name": "--",
    "id": "--",
    "class": "--",
    "status": "Waiting",
    "final_decision": "WAITING",
    "image": None,
    "uniform_ok": None,
    "id_card_ok": None,
}


# ================= VIDEO STREAM =================
def generate_frames():
    global camera, last_result

    while True:
        if camera is None:
            time.sleep(0.1)
            continue

        frame = camera.get_frame()
        if frame is None:
            continue

        try:
            frame, recognized_id, face_image = recognize_frame(
                frame, present_students, seen_counter
            )
        except Exception as e:
            print("Recognition error:", e)
            continue

        # ✅ Always run compliance detection
        compliance = compliance_detector.check_frame(frame)

        uniform_ok = compliance.get("uniform_ok", False)
        id_card_ok = compliance.get("id_card_ok", False)
        is_compliant = compliance.get("compliant", False)

        if recognized_id:

            clean_id = normalize_student_id(recognized_id)
            student = get_student_by_id(clean_id)

            if student:

                if is_compliant:
                    status = "Recognized (Uniform + ID Card OK)"
                    decision = "ENTRY ALLOWED"
                else:
                    missing_parts = []

                    if not uniform_ok:
                        missing_parts.append("UNIFORM")

                    if not id_card_ok:
                        missing_parts.append("ID CARD")

                    status = "Missing: " + ", ".join(missing_parts)
                    decision = "ENTRY DENIED"

                    create_fine_if_missing(
                        date_str=datetime.now().strftime("%Y-%m-%d"),
                        student_id=clean_id,
                        student_name=student.get("name", "--"),
                        reason=status,
                    )

                last_result = {
                    "name": student.get("name", "--"),
                    "id": clean_id,
                    "class": student.get("class", "--"),
                    "status": status,
                    "final_decision": decision,
                    "image": student.get("image") or face_image,
                    "uniform_ok": uniform_ok,
                    "id_card_ok": id_card_ok,
                }

            else:

                last_result = {
                    "name": "Unknown",
                    "id": clean_id,
                    "class": "--",
                    "status": "Face Not In Database",
                    "final_decision": "WAITING",
                    "image": face_image,
                    "uniform_ok": uniform_ok,
                    "id_card_ok": id_card_ok,
                }

        else:

            last_result = {
                "name": "Guest",
                "id": "--",
                "class": "--",
                "status": "Unknown Face",
                "final_decision": "WAITING",
                "image": None,
                "uniform_ok": uniform_ok,
                "id_card_ok": id_card_ok,
            }

        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes()
            + b"\r\n"
        )


# ================= AUTH ROUTES =================
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        if request.form.get("username") == "admin" and request.form.get("password") == "admin123":
            session["user"] = "admin"
            return redirect(url_for("home"))
        return render_template("login.html", error="Invalid credentials")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ================= CORE ROUTES =================
@app.route("/")
def home():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/start-camera", methods=["POST"])
def start_camera():
    global camera
    if camera is None:
        camera = ThreadedCamera(0)
    return ("", 204)


@app.route("/stop-camera", methods=["POST"])
def stop_camera():
    global camera, last_result
    if camera:
        camera.stop()
        camera = None

    last_result = {
        "name": "--",
        "id": "--",
        "class": "--",
        "status": "Waiting",
        "final_decision": "WAITING",
        "image": None,
        "uniform_ok": None,
        "id_card_ok": None,
    }
    return ("", 204)


@app.route("/current_student")
def get_current_student():
    return jsonify(last_result)


@app.route("/attendance")
def attendance_students():
    attendance_data = load_attendance_grouped_by_date()
    fines_data = load_fines_grouped_by_date()
    total_attendance = sum(len(rows) for rows in attendance_data.values())
    total_fines = sum(len(rows) for rows in fines_data.values())
    unpaid_fines = sum(
        1
        for rows in fines_data.values()
        for row in rows
        if str(row.get("status", "")).upper() == "UNPAID"
    )
    paid_fines = total_fines - unpaid_fines
    return render_template(
        "attendance.html",
        attendance_data=attendance_data,
        fines_data=fines_data,
        summary={
            "total_attendance": total_attendance,
            "total_fines": total_fines,
            "unpaid_fines": unpaid_fines,
            "paid_fines": paid_fines,
        },
    )


@app.route("/api/attendance")
def attendance_api():
    return jsonify(load_attendance_grouped_by_date())


@app.route("/api/fines")
def fines_api():
    return jsonify(load_fines_grouped_by_date())


@app.route("/fines/pay", methods=["POST"])
def pay_fine():
    if "user" not in session:
        return redirect(url_for("login"))
    fine_date = request.form.get("date", "")
    student_id = request.form.get("id", "")
    mark_fine_paid_and_mark_attendance(fine_date, student_id)
    return redirect(url_for("attendance_students"))
# ================= TRAIN STUDENT ROUTES =================
@app.route("/api/train", methods=["POST"])
def api_train():
    data = request.json
    name = data.get("name", "").strip().upper()
    student_id = data.get("id", "").strip()

    # ❌ NAME CHECK
    if not re.fullmatch(r"[A-Z ]+", name):
        return jsonify({
            "error": "Student Name must contain only CAPITAL letters and spaces"
        }), 400

    # ❌ ID CHECK (NUMERIC + MIN 3 DIGITS)
    if not re.fullmatch(r"\d{3,}", student_id):
        return jsonify({
            "error": "Student ID must be numeric and minimum 3 digits"
        }), 400

    # ❌ ID ALREADY EXISTS IN CSV
    if student_id_exists_in_csv(student_id):
        return jsonify({
            "error": "Student ID already exists. Please try again."
        }), 409

    # ✅ CONTINUE TRAINING (ID IS NEW)
    global current_train_name, current_train_id
    current_train_name = name
    current_train_id = student_id

    return jsonify({ "status": "TRAINING_STARTED" })


@app.route("/train/video_feed")
def train_video_feed():
    def gen():
        global train_stream_active, train_camera

        while train_stream_active and train_camera:
            frame = train_camera.get_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            ret, buffer = cv2.imencode(".jpg", frame)
            if not ret:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buffer.tobytes() +
                b"\r\n"
            )

        # 🔴 STREAM ENDS CLEANLY HERE
        print("[INFO] Training video stream stopped")

    return Response(
        gen(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/train/start", methods=["POST"])
def start_training_camera():
    global train_camera, train_face_count, train_frames, train_stream_active

    train_face_count = 0
    train_frames = []
    train_stream_active = True
    train_camera = TrainingCamera(0)

    return jsonify({"status": "TRAIN_CAMERA_STARTED"})

@app.route("/train/capture", methods=["POST"])
def capture_training_face():
    global train_face_count, train_frames

    # 🔒 HARD LIMIT CHECK
    if train_face_count >= TRAIN_SAMPLES:
        return jsonify({
            "error": "Maximum 50 images reached",
            "count": train_face_count
        }), 400

    frame = train_camera.get_frame()
    if frame is None:
        return jsonify({"error": "No frame"}), 400

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) != 1:
        return jsonify({"error": "Ensure exactly one face"}), 400

    x, y, w, h = faces[0]
    face_img = gray[y:y+h, x:x+w]
    face_img = cv2.resize(face_img, FACE_SIZE)

    train_frames.append(face_img)
    train_face_count += 1

    return jsonify({
        "count": train_face_count,
        "remaining": TRAIN_SAMPLES - train_face_count
    })


@app.route("/train/stop", methods=["POST"])
def stop_training_camera():
    global train_camera, train_frames, train_face_count, train_stream_active, current_train_name, current_train_id

    # 🔴 STOP STREAM
    train_stream_active = False

    # 🔴 STOP CAMERA
    if train_camera:
        train_camera.stop()
        train_camera = None

    # 🔴 CLEAR TEMP DATA
    train_frames.clear()
    train_face_count = 0

    # 🔴 DELETE STUDENT DATASET FOLDER
    if current_train_name and current_train_id:
        person_dir = DATASET_DIR / f"{current_train_name}_{current_train_id}"

        if os.path.exists(person_dir):
            shutil.rmtree(person_dir)
            print(f"[INFO] Deleted dataset: {person_dir}")

    # 🔴 RESET TRAINING IDENTITY
    current_train_name = None
    current_train_id = None

    return jsonify({"status": "TRAINING_CANCELLED"})


@app.route("/train/finish", methods=["POST"])
def finish_training():
    global train_camera, train_frames, train_stream_active
    global current_train_name, current_train_id

    if len(train_frames) < TRAIN_SAMPLES:
        return jsonify({"error": "Not enough samples"}), 400

    # 🔴 STOP STREAM FIRST
    train_stream_active = False

    trainer.train_from_frames(
        train_frames,
        current_train_name,
        current_train_id
    )

    if train_camera:
        train_camera.stop()
        train_camera = None

    train_frames.clear()

    trained_id = current_train_id
    trained_name = current_train_name

    auto_image = ""
    candidate = STUDENTS_IMG_DIR / f"{trained_id}.jpg"
    if candidate.exists():
        auto_image = f"{STUDENTS_IMG_URL_PREFIX}/{trained_id}.jpg"

    # Automatically add/update the student in CSV after successful training.
    save_student_to_csv(
        student_id=trained_id,
        name=trained_name,
        student_class="",
        image_path=auto_image,
    )

    current_train_name = None
    current_train_id = None

    return jsonify({
        "status": "TRAINING_COMPLETED",
        "trained_id": trained_id
    })


@app.route("/student/save_details", methods=["POST"])
def save_student_details():
    student_id = (request.form.get("id") or "").strip()
    name = (request.form.get("name") or "").strip()
    student_class = (request.form.get("class") or "").strip()
    image = request.files.get("image")

    if not all([student_id, name]):
        return jsonify({"error": "Student ID and Name are required"}), 400

    image_url = ""
    if image and getattr(image, "filename", ""):
        image_path = STUDENTS_IMG_DIR / f"{student_id}.jpg"
        image.save(str(image_path))
        image_url = f"{STUDENTS_IMG_URL_PREFIX}/{student_id}.jpg"

    save_student_to_csv(
        student_id=student_id,
        name=name,
        student_class=student_class,
        image_path=image_url,
    )

    return jsonify({"status": "STUDENT_DETAILS_SAVED"})

# ================= NAVIGATION ROUTES =================
@app.route("/train")
def train_students():
    return render_template("train.html")

@app.route("/students")
def students_list():
    return render_template("students.html", students=list_students())

@app.route("/admin")
def admin_panel():
    return render_template("admin.html")

@app.route("/notification")
def notification():
    return render_template("notification.html")

@app.route("/support")
def support():
    return render_template("support.html")


# ================= MAIN =================
if __name__ == "__main__":
    app.run(debug=True, threaded=True)



