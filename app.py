from flask import Flask, request, jsonify
import cv2
import os
import re
import numpy as np
import pickle
import hashlib
import csv
from insightface.app import FaceAnalysis
from werkzeug.utils import secure_filename

# ------------------------------
# CONFIG
# ------------------------------
DB_FOLDER = "train"
EXTRACTED = "extracted_faces"
EMB_FILE = "embeddings_db.pkl"
SECTION = "ALL"  # change to A / B / C / ALL
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
ATTENDANCE_CSV = "attendance.csv"

os.makedirs(EXTRACTED, exist_ok=True)

# ------------------------------
# INIT MODEL
# ------------------------------
print("🔄 Loading Buffalo model...")
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0, det_size=(640, 640))
print("✅ Model loaded")

# ------------------------------
# HELPERS
# ------------------------------
def parse_ad_number(folder_name: str):
    if folder_name.strip().upper() == "NA":
        return "NA"
    m = re.search(r'AD\s*0*([0-9]+)', folder_name, flags=re.IGNORECASE)
    if not m:
        return None
    return int(m.group(1))

def is_valid_folder(folder_name: str, choice: str) -> bool:
    tag = parse_ad_number(folder_name)
    if tag == "NA":
        return True
    if tag is None:
        return False
    if choice == "ALL":
        return True
    if choice == "A":
        return 1 <= tag <= 64
    if choice == "B":
        return 65 <= tag <= 127
    if choice == "C":
        return tag >= 128
    return False

def compute_folder_signature(folder_path: str) -> str:
    sig = hashlib.sha1()
    for fname in sorted(os.listdir(folder_path)):
        fpath = os.path.join(folder_path, fname)
        if os.path.isfile(fpath):
            stat = os.stat(fpath)
            sig.update(fname.encode())
            sig.update(str(stat.st_mtime).encode())
    return sig.hexdigest()

def load_db():
    if os.path.exists(EMB_FILE):
        try:
            with open(EMB_FILE, "rb") as f:
                db_data = pickle.load(f)
            embeddings_db = db_data.get("embeddings", {}) or {}
            signatures = db_data.get("signatures", {}) or {}
            return embeddings_db, signatures
        except:
            return {}, {}
    return {}, {}

def save_db(embeddings_db, signatures):
    with open(EMB_FILE, "wb") as f:
        pickle.dump({"embeddings": embeddings_db, "signatures": signatures}, f)

def recognize_face(embedding, embeddings_db, threshold=0.35):
    best_id, best_score = "Unknown", -1
    embedding = embedding / np.linalg.norm(embedding)
    for sid, ref_emb_list in embeddings_db.items():
        if not isinstance(ref_emb_list, list):
            ref_emb_list = [ref_emb_list]
        for ref_emb in ref_emb_list:
            ref_emb = ref_emb / np.linalg.norm(ref_emb)
            sim = float(np.dot(embedding, ref_emb))
            if sim > best_score:
                best_score = sim
                best_id = sid
    return (best_id if best_score >= threshold else "Unknown", best_score)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_attendance_csv(marked_rolls):
    with open(ATTENDANCE_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["roll_number", "present"])
        for roll in marked_rolls:
            writer.writerow([roll, "Yes"])

# ------------------------------
# FLASK APP
# ------------------------------
app = Flask(__name__)

@app.route("/train", methods=["POST"])
def train():
    embeddings_db, signatures = load_db()
    updated, skipped_no_face = 0, 0

    for student_id in sorted(os.listdir(DB_FOLDER)):
        student_path = os.path.join(DB_FOLDER, student_id)
        if not os.path.isdir(student_path):
            continue
        if not is_valid_folder(student_id, SECTION):
            continue

        current_sig = compute_folder_signature(student_path)
        prev_sig = signatures.get(student_id)
        if prev_sig == current_sig and student_id in embeddings_db:
            continue  # unchanged

        student_embeddings = []

        for img_name in sorted(os.listdir(student_path)):
            img_path = os.path.join(student_path, img_name)
            if not os.path.isfile(img_path):
                continue
            img = cv2.imread(img_path)
            if img is None:
                continue

            faces = face_app.get(img)
            if not faces:
                continue

            for face in faces:
                emb = face.embedding / np.linalg.norm(face.embedding)
                student_embeddings.append(emb)

        if student_embeddings:
            embeddings_db[student_id] = student_embeddings
            signatures[student_id] = current_sig
            updated += 1
        else:
            embeddings_db.pop(student_id, None)
            signatures.pop(student_id, None)
            skipped_no_face += 1

    save_db(embeddings_db, signatures)
    return jsonify({
        "status": "ok",
        "updated": updated,
        "skipped_no_face": skipped_no_face,
        "total_students": len(embeddings_db)
    })

@app.route("/recognize", methods=["POST"])
def recognize():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(EXTRACTED, filename)
    file.save(save_path)

    img = cv2.imread(save_path)
    if img is None:
        return jsonify({"error": "Could not read image"}), 400

    embeddings_db, _ = load_db()
    faces = face_app.get(img)
    results = []
    marked_ids = []

    if not faces:
        return jsonify({"status": "ok", "results": [], "marked_rolls": []})

    for i, face in enumerate(faces, 1):
        x1, y1, x2, y2 = map(int, face.bbox)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        crop_resized = cv2.resize(crop, (160, 160))
        face_file = f"{os.path.splitext(filename)[0]}_face{i}.jpg"
        cv2.imwrite(os.path.join(EXTRACTED, face_file), crop_resized)

        student, score = recognize_face(face.embedding, embeddings_db)
        results.append({
            "face_file": face_file,
            "assigned_label": student,
            "similarity": round(score, 3)
        })

        # extract roll number
        
        if student != "Unknown":
            marked_ids.append(student)

    # Save attendance CSV
    save_attendance_csv(marked_ids)

    return jsonify({
        "status": "ok",
        "results": results,
        "marked_ids": marked_ids,
        "message": f"{len(marked_ids)} students marked present"
    })
@app.route("/mark_manual", methods=["POST"])
def mark_manual():
    """
    Receives JSON: { "rolls": [6, 11, 22], "section": "A" }
    Saves the marked rolls to CSV, overwriting previous entries.
    """
    data = request.get_json()
    if not data or "rolls" not in data:
        return jsonify({"error": "No rolls provided"}), 400

    rolls = data["rolls"]
    section = data.get("section", "ALL")

    # Optional: filter rolls by section if needed
    if section == "A":
        rolls = [r for r in rolls if 1 <= r <= 64]
    elif section == "B":
        rolls = [r for r in rolls if 65 <= r <= 127]
    elif section == "C":
        rolls = [r for r in rolls if r >= 128]

    # Save CSV
    with open(ATTENDANCE_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["roll_number", "present"])
        for roll in rolls:
            writer.writerow([roll, "Yes"])

    return jsonify({
        "status": "ok",
        "message": f"{len(rolls)} students marked present manually"
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
