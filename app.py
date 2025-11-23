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
from config import get_config

config = get_config()
DB_FOLDER = config.DB_FOLDER
EXTRACTED = config.EXTRACTED
EMB_FILE = config.EMB_FILE
SECTION = config.SECTION
ALLOWED_EXTENSIONS = config.ALLOWED_EXTENSIONS
ATTENDANCE_CSV = config.ATTENDANCE_CSV

# Create necessary directories
os.makedirs(EXTRACTED, exist_ok=True)
os.makedirs(os.path.dirname(EMB_FILE), exist_ok=True)
os.makedirs(os.path.dirname(ATTENDANCE_CSV), exist_ok=True)

# ------------------------------
# INIT MODEL
# ------------------------------
print("🔄 Loading Buffalo model...")



# 1️⃣ Load full buffalo_l model with optimized settings
face_app = FaceAnalysis(name="buffalo_s", allowed_modules=['detection', 'recognition'])
face_app.prepare(ctx_id=0, det_size=(320, 320))

# Lower detection thresholds for more sensitive detection
try:
    if hasattr(face_app, 'models') and 'detection' in face_app.models:
        det_model = face_app.models['detection']
        if hasattr(det_model, 'det_thresh'):
            det_model.det_thresh = 0.3  # Lower from default 0.5
            print(f"✓ Set detection threshold to {det_model.det_thresh}")
        if hasattr(det_model, 'nms_thresh'):
            det_model.nms_thresh = 0.3  # Lower NMS threshold  
            print(f"✓ Set NMS threshold to {det_model.nms_thresh}")
except Exception as e:
    print(f"⚠️ Could not modify detection parameters: {e}")

print("✅ Buffalo model + SCRFD detector loaded successfully with optimized settings")

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
    try:
        sig = hashlib.sha1()
        if not os.path.exists(folder_path):
            return ""
        for fname in sorted(os.listdir(folder_path)):
            fpath = os.path.join(folder_path, fname)
            if os.path.isfile(fpath):
                try:
                    stat = os.stat(fpath)
                    sig.update(fname.encode())
                    sig.update(str(stat.st_mtime).encode())
                except (OSError, UnicodeEncodeError) as e:
                    print(f"Error processing file {fpath}: {e}")
                    continue
        return sig.hexdigest()
    except Exception as e:
        print(f"Error computing folder signature for {folder_path}: {e}")
        return ""

def load_db():
    if os.path.exists(EMB_FILE):
        try:
            with open(EMB_FILE, "rb") as f:
                db_data = pickle.load(f)
            embeddings_db = db_data.get("embeddings", {}) or {}
            signatures = db_data.get("signatures", {}) or {}
            return embeddings_db, signatures
        except Exception as e:
            print(f"Error loading embeddings database: {e}")
            return {}, {}
    return {}, {}

def save_db(embeddings_db, signatures):
    try:
        with open(EMB_FILE, "wb") as f:
            pickle.dump({"embeddings": embeddings_db, "signatures": signatures}, f)
    except Exception as e:
        print(f"Error saving embeddings database: {e}")
        raise

def recognize_face(embedding, embeddings_db, threshold=0.35):
    best_id, best_score = "Unknown", -1
    
    # Normalize embedding safely
    emb_norm = np.linalg.norm(embedding)
    if emb_norm == 0:
        return "Unknown", 0.0
    embedding = embedding / emb_norm
    
    for sid, ref_emb_list in embeddings_db.items():
        if not isinstance(ref_emb_list, list):
            ref_emb_list = [ref_emb_list]
        for ref_emb in ref_emb_list:
            # Normalize reference embedding safely
            ref_norm = np.linalg.norm(ref_emb)
            if ref_norm == 0:
                continue
            ref_emb = ref_emb / ref_norm
            sim = float(np.dot(embedding, ref_emb))
            if sim > best_score:
                best_score = sim
                best_id = sid
    return (best_id if best_score >= threshold else "Unknown", best_score)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def enhanced_face_detection(img, face_app_instance):
    """Enhanced face detection with multiple preprocessing approaches"""
    
    # Try multiple preprocessing techniques
    preprocessing_methods = [
        ("original", lambda x: x),
        ("bilateral", lambda x: cv2.bilateralFilter(x, 9, 75, 75)),
        ("hist_eq_gray", lambda x: cv2.cvtColor(cv2.equalizeHist(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)), cv2.COLOR_GRAY2BGR)),
        ("clahe_strong", lambda x: apply_clahe(x)),
        ("gamma_0.7", lambda x: adjust_gamma(x, 0.7)),
        ("gamma_1.5", lambda x: adjust_gamma(x, 1.5)),
    ]
    
    all_faces = []
    successful_methods = []
    
    for method_name, preprocess_func in preprocessing_methods:
        try:
            processed_img = preprocess_func(img)
            faces = face_app_instance.get(processed_img)
            
            if faces:
                all_faces.extend(faces)
                successful_methods.append(f"{method_name}({len(faces)})")
                
        except Exception as e:
            print(f"Error in {method_name} preprocessing: {e}")
            continue
    
    # Remove duplicate detections
    unique_faces = remove_duplicate_faces(all_faces)
    
    if successful_methods:
        print(f"    Detection methods that worked: {', '.join(successful_methods)}")
    
    return unique_faces

def apply_clahe(img):
    """Apply CLAHE for contrast enhancement"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def adjust_gamma(img, gamma=1.0):
    """Adjust gamma for brightness correction"""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)

def remove_duplicate_faces(faces, iou_threshold=0.5):
    """Remove duplicate face detections based on IoU"""
    if len(faces) <= 1:
        return faces
    
    # Sort by confidence if available
    faces_with_conf = [(getattr(face, 'det_score', 1.0), face) for face in faces]
    faces_with_conf.sort(key=lambda x: x[0], reverse=True)
    
    keep = []
    for conf, face in faces_with_conf:
        bbox1 = face.bbox
        is_duplicate = False
        
        for kept_face in keep:
            bbox2 = kept_face.bbox
            iou = calculate_iou_simple(bbox1, bbox2)
            if iou > iou_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            keep.append(face)
    
    return keep

def calculate_iou_simple(bbox1, bbox2):
    """Calculate IoU between two bounding boxes"""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def save_attendance_csv(marked_rolls):
    try:
        with open(ATTENDANCE_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["roll_number", "present"])
            for roll in marked_rolls:
                writer.writerow([roll, "Yes"])
    except Exception as e:
        print(f"Error saving attendance CSV: {e}")
        raise

# ------------------------------
# FLASK APP
# ------------------------------
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH

@app.route("/", methods=["GET"])
def home():
    """Home page with API information"""
    return jsonify({
        "message": "Face Recognition Attendance System",
        "endpoints": {
            "/train": "POST - Train the face recognition model",
            "/recognize": "POST - Recognize faces and mark attendance",
            "/mark_manual": "POST - Manually mark attendance"
        },
        "status": "running"
    })

@app.route("/train", methods=["POST"])
def train():
    try:
        if not os.path.exists(DB_FOLDER):
            return jsonify({"error": f"Training folder '{DB_FOLDER}' not found"}), 400
            
        embeddings_db, signatures = load_db()
        updated, skipped_no_face = 0, 0

        for student_id in sorted(os.listdir(DB_FOLDER)):
            student_path = os.path.join(DB_FOLDER, student_id)
            if not os.path.isdir(student_path):
                continue
            if not is_valid_folder(student_id, SECTION):
                continue

            try:
                current_sig = compute_folder_signature(student_path)
                prev_sig = signatures.get(student_id)
                if prev_sig == current_sig and student_id in embeddings_db:
                    continue  # unchanged

                student_embeddings = []

                for img_name in sorted(os.listdir(student_path)):
                    img_path = os.path.join(student_path, img_name)
                    if not os.path.isfile(img_path):
                        continue
                    
                    try:
                        img = cv2.imread(img_path)
                        if img is None:
                            print(f"Warning: Could not read image {img_path}")
                            continue

                        # Use enhanced face detection
                        print(f"  Processing: {img_name}")
                        faces = enhanced_face_detection(img, face_app)
                        
                        if not faces:
                            print(f"    No faces detected in {img_name}")
                            continue

                        print(f"    Found {len(faces)} face(s)")
                        for face in faces:
                            # Normalize embedding safely
                            emb_norm = np.linalg.norm(face.embedding)
                            if emb_norm > 0:
                                emb = face.embedding / emb_norm
                                student_embeddings.append(emb)
                    except Exception as e:
                        print(f"Error processing image {img_path}: {e}")
                        continue

                if student_embeddings:
                    embeddings_db[student_id] = student_embeddings
                    signatures[student_id] = current_sig
                    updated += 1
                else:
                    embeddings_db.pop(student_id, None)
                    signatures.pop(student_id, None)
                    skipped_no_face += 1
            except Exception as e:
                print(f"Error processing student folder {student_id}: {e}")
                continue

        save_db(embeddings_db, signatures)
        return jsonify({
            "status": "ok",
            "updated": updated,
            "skipped_no_face": skipped_no_face,
            "total_students": len(embeddings_db)
        })
    except Exception as e:
        print(f"Error in train endpoint: {e}")
        return jsonify({"error": "Internal server error during training"}), 500

@app.route("/recognize", methods=["POST"])
def recognize():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400
        if not allowed_file(file.filename):
            return jsonify({"error": "File type not allowed"}), 400

        filename = secure_filename(file.filename)
        save_path = os.path.join(EXTRACTED, filename)
        
        try:
            file.save(save_path)
        except Exception as e:
            return jsonify({"error": f"Could not save uploaded file: {str(e)}"}), 500

        try:
            img = cv2.imread(save_path)
            if img is None:
                return jsonify({"error": "Could not read image file"}), 400
        except Exception as e:
            return jsonify({"error": f"Error reading image: {str(e)}"}), 400

        embeddings_db, _ = load_db()
        if not embeddings_db:
            return jsonify({"error": "No trained embeddings found. Please train the model first."}), 400
            
        try:
            faces = enhanced_face_detection(img, face_app)
        except Exception as e:
            return jsonify({"error": f"Error during face detection: {str(e)}"}), 500
            
        results = []
        marked_ids = []

        if not faces:
            return jsonify({"status": "ok", "results": [], "marked_ids": [], "message": "No faces detected in the image"})

        for i, face in enumerate(faces, 1):
            try:
                x1, y1, x2, y2 = map(int, face.bbox)
                
                # Validate bounding box
                if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0:
                    continue
                    
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

                if student != "Unknown":
                    marked_ids.append(student)
            except Exception as e:
                print(f"Error processing face {i}: {e}")
                continue

        # Save attendance CSV
        try:
            save_attendance_csv(marked_ids)
        except Exception as e:
            return jsonify({"error": f"Could not save attendance: {str(e)}"}), 500

        return jsonify({
            "status": "ok",
            "results": results,
            "marked_ids": marked_ids,
            "message": f"{len(marked_ids)} students marked present"
        })
    except Exception as e:
        print(f"Error in recognize endpoint: {e}")
        return jsonify({"error": "Internal server error during recognition"}), 500
@app.route("/mark_manual", methods=["POST"])
def mark_manual():
    """
    Receives JSON: { "rolls": [6, 11, 22], "section": "A" }
    Saves the marked rolls to CSV, overwriting previous entries.
    """
    try:
        data = request.get_json()
        if not data or "rolls" not in data:
            return jsonify({"error": "No rolls provided"}), 400

        rolls = data["rolls"]
        if not isinstance(rolls, list):
            return jsonify({"error": "Rolls must be a list"}), 400
            
        # Validate roll numbers
        valid_rolls = []
        for roll in rolls:
            if isinstance(roll, (int, str)):
                try:
                    roll_num = int(roll)
                    if roll_num > 0:
                        valid_rolls.append(roll_num)
                except (ValueError, TypeError):
                    continue
            
        if not valid_rolls:
            return jsonify({"error": "No valid roll numbers provided"}), 400
            
        section = data.get("section", "ALL")

        # Optional: filter rolls by section if needed
        if section == "A":
            valid_rolls = [r for r in valid_rolls if 1 <= r <= 64]
        elif section == "B":
            valid_rolls = [r for r in valid_rolls if 65 <= r <= 127]
        elif section == "C":
            valid_rolls = [r for r in valid_rolls if r >= 128]

        # Save CSV
        try:
            with open(ATTENDANCE_CSV, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["roll_number", "present"])
                for roll in valid_rolls:
                    writer.writerow([roll, "Yes"])
        except Exception as e:
            return jsonify({"error": f"Could not save attendance: {str(e)}"}), 500

        return jsonify({
            "status": "ok",
            "message": f"{len(valid_rolls)} students marked present manually"
        })
    except Exception as e:
        print(f"Error in mark_manual endpoint: {e}")
        return jsonify({"error": "Internal server error during manual marking"}), 500


if __name__ == "__main__":
    import os
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV', 'production') != 'production'
    app.run(debug=debug, host='0.0.0.0', port=port)
