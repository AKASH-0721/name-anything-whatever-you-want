from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import csv
import io

app = Flask(__name__)
CORS(app)

# Simplified config for Vercel
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET"])
def home():
    """Home page for Vercel deployment"""
    return jsonify({
        "message": "Face Recognition Attendance System",
        "platform": "Vercel (Serverless)",
        "status": "running",
        "endpoints": {
            "/health": "GET - Health check",
            "/mark_manual": "POST - Manual attendance marking",
            "/": "GET - API information"
        },
        "note": "Training and recognition disabled on Vercel due to package size limits"
    })

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "platform": "Vercel",
        "features": ["manual_attendance"],
        "limitations": "Face recognition disabled due to serverless constraints"
    })

@app.route("/mark_manual", methods=["POST"])
def mark_manual():
    """Manual attendance marking - lightweight version"""
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

        # Optional: filter rolls by section
        if section == "A":
            valid_rolls = [r for r in valid_rolls if 1 <= r <= 64]
        elif section == "B":
            valid_rolls = [r for r in valid_rolls if 65 <= r <= 127]
        elif section == "C":
            valid_rolls = [r for r in valid_rolls if r >= 128]

        # Create CSV content
        csv_content = io.StringIO()
        writer = csv.writer(csv_content)
        writer.writerow(["roll_number", "present"])
        for roll in valid_rolls:
            writer.writerow([roll, "Yes"])
        
        # Return CSV content (since we can't save files in Vercel)
        return jsonify({
            "status": "ok",
            "message": f"{len(valid_rolls)} students marked present manually",
            "marked_rolls": valid_rolls,
            "csv_data": csv_content.getvalue(),
            "platform": "vercel"
        })
        
    except Exception as e:
        return jsonify({"error": "Manual marking failed"}), 500

@app.route("/train", methods=["POST"])
def train_disabled():
    """Training disabled on Vercel"""
    return jsonify({
        "error": "Training is not supported on Vercel due to package size limitations",
        "recommendation": "Use Docker or Render deployment for full ML features",
        "alternative": "Use manual attendance marking endpoint"
    }), 501

@app.route("/recognize", methods=["POST"])
def recognize_disabled():
    """Recognition disabled on Vercel"""
    return jsonify({
        "error": "Face recognition is not supported on Vercel due to package size limitations",
        "recommendation": "Use Docker or Render deployment for full ML features",
        "alternative": "Use manual attendance marking endpoint"
    }), 501

# Vercel entry point
def handler(request):
    return app(request)

if __name__ == "__main__":
    app.run(debug=True)