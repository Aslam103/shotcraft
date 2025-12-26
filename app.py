from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os, cv2, time, logging
import numpy as np
from collections import deque

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

app = Flask(__name__)

# ================= CONFIG =================
UPLOADS = "uploads"
RESULTS = "results"
MODEL_PATH = "models/basketball_yolo.pt"

os.makedirs(UPLOADS, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)

logging.basicConfig(level=logging.INFO)

# Shot intelligence tuning
FRAME_SKIP = 2
RIM_THRESHOLD = 60
COOLDOWN_MULTIPLIER = 2

SMOOTHING_ALPHA = 0.6
MIN_ARC_HEIGHT = 35
MIN_VERTICAL_SPEED = 2
SHOT_CONFIRM_FRAMES = 6

# ================= MODEL ==================
model = None
if YOLO and os.path.exists(MODEL_PATH):
    model = YOLO(MODEL_PATH)
    logging.info("YOLO loaded")
else:
    logging.warning("YOLO not available – fallback active")

# ================= DETECTOR =================
class BasketballDetector:
    def detect_ball(self, frame):
        if not model:
            return None, 0.0
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = model(rgb, conf=0.25, classes=[32], verbose=False)[0]
        if not res.boxes:
            return None, 0.0
        box = res.boxes.xyxy[0].cpu().numpy()
        conf = float(res.boxes.conf[0])
        return tuple(map(int, box)), conf

# ================= RIM ====================
def detect_rim(frame, h):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=80)
    ys = [l[0][1] for l in lines] if lines is not None else []
    return int(np.median(ys)) if ys else int(h * 0.33)

# ================= ROUTE ==================
@app.route("/analyze", methods=["POST"])
def analyze():
    video = request.files.get("video")
    if not video:
        return jsonify({"error": "no file"}), 400

    filename = secure_filename(video.filename)
    path = os.path.join(UPLOADS, filename)
    video.save(path)

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(3))
    height = int(cap.get(4))
    total_frames = int(cap.get(7))

    ret, first = cap.read()
    if not ret:
        return jsonify({"error": "bad video"}), 400

    rim_y = detect_rim(first, height)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    out_name = f"processed_{filename}"
    out_path = os.path.join(RESULTS, out_name)
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    detector = BasketballDetector()

    attempts = made = 0
    shot_state = 0
    cooldown = 0
    arc_peak_y = None
    confirm_counter = 0

    smoothed_y = prev_y = None
    frame_count = detections = 0
    start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            out.write(frame)
            continue

        processed = frame.copy()
        cv2.line(processed, (0, rim_y), (width, rim_y), (0,0,255), 2)

        box, conf = detector.detect_ball(frame)
        if box and conf > 0.25:
            detections += 1
            x1,y1,x2,y2 = box
            raw_y = (y1+y2)//2

            if smoothed_y is None:
                smoothed_y = raw_y
            else:
                smoothed_y = int(SMOOTHING_ALPHA*smoothed_y + (1-SMOOTHING_ALPHA)*raw_y)

            ball_y = smoothed_y
            ball_x = (x1+x2)//2

            cv2.rectangle(processed,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.circle(processed,(ball_x,ball_y),5,(255,255,0),-1)

            if cooldown > 0:
                cooldown -= 1
            else:
                in_zone = abs(ball_y - rim_y) < RIM_THRESHOLD

                if shot_state == 0 and in_zone:
                    attempts += 1
                    shot_state = 1
                    arc_peak_y = ball_y
                    confirm_counter = 0
                    cooldown = int(fps//2)

                elif shot_state == 1 and ball_y < rim_y:
                    shot_state = 2

                elif shot_state == 2:
                    arc_height = rim_y - arc_peak_y
                    vertical_speed = ball_y - (prev_y or ball_y)

                    if arc_height > MIN_ARC_HEIGHT and vertical_speed > MIN_VERTICAL_SPEED:
                        confirm_counter += 1

                    if confirm_counter >= SHOT_CONFIRM_FRAMES:
                        made += 1
                        shot_state = 0
                        cooldown = int(fps*COOLDOWN_MULTIPLIER)

                    elif abs(ball_y - rim_y) > 200:
                        shot_state = 0

            prev_y = ball_y

        cv2.putText(processed,f"A:{attempts}  M:{made}",(10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        out.write(processed)

    cap.release()
    out.release()
    os.remove(path)

    elapsed = round(time.time()-start,1)

    return jsonify({
        "video_info":{
            "filename":filename,
            "fps":fps,
            "frames":total_frames
        },
        "analysis":{
            "processing_time_seconds":elapsed,
            "frames_analyzed":frame_count,
            "ball_detections":detections
        },
        "results":{
            "attempts":attempts,
            "shots_made":made,
            "shots_missed":max(0,attempts-made),
            "percentage":round((made/attempts*100),1) if attempts else 0,
            "rim_position":rim_y
        },
        "output":{
            "video_file":out_name,
            "download_url":f"/results/{out_name}"
        },
        "status":"success"
    })

@app.route("/results/<path:filename>")
def download(filename):
    return send_from_directory(RESULTS, filename, as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
