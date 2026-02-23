import cv2
import numpy as np
import time
import threading
from flask import Flask, Response, render_template, jsonify

app = Flask(__name__)

# Cascades (vienen incluidas en opencv-python)
CV2_DATA = cv2.data.haarcascades
face_cascade = cv2.CascadeClassifier(CV2_DATA + "haarcascade_frontalface_default.xml")
eye_cascade  = cv2.CascadeClassifier(CV2_DATA + "haarcascade_eye.xml")

# Config 
SLEEP_SEC   = 1.5   # segundos con ojos cerrados → alarma
MIN_FACE_W  = 100   # ignorar caras muy pequeñas (ruido)

# ── Estado compartido
state = {
    "eyes_open":  True,
    "closed_sec": 0.0,
    "alerts":     0,
    "face_found": False,
    "alarm":      False,
    "running":    False,
    "sensitivity": 1.15,   # scaleFactor cascade 
}
lock = threading.Lock()

latest_frame = None
frame_lock   = threading.Lock()
closed_since = None


# Detección 
def detect_eyes_open(gray_face_roi):
    eyes = eye_cascade.detectMultiScale(
        gray_face_roi,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(20, 20),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    return len(eyes) > 0, eyes


def capture_loop():
    global latest_frame, closed_since

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    with lock:
        state["running"] = True

    while True:
        with lock:
            if not state["running"]:
                break
            sensitivity = state["sensitivity"]

        ret, frame = cap.read()
        if not ret:
            time.sleep(0.03)
            continue

        frame = cv2.flip(frame, 1)   # espejo
        h, w  = frame.shape[:2]
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray  = cv2.equalizeHist(gray)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=sensitivity,
            minNeighbors=5,
            minSize=(MIN_FACE_W, MIN_FACE_W),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        now        = time.time()
        face_found = len(faces) > 0
        eyes_open  = False

        if face_found:
            # Tomar la cara más grande
            faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
            fx, fy, fw, fh = faces[0]

            # Dibujar rectángulo de cara
            cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (80, 200, 120), 2)

            # ROI: mitad superior de la cara (zona de ojos)
            eye_y1 = fy + int(fh * 0.15)
            eye_y2 = fy + int(fh * 0.60)
            roi_gray  = gray[eye_y1:eye_y2, fx:fx+fw]
            roi_color = frame[eye_y1:eye_y2, fx:fx+fw]

            eyes_open, eyes = detect_eyes_open(roi_gray)

            for (ex, ey, ew, eh) in eyes:
                cx = ex + ew // 2
                cy = ey + eh // 2
                col = (80, 220, 80) if eyes_open else (50, 50, 255)
                cv2.circle(roi_color, (cx, cy), ew // 2, col, 2)
                cv2.circle(roi_color, (cx, cy), 3,     col, -1)

            # Label estado
            label     = "OJOS ABIERTOS" if eyes_open else "OJOS CERRADOS"
            col_label = (80, 220, 80)   if eyes_open else (50, 50, 255)
            cv2.putText(frame, label, (fx, fy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, col_label, 2)

            # Lógica de tiempo
            if not eyes_open:
                if closed_since is None:
                    closed_since = now
                elapsed = now - closed_since

                # Barra de progreso roja en la parte inferior de la cara
                bar_max = fw
                bar_w   = int(min(1.0, elapsed / SLEEP_SEC) * bar_max)
                cv2.rectangle(frame,
                              (fx, fy + fh + 4),
                              (fx + bar_w, fy + fh + 14),
                              (50, 50, 255), -1)
                cv2.rectangle(frame,
                              (fx, fy + fh + 4),
                              (fx + fw, fy + fh + 14),
                              (100, 100, 100), 1)

                # Tiempo encima de la barra
                cv2.putText(frame, f"{elapsed:.1f}s / {SLEEP_SEC:.1f}s",
                            (fx, fy + fh + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 120, 120), 1)

                with lock:
                    state["closed_sec"] = round(elapsed, 2)
                    if elapsed >= SLEEP_SEC and not state["alarm"]:
                        state["alarm"]  = True
                        state["alerts"] += 1
            else:
                closed_since = None
                with lock:
                    state["closed_sec"] = 0.0

        else:
            closed_since = None
            with lock:
                state["closed_sec"] = 0.0
            cv2.putText(frame, "Buscando cara...", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (120, 120, 120), 2)

        # HUD: instrucciones
        cv2.putText(frame, "SleepGuard", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 90), 1)

        with lock:
            state["face_found"] = face_found
            state["eyes_open"]  = eyes_open

        with frame_lock:
            latest_frame = frame.copy()

    cap.release()


def gen_frames():
    while True:
        with frame_lock:
            if latest_frame is None:
                time.sleep(0.04)
                continue
            frame = latest_frame.copy()

        ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ret:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
               + buf.tobytes() + b"\r\n")
        time.sleep(0.033)  # ~30 fps


# Flask routes
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video")
def video():
    return Response(gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/state")
def get_state():
    with lock:
        return jsonify(dict(state))


@app.route("/dismiss", methods=["POST"])
def dismiss():
    with lock:
        state["alarm"] = False
    return jsonify({"ok": True})


@app.route("/set_sensitivity/<float:val>", methods=["POST"])
def set_sensitivity(val):
    # scaleFactor
    with lock:
        state["sensitivity"] = max(1.05, min(1.4, val))
    return jsonify({"sensitivity": state["sensitivity"]})


@app.route("/start", methods=["POST"])
def start():
    with lock:
        if state["running"]:
            return jsonify({"ok": True})
    threading.Thread(target=capture_loop, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/stop", methods=["POST"])
def stop():
    with lock:
        state["running"] = False
    return jsonify({"ok": True})


if __name__ == "__main__":
    threading.Thread(target=capture_loop, daemon=True).start()
    print("\n" + "="*50)
    print("  SleepGuard corriendo en http://localhost:5050")
    print("  Presiona Ctrl+C para detener")
    print("="*50 + "\n")
    app.run(host="0.0.0.0", port=5050, debug=False, threaded=True)