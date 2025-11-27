import pickle
import threading
import queue
import time
import socket
import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, Response
from flask_cors import CORS
import pyttsx3

# -------------------- Flask --------------------
app = Flask(__name__)
CORS(app)

# -------------------- Model --------------------
model = None
try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
    print("[INIT] Model loaded.")
except Exception as e:
    print("[WARN] Could not load model:", e)

# -------------------- MediaPipe --------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,          # realtime mode
    max_num_hands=1,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)
drawing_styles = mp.solutions.drawing_styles
drawing_utils = mp.solutions.drawing_utils

labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H',
    8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
    15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V',
    22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'Hello', 27: 'Done',
    28: 'Thank You', 29: 'I Love you', 30: 'Sorry', 31: 'Please',
    32: 'You are welcome.'
}

# -------------------- Text-to-Speech --------------------
tts_engine = pyttsx3.init()
tts_queue: "queue.Queue[str]" = queue.Queue()

def tts_worker():
    """Single thread that owns pyttsx3.runAndWait()."""
    while True:
        text = tts_queue.get()
        if text is None:
            break
        try:
            tts_engine.say(text)
            tts_engine.runAndWait()
        except Exception as e:
            print("[TTS] Error:", e)
        finally:
            tts_queue.task_done()

threading.Thread(target=tts_worker, daemon=True).start()

def speak_text(text: str):
    """Enqueue text for TTS; safe across threads."""
    if not text:
        return
    normalized = text.lower()
    tts_queue.put(normalized)

# -------------------- Camera / Inference --------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[WARN] Could not open webcam 0.")

latest_frame = None
latest_lock = threading.Lock()

def camera_loop():
    global latest_frame
    last_prediction = None
    last_said_at = 0.0
    say_cooldown = 0.8  # seconds between TTS calls

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # draw skeleton
                drawing_utils.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    drawing_styles.get_default_hand_landmarks_style(),
                    drawing_styles.get_default_hand_connections_style()
                )

                # features
                xs, ys, data_aux = [], [], []
                for lm in hand_landmarks.landmark:
                    xs.append(lm.x)
                    ys.append(lm.y)
                min_x, min_y = min(xs), min(ys)

                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min_x)
                    data_aux.append(lm.y - min_y)

                x1 = max(int(min(xs) * W) - 10, 0)
                y1 = max(int(min(ys) * H) - 10, 0)
                x2 = min(int(max(xs) * W) + 10, W - 1)
                y2 = min(int(max(ys) * H) + 10, H - 1)

                try:
                    if model is not None:
                        prediction = model.predict([np.asarray(data_aux)])
                        predicted = labels_dict.get(int(prediction[0]), "?")

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
                        cv2.putText(
                            frame, predicted, (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 3, cv2.LINE_AA
                        )

                        # speak if prediction changes, with cooldown
                        now = time.time()
                        if predicted != last_prediction and (now - last_said_at) > say_cooldown:
                            speak_text(predicted)
                            last_prediction = predicted
                            last_said_at = now
                except Exception as e:
                    print("[PREDICT] Error:", e)

        # publish latest frame
        with latest_lock:
            latest_frame = frame

# start camera thread
threading.Thread(target=camera_loop, daemon=True).start()

# -------------------- Routes --------------------
@app.route("/")
def index():
    return """
    <html>
      <head>
        <title>Sign Language Translator</title>
        <meta name="viewport" content="width=device-width, initial-scale=1"/>
        <style>
          body{background:#111;color:#eee;font-family:system-ui,Arial;padding:24px}
          .wrap{max-width:900px;margin:0 auto;text-align:center}
          img{border-radius:12px;box-shadow:0 10px 30px rgba(0,0,0,.4)}
        </style>
      </head>
      <body>
        <div class="wrap">
          <h1>Sign Language Translator</h1>
          <p>If the video doesn't load immediately, wait a moment or refresh.</p>
          <img src="/video_feed" width="720" height="540"/>
        </div>
      </body>
    </html>
    """

@app.route("/video_feed")
def video_feed():
    def generate():
        boundary = b"--frame"
        while True:
            frame = None
            with latest_lock:
                if latest_frame is not None:
                    frame = latest_frame.copy()
            if frame is None:
                time.sleep(0.02)
                continue
            ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok:
                continue
            frame_bytes = buffer.tobytes()
            yield (
                boundary + b"\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                frame_bytes + b"\r\n"
            )

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

# -------------------- Helper: find free port --------------------
def find_free_port(preferred=5000, fallback=8080):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", preferred))
        sock.close()
        return preferred
    except OSError:
        sock.close()
        return fallback

# -------------------- Main --------------------
if __name__ == "__main__":
    port = find_free_port()
    print(f"[INIT] Starting server on http://localhost:{port}/")
    app.run(host="0.0.0.0", port=port, debug=True)


