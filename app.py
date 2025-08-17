import pickle
import threading
import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, Response
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# --- Load your ML model ---
try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
except Exception as e:
    print("Error loading the model:", e)
    model = None

# --- Mediapipe setup ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
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

# --- Camera globals ---
cap = cv2.VideoCapture(0)
latest_frame = None

def camera_loop():
    global latest_frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                drawing_utils.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    drawing_styles.get_default_hand_landmarks_style(),
                    drawing_styles.get_default_hand_connections_style()
                )

                # Extract features for prediction
                x_, y_, data_aux = [], [], []
                for landmark in hand_landmarks.landmark:
                    x_.append(landmark.x)
                    y_.append(landmark.y)

                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x - min(x_))
                    data_aux.append(landmark.y - min(y_))

                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                try:
                    if model:
                        prediction = model.predict([np.asarray(data_aux)])
                        predicted_character = labels_dict[int(prediction[0])]

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                        cv2.putText(
                            frame, predicted_character, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA
                        )
                except Exception as e:
                    print("Prediction error:", e)

        latest_frame = frame

# Start background camera thread
threading.Thread(target=camera_loop, daemon=True).start()

@app.route('/')
def index():
    return '''
        <html>
            <head>
                <title>Sign Language Translator</title>
            </head>
            <body style="text-align:center; background-color: #f2f2f2;">
                <h1>Sign Language Translator</h1>
                <img src="/video_feed" width="640" height="480" style="border: 2px solid black;" />
            </body>
        </html>
    '''

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            if latest_frame is not None:
                ret, buffer = cv2.imencode('.jpg', latest_frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
