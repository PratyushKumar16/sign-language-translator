import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 33
dataset_size = 100

cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands (compatible with mediapipe-silicon)
import mediapipe

hands = mediapipe.solutions.hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {j}')

    # Ready prompt
    done = False
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    # Data collection loop
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            continue

        # Process frame with MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Draw landmarks if hand detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mediapipe.solutions.drawing_utils.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mediapipe.solutions.hands.HAND_CONNECTIONS,
                    mediapipe.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mediapipe.solutions.drawing_styles.get_default_hand_connections_style()
                )

        # Show and save frame
        cv2.imshow('frame', frame)
        cv2.waitKey(25)

        # Save pre-processed frame with landmarks
        output_path = os.path.join(class_dir, f'{counter}.jpg')
        cv2.imwrite(output_path, frame)
        print(f'Saved: {output_path}')

        counter += 1

# Cleanup
cap.release()
cv2.destroyAllWindows()
hands.close()