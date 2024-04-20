import cv2
import mediapipe as mp
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
photo_folder='photos/previous'
os.makedirs(photo_folder,exist_ok=True)
photo_count=0
mp_drawing = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    

    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    

    cv2.imshow('Hand Landmarks Detection', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        photo_count += 1
        photo_name = f"hand_{photo_count}.jpg"
        photo_path = os.path.join(photo_folder, photo_name)
        cv2.imwrite(photo_path, frame)
        print(f"Photo {photo_name} saved.")


cap.release()
cv2.destroyAllWindows()
