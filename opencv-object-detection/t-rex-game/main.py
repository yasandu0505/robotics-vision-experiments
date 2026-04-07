import cv2
import pyautogui
import mediapipe as mp
import time

mp_hands    = mp.solutions.hands
mp_drawing  = mp.solutions.drawing_utils
mp_styles   = mp.solutions.drawing_styles         

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,                             
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Finger tip / pip landmark indices
FINGER_TIPS = [8, 12, 16, 20]
THUMB_TIP   = 4

# debounce
last_action_time = 0
DEBOUNCE_SECONDS = 0.5          # min gap between space presses

def fingers_up(hand_landmarks, handedness_label):
    """
    Returns a list of 5 booleans (thumb → pinky).
    handedness_label: 'Left' or 'Right' (mediapipe uses mirrored labels
    when the frame is flipped, so we correct for that).
    """
    lm = hand_landmarks.landmark
    fingers = []

    # Thumb — compare x-axis; direction depends on which hand
    # After cv2.flip(frame,1), mediapipe 'Right' = user's right hand
    if handedness_label == "Right":
        fingers.append(1 if lm[THUMB_TIP].x < lm[THUMB_TIP - 1].x else 0)
    else:
        fingers.append(1 if lm[THUMB_TIP].x > lm[THUMB_TIP - 1].x else 0)

    # Index → Pinky — compare y-axis (tip above pip = extended)
    for tip in FINGER_TIPS:
        fingers.append(1 if lm[tip].y < lm[tip - 2].y else 0)

    return fingers

# webcam setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam — check device index.")

pyautogui.FAILSAFE = False      # disable corner-abort so it doesn't crash mid-use

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame grab failed, retrying…")
        continue

    frame     = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # mediapipe 0.10.x: mark image as not writeable for performance
    rgb_frame.flags.writeable = False
    result = hands.process(rgb_frame)
    rgb_frame.flags.writeable = True

    if result.multi_hand_landmarks and result.multi_handedness:
        for hand_landmarks, hand_info in zip(
            result.multi_hand_landmarks, result.multi_handedness
        ):
            label = hand_info.classification[0].label   # 'Left' or 'Right'

            # sketch style drawing of the hand
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0, 255, 0),   thickness=2)
            )

            # gesture detection
            finger_status  = fingers_up(hand_landmarks, label)
            total_fingers  = sum(finger_status)
            now            = time.time()

            # FIST
            if total_fingers == 0:                      
                print("FIST DETECTED - firing space")
                if now - last_action_time > DEBOUNCE_SECONDS:
                    pyautogui.press("space")
                    last_action_time = now
                cv2.putText(frame, "FIST  ->  SPACE!", (50, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # OPEN HAND
            elif total_fingers == 5:                    
                cv2.putText(frame, "FIVE FINGERS", (50, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("T-Rex control on chorem", frame)
    if cv2.waitKey(1) & 0xFF == 27:     # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()