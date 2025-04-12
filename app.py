import cv2
import mediapipe as mp
import numpy as np
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
vol_range = volume.GetVolumeRange()
min_vol = vol_range[0]
max_vol = vol_range[1]

hold_volume = False
current_volume = volume.GetMasterVolumeLevel()

def is_thumb_up(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
    pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]

    if (thumb_tip.y < thumb_ip.y and
        index_mcp.y < thumb_ip.y and
        middle_mcp.y < thumb_ip.y and
        ring_mcp.y < thumb_ip.y and
        pinky_mcp.y < thumb_ip.y):
        return False
    return True

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label  
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if label == 'Left':
                if is_thumb_up(hand_landmarks):
                    hold_volume = True
                    current_volume = volume.GetMasterVolumeLevel()
                    cv2.putText(img, 'Ok', (164, 100), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 126, 0), 2)
                else:
                    hold_volume = False

            elif label == 'Right' and not hold_volume:
                x1 = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * img.shape[1])
                y1 = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * img.shape[0])
                x2 = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * img.shape[1])
                y2 = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * img.shape[0])

                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
                cv2.circle(img, (x1, y1), 7, (0, 255, 0), cv2.FILLED)
                cv2.circle(img, (x2, y2), 7, (0, 255, 0), cv2.FILLED)

                length = math.hypot(x2 - x1, y2 - y1)

                vol = np.interp(length, [20, 200], [min_vol, max_vol])
                volume.SetMasterVolumeLevel(vol, None)

                vol_bar = np.interp(length, [20, 200], [400, 150])
                cv2.rectangle(img, (50, 150), (85, 400), (0, 0, 255), 3)
                cv2.rectangle(img, (50, int(vol_bar)), (85, 400), (0, 0, 255), cv2.FILLED)

    cv2.imshow("Adjust_Volumne", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
