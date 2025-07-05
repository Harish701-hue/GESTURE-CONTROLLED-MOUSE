import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import mediapipe as mp
import pyautogui
import time
import math
import numpy as np
import ctypes
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL

# Screen size
screen_width, screen_height = pyautogui.size()

# Mediapipe setup
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.8)
mpDraw = mp.solutions.drawing_utils

# Camera
cap = cv2.VideoCapture(0)

# Smoothing
prev_x, prev_y = 0, 0
smoothening = 7

# States
click_state = False
drag_active = False
scroll_y = 0
prev_volume_y = None

# --------- Utility Functions ---------
def fingers_up(lmList):
    fingers = []
    fingers.append(lmList[4][0] > lmList[3][0])  # Thumb
    for tip_id in [8, 12, 16, 20]:
        fingers.append(lmList[tip_id][1] < lmList[tip_id - 2][1])
    return fingers

def get_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def control_volume_gesture(current_y, prev_y):
    delta = prev_y - current_y
    if abs(delta) < 10:
        return
    try:
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = ctypes.cast(interface, ctypes.POINTER(IAudioEndpointVolume))
        current = volume.GetMasterVolumeLevelScalar()
        step = 0.05 if delta > 0 else -0.05
        new_volume = max(0.0, min(1.0, current + step))
        volume.SetMasterVolumeLevelScalar(new_volume, None)
    except:
        pass

# --------- Main Loop ---------
try:
    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_img)

        lmList = []
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append((cx, cy))
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

        if lmList:
            try:
                fingers = fingers_up(lmList)

                # 🖱 Move Cursor – Index Only
                if fingers == [0, 1, 0, 0, 0]:
                    x, y = lmList[8]
                    if 50 < x < w - 50 and 50 < y < h - 50:
                        screen_x = np.interp(x, (0, w), (0, screen_width))
                        screen_y = np.interp(y, (0, h), (0, screen_height))
                        curr_x = prev_x + (screen_x - prev_x) / smoothening
                        curr_y = prev_y + (screen_y - prev_y) / smoothening
                        safe_x = max(100, min(screen_width - 100, curr_x))
                        safe_y = max(100, min(screen_height - 100, curr_y))
                        pyautogui.moveTo(safe_x, safe_y)
                        prev_x, prev_y = curr_x, curr_y

                # 👆 Left Click – Thumb + Index close
                elif fingers[0] == 1 and fingers[1] == 1 and get_distance(lmList[4], lmList[8]) < 30:
                    if not click_state:
                        pyautogui.click()
                        click_state = True
                else:
                    click_state = False

                # 👉 Right Click – Index + Middle
                if fingers[1] == 1 and fingers[2] == 1 and fingers.count(1) == 2:
                    pyautogui.rightClick()
                    time.sleep(0.3)

                # 👉 Double Click – Index + Ring
                if fingers[1] == 1 and fingers[3] == 1 and fingers.count(1) == 2:
                    pyautogui.doubleClick()
                    time.sleep(0.3)

                # 🖐 Drag – Thumb + Index close
                if fingers[0] == 1 and fingers[1] == 1 and get_distance(lmList[4], lmList[8]) < 30:
                    if not drag_active:
                        pyautogui.mouseDown()
                        drag_active = True
                    else:
                        x, y = lmList[8]
                        screen_x = np.interp(x, (0, w), (0, screen_width))
                        screen_y = np.interp(y, (0, h), (0, screen_height))
                        pyautogui.moveTo(screen_x, screen_y)
                elif drag_active:
                    pyautogui.mouseUp()
                    drag_active = False

                # 🌀 Scroll – Index + Middle
                if fingers[1] == 1 and fingers[2] == 1 and fingers.count(1) == 2:
                    scroll_current = lmList[8][1]
                    if scroll_y != 0:
                        pyautogui.scroll(scroll_y - scroll_current)
                    scroll_y = scroll_current

                # 🔀 Multi-select – Index + Middle + Ring
                if fingers[1] and fingers[2] and fingers[3]:
                    pyautogui.keyDown('shift')
                    pyautogui.click()
                    pyautogui.keyUp('shift')
                    time.sleep(0.3)

                # 🔊 Volume – Thumb + Index Up & Slide Up/Down
                if fingers == [1, 1, 0, 0, 0]:
                    center_y = lmList[0][1]
                    if prev_volume_y is not None:
                        control_volume_gesture(center_y, prev_volume_y)
                    prev_volume_y = center_y
                else:
                    prev_volume_y = None

            except Exception as e:
                pass

        cv2.imshow("Virtual Mouse", img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

except Exception as e:
    import traceback
    traceback.print_exc()

finally:
    cap.release()
    cv2.destroyAllWindows()








