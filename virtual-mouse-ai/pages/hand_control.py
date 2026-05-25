import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import threading
import time
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class HandControlThread(threading.Thread):
    def __init__(self, update_image_callback, get_settings, stop_flag):
        super().__init__()
        self.update_image_callback = update_image_callback
        self.get_settings = get_settings
        self.stop_flag = stop_flag
        self.screen_w, self.screen_h = pyautogui.size()
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        self.volume = cast(interface, POINTER(IAudioEndpointVolume))
        self.prev_index_y = None
        self.double_click_last_time = 0

    def run(self):
        cap = cv2.VideoCapture(0)
        with mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        ) as hands:
            while not self.stop_flag["stop"]:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)
                h, w, _ = frame.shape
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.process_landmarks(hand_landmarks, w, h, frame)
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                # Send image to GUI
                if self.update_image_callback:
                    self.update_image_callback(frame)
            cap.release()

    def process_landmarks(self, hand_landmarks, w, h, frame):
        lm = hand_landmarks.landmark
        tips = [lm[4], lm[8], lm[12], lm[16], lm[20]]  # Thumb & fingers
        fingers_up = [int(lm[tip.id].y < lm[tip.id - 2].y) for tip in [lm[4], lm[8], lm[12], lm[16], lm[20]] if hasattr(tip, 'id')]
        x8, y8 = int(lm[8].x * self.screen_w), int(lm[8].y * self.screen_h)  # Index tip
        # Detect extended index for mouse move
        if self.is_finger_up(lm, 1) and not any([self.is_finger_up(lm, i) for i in (2, 3, 4)]):
            pyautogui.moveTo(x8, y8, duration=0.1)
        # Fist to close
        if all(not self.is_finger_up(lm, i) for i in range(1,5)):
            pyautogui.hotkey('alt', 'f4')
        # Thumb & index pinch for volume
        thumb_tip = np.array([lm[4].x, lm[4].y])
        index_tip = np.array([lm[8].x, lm[8].y])
        dist_thumb_index = np.linalg.norm(thumb_tip - index_tip)
        if self.is_finger_up(lm, 0) and self.is_finger_up(lm, 1) and not self.is_finger_up(lm, 2):
            vol = np.interp(dist_thumb_index, [0.02, 0.11], [0.0, 1.0])
            self.volume.SetMasterVolumeLevelScalar(vol, None)

        # Index & middle pinch for "back"
        middle_tip = np.array([lm[12].x, lm[12].y])
        dist_index_middle = np.linalg.norm(index_tip - middle_tip)
        if self.is_finger_up(lm, 1) and self.is_finger_up(lm, 2) and dist_index_middle < 0.05:
            pyautogui.hotkey('alt', 'left')

        # Four up (2-5) for right click
        if all(self.is_finger_up(lm, i) for i in [1, 2, 3, 4]) and not self.is_finger_up(lm, 0):
            pyautogui.click(button="right")
        # Quick index up/down for double click
        index_y = lm[8].y
        if self.prev_index_y is not None:
            if abs(index_y - self.prev_index_y) > 0.1:
                now = time.time()
                if now - self.double_click_last_time < 0.5:
                    pyautogui.doubleClick()
                self.double_click_last_time = now
        self.prev_index_y = index_y

    def is_finger_up(self, lm, finger):  # 0:Thumb, 1:Index...
        if finger == 0:  # Thumb
            return lm[4].x < lm[3].x
        return lm[finger * 4].y < lm[finger * 4 - 2].y

def start_hand_control(update_image_callback, get_settings, stop_flag):
    thread = HandControlThread(update_image_callback, get_settings, stop_flag)
    thread.daemon = True
    thread.start()
    return thread