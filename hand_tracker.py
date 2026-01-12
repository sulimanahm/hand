"""
Hand tracking module using MediaPipe.
كلاس HandTracker يوفر:
- كشف اليدين
- إرجاع إحداثيات التيب (index fingertip) أو أي لاندمارك آخر
- تسوية الحركة (smoothing)
- حساب المسافات و صندوق الحيز (bbox)
- دوال مساعدة لاكتشاف الاصابع المرفوعة
"""
import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple, Optional

class HandTracker:
    def __init__(
        self,
        mode: bool = False,
        maxHands: int = 1,
        detectionCon: float = 0.7,
        trackCon: float = 0.5,
        smoothing: float = 0.75
    ):
        """
        mode: static image mode (False for video stream)
        maxHands: أقصى عدد من الأيادي
        detectionCon: عتبة كشف اليد
        trackCon: عتبة تتبع اليد
        smoothing: معامل تسوية الحركة (0..1). أقرب لـ1 -> أكثر سلاسة (أبطأ استجابة)
        """
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.smoothing = smoothing

        self.mpHands = mp.solutions.hands
        self.mpDraw = mp.solutions.drawing_utils
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )

        # لتخزين آخر موقع مسوّى لكل لاندمارك
        self._smooth_landmarks = None
        self.last_bbox = None
        self.last_handedness = None

    def _normalize_to_pixel(self, lm, w: int, h: int) -> Tuple[int, int]:
        # تحويل من normalized (0..1) إلى بكسل
        return int(lm.x * w), int(lm.y * h)

    def process(self, frame: np.ndarray, draw: bool = True) -> Tuple[np.ndarray, bool]:
        """
        يمرر فريم BGR من OpenCV، يعيد الفريم الموسوم (إن طلب) وبوليان هل يوجد يد؟
        يحدث داخليًا self.last_landmarks و self.last_bbox و self.last_handedness
        """
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        results = self.hands.process(imgRGB)

        self.last_landmarks = []
        self.last_bbox = None
        self.last_handedness = None

        if results.multi_hand_landmarks:
            # لأبسط حالة نأخذ اليد الأولى فقط أو جميع الأيادي لو أردت
            for hand_idx, handLms in enumerate(results.multi_hand_landmarks):
                # احصل على handedness من results.multi_handedness إن كان موجودًا
                if results.multi_handedness and len(results.multi_handedness) > hand_idx:
                    self.last_handedness = results.multi_handedness[hand_idx].classification[0].label

                lm_pixels = []
                x_list = []
                y_list = []
                for id, lm in enumerate(handLms.landmark):
                    px, py = self._normalize_to_pixel(lm, w, h)
                    lm_pixels.append((id, px, py))
                    x_list.append(px)
                    y_list.append(py)

                # تسوية الحركة عبر متوسط متحرك (exponential smoothing)
                if self._smooth_landmarks is None or len(self._smooth_landmarks) != len(lm_pixels):
                    # تهيئة عند المرة الأولى
                    self._smooth_landmarks = [(id, x, y) for (id, x, y) in lm_pixels]
                else:
                    new_smooth = []
                    for (id_old, x_old, y_old), (id_new, x_new, y_new) in zip(self._smooth_landmarks, lm_pixels):
                        sx = int(x_old + self.smoothing * (x_new - x_old))
                        sy = int(y_old + self.smoothing * (y_new - y_old))
                        new_smooth.append((id_new, sx, sy))
                    self._smooth_landmarks = new_smooth

                self.last_landmarks = list(self._smooth_landmarks)

                # bounding box
                xmin, xmax = min(x_list), max(x_list)
                ymin, ymax = min(y_list), max(y_list)
                self.last_bbox = (xmin, ymin, xmax, ymax)

                if draw:
                    # ارسم انطباعات Mediapipe الأصلية على الفريم غير المصفّى
                    self.mpDraw.draw_landmarks(frame, handLms, self.mpHands.HAND_CONNECTIONS)

                # في هذه النسخة نأخذ اليد الأولى فقط ثم نكسر
                break

            return frame, True
        else:
            return frame, False

    def get_landmark(self, id: int) -> Optional[Tuple[int, int]]:
        """
        إرجاع (x, y) لبكسل للاندمارك المطلوب بعد التسوية، أو None إن لم توجد يد
        id: رقم اللاندمارك (0..20 حسب MediaPipe)
        """
        if not self.last_landmarks:
            return None
        for (idx, x, y) in self.last_landmarks:
            if idx == id:
                return (x, y)
        return None

    def get_tip(self, finger: str = "index") -> Optional[Tuple[int, int]]:
        """
        finger: 'index' أو 'thumb' أو 'middle' أو 'ring' أو 'pinky'
        يعيد إحداثيات TIP الخاصة بالإصبع
        """
        finger_map = {
            "thumb": 4,
            "index": 8,
            "middle": 12,
            "ring": 16,
            "pinky": 20
        }
        if finger not in finger_map:
            raise ValueError("finger must be one of 'thumb','index','middle','ring','pinky'")
        return self.get_landmark(finger_map[finger])

    def fingers_up(self) -> Optional[List[bool]]:
        """
        يعيد قائمة من 5 عناصر (thumb,index,middle,ring,pinky) حقيقية/خاطئة إن كانت الإصبع مرفوعة.
        تعتمد المنطقية الأساسية على موقع TIP مقابل PIP (لـ index..pinky) وعلى اتجاه الإبهام بناءً على handedness.
        """
        if not self.last_landmarks:
            return None

        # تحويل القائمة إلى dict id -> (x,y)
        lm = {idx: (x, y) for (idx, x, y) in self.last_landmarks}

        fingers = []
        # للسبابة حتى الخنصر: TIP id > PIP id للمحور Y أصغر يعني مرفوع (في إحداثيات الصورة، y أصغر = أعلى)
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        for tip, pip in zip(tips, pips):
            if tip in lm and pip in lm:
                fingers.append(lm[tip][1] < lm[pip][1])  # True إن كان أعلى (y أصغر)
            else:
                fingers.append(False)

        # الإبهام: نقارن على محور X اعتمادًا على اليد (left/right)
        thumb_up = False
        if 4 in lm and 3 in lm:
            if self.last_handedness == "Right":
                thumb_up = lm[4][0] > lm[3][0]  # في اليد اليمنى، الإبهام للجهة اليمنى إذا مرفوع تقريبًا
            else:
                thumb_up = lm[4][0] < lm[3][0]
        # ترتيب النتيجة: thumb, index, middle, ring, pinky
        return [thumb_up] + fingers

    def distance(self, id1: int, id2: int) -> Optional[float]:
        """
        يحسب المسافة الإقليدية ببكسل بين لاندماركين
        """
        p1 = self.get_landmark(id1)
        p2 = self.get_landmark(id2)
        if p1 is None or p2 is None:
            return None
        return float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))

    def get_bbox(self) -> Optional[Tuple[int, int, int, int]]:
        """
        رجوع bounding box (xmin, ymin, xmax, ymax) إن وجدت
        """
        return self.last_bbox