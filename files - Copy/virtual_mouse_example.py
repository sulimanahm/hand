"""
مثال بسيط لعمل ماوس افتراضي باستخدام HandTracker و pyautogui.
تشغيل:
    python virtual_mouse_example.py
"""
import cv2
import pyautogui
import time
from hand_tracker import HandTracker
import numpy as np

def map_coords(x, y, frame_w, frame_h, screen_w, screen_h, margin=0.0):
    """
    تحويل نقطة من نظام كاميرا (0..frame_w, 0..frame_h) إلى نظام الشاشة (0..screen_w, 0..screen_h)
    margin: نسبة حاشية لتقليل الحواف (0..0.5)
    """
    # قلل المساحة الفعالة إن أردت لتجنب الحواف
    mx0 = int(frame_w * margin)
    my0 = int(frame_h * margin)
    mx1 = int(frame_w * (1 - margin))
    my1 = int(frame_h * (1 - margin))

    x_clamped = max(mx0, min(x, mx1))
    y_clamped = max(my0, min(y, my1))

    # تحويل نسبي (0..1)
    rx = (x_clamped - mx0) / (mx1 - mx0)
    ry = (y_clamped - my0) / (my1 - my0)

    # انعكاس المحور X لأن صورة الكاميرا عادة معكوسة للمستخدم
    screen_x = int((1 - rx) * screen_w)
    screen_y = int(ry * screen_h)
    return screen_x, screen_y

def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    tracker = HandTracker(maxHands=1, detectionCon=0.7, trackCon=0.6, smoothing=0.8)
    screen_w, screen_h = pyautogui.size()
    click_down = False
    click_cooldown = 0.3  # تفادي نقرات سريعة متتالية
    last_click_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # حجم الفريم
        fh, fw = frame.shape[:2]
        # انعكاس للمظهر الطبيعي (اختياري)
        frame = cv2.flip(frame, 1)
        annotated, found = tracker.process(frame, draw=True)

        if found:
            tip = tracker.get_tip("index")
            if tip:
                tx, ty = tip
                # لأننا قلبنا الفريم، tip متوافق مع الإطار المعكوس
                sx, sy = map_coords(tx, ty, fw, fh, screen_w, screen_h, margin=0.05)
                # حرّك الماوس — يمكنك تخفيف الحركة هنا لو أردت استخدام smoothing آخر
                pyautogui.moveTo(sx, sy, duration=0)  # duration=0 لتحريك فوري
                cv2.circle(annotated, (tx, ty), 8, (0, 255, 0), cv2.FILLED)

            # قياس مسافة بين الإبهام والسبابة لعمل نقر
            dist = tracker.distance(4, 8)  # IDs: thumb tip=4, index tip=8
            if dist is not None:
                # اختبار عتبة (بقيمة بكسل؛ قد تحتاج ضبط بناء على الكاميرا والبعد)
                if dist < 40:
                    now = time.time()
                    if now - last_click_time > click_cooldown:
                        pyautogui.click()
                        last_click_time = now
                        cv2.putText(annotated, "Click", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                # Optional: show distance
                cv2.putText(annotated, f"Dist:{int(dist)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # عرض FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
        pTime = cTime
        cv2.putText(annotated, f'FPS:{int(fps)}', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        cv2.imshow("Virtual Mouse", annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC للخروج
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()