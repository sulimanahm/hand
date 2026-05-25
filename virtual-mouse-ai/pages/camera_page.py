import customtkinter as ctk
from theme import *
from pages.hand_control import start_hand_control

import cv2
from PIL import Image, ImageTk

def create_camera_page(parent, get_settings):
    frame = ctk.CTkFrame(parent, fg_color="transparent")

    title = ctk.CTkLabel(frame, text="الكاميرا وتتبع اليد (Live Feed)", font=FONT_TITLE, text_color=TEXT_WHITE)
    title.pack(anchor="e", pady=(0,10))

    video_frame = ctk.CTkFrame(frame, fg_color=CARD_COLOR, corner_radius=18)
    video_frame.pack(fill="both", expand=True, pady=10)
    video_label = ctk.CTkLabel(video_frame, text="")
    video_label.place(relx=0.5, rely=0.5, anchor="center")

    # وضع تسمية توضيحية إذا الكاميرا غير متوفرة
    mini_map = ctk.CTkFrame(video_frame, width=130, height=80, fg_color=BG_COLOR, corner_radius=10, border_width=1, border_color=ACCENT_COLOR)
    mini_map.place(relx=0.98, rely=0.96, anchor="se")
    ctk.CTkLabel(mini_map, text="تتبع تجريدي", font=("Cairo",10), text_color=TEXT_WHITE).place(relx=0.5, rely=0.5, anchor="center")

    bottom_frame = ctk.CTkFrame(frame, fg_color="transparent")
    bottom_frame.pack(fill="x", pady=10)

    status_label = ctk.CTkLabel(bottom_frame, text="🟢 متصل | الكاميرا: افتراضية | التتبع: عالي الدقة", font=FONT_SUB, text_color=ACCENT_COLOR)
    status_label.pack(side="right")

    stop_flag = {"stop": False}
    hand_thread = [None]  # list to make mutable in nest

    # تحديث الصورة في الواجهة
    def update_image(cv_frame):
        cv2image = cv2.cvtColor(cv_frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
        video_label.update()

    # زر بدء التتبع
    def start_tracking():
        if hand_thread[0] is None:
            stop_flag["stop"] = False
            hand_thread[0] = start_hand_control(update_image, get_settings, stop_flag)
            status_label.configure(text="🟢 التتبع نشط")
    def stop_tracking():
        stop_flag["stop"] = True
        hand_thread[0] = None
        status_label.configure(text="🚫 التتبع متوقف")

    start_btn = ctk.CTkButton(bottom_frame, text="تشغيل التتبع", font=FONT_HEADER, fg_color=ACCENT_COLOR, corner_radius=15, width=150, command=start_tracking)
    start_btn.pack(side="left")
    stop_btn = ctk.CTkButton(bottom_frame, text="إيقاف", font=FONT_HEADER, fg_color="#ef4444", corner_radius=12, width=90, command=stop_tracking)
    stop_btn.pack(side="left", padx=(10, 0))

    # عند فتح الصفحة، تبدأ التتبع مباشرةً تلقائيًا
    frame.after(1000, start_tracking)

    # عند تدمير الصفحة النوّه بوقف التتبع
    def on_destroy(event=None):
        stop_tracking()
    frame.bind("<Destroy>", on_destroy)

    return frame