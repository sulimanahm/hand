import customtkinter as ctk
import threading, time
from theme import BG_COLOR, CARD_COLOR, ACCENT_COLOR, TEXT_WHITE, TEXT_GREY, FONT_HEADER, FONT_SUB

def create_welcome_screen(parent, on_finish):
    frame = ctk.CTkFrame(parent, fg_color=BG_COLOR)
    center_frame = ctk.CTkFrame(frame, fg_color=BG_COLOR)
    center_frame.place(relx=0.5, rely=0.5, anchor="center")

    icon_label = ctk.CTkLabel(center_frame, text="🖐️🤖", font=("Cairo", 60), text_color=ACCENT_COLOR)
    icon_label.pack(pady=(0, 20))
    title_label = ctk.CTkLabel(center_frame, text="مرحباً بك في برنامج Virtual Mouse AI", font=("Cairo", 32, "bold"), text_color=TEXT_WHITE)
    title_label.pack(pady=5)
    sub_label = ctk.CTkLabel(center_frame, text="تحكم بجهازك بذكاء وحرية كاملة عبر حركات اليد", font=FONT_SUB, text_color=TEXT_GREY)
    sub_label.pack(pady=(0, 30))

    progress = ctk.CTkProgressBar(center_frame, width=400, height=10, progress_color=ACCENT_COLOR, fg_color=CARD_COLOR)
    progress.pack(pady=20)
    progress.set(0)

    start_btn = ctk.CTkButton(center_frame, text="ابدأ الآن", font=FONT_HEADER, fg_color=ACCENT_COLOR, 
                              hover_color="#1e90ff", corner_radius=15, height=45, width=200, 
                              state="disabled", command=on_finish)
    start_btn.pack(pady=20)

    def simulate_loading():
        for i in range(101):
            time.sleep(0.01)
            progress.set(i / 100)
        start_btn.configure(state="normal", text="الدخول للتطبيق")
    threading.Thread(target=simulate_loading, daemon=True).start()

    return frame