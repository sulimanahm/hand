import customtkinter as ctk
from theme import *
# لا تعتمد جميع الألوان في الأعلى

def create_settings_page(parent, settings_data, apply_settings):
    frame = ctk.CTkFrame(parent, fg_color="transparent")
    title = ctk.CTkLabel(frame, text="إعدادات الحساسية والتحكم ومظهر البرنامج", font=FONT_TITLE, text_color=TEXT_WHITE)
    title.pack(anchor="e", pady=(0,20))
    content = ctk.CTkFrame(frame, fg_color="transparent")
    content.pack(fill="both", expand=True)
    content.columnconfigure(0, weight=1)
    content.columnconfigure(1, weight=1)

    right_panel = ctk.CTkFrame(content, fg_color=CARD_COLOR, corner_radius=15)
    right_panel.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
    ctk.CTkLabel(right_panel, text="أنماط التحكم", font=FONT_HEADER, text_color=TEXT_WHITE).pack(pady=15)
    switches = ["نمط النقر التلقائي", "التمرير الذكي (Smart Scroll)", "التمرير العادي"]
    for text in switches:
        row = ctk.CTkFrame(right_panel, fg_color="transparent")
        row.pack(fill="x", padx=20, pady=10)
        ctk.CTkSwitch(row, text="", progress_color=ACCENT_COLOR).pack(side="left")
        ctk.CTkLabel(row, text=text, font=FONT_SUB, text_color=TEXT_GREY).pack(side="right")

    left_panel = ctk.CTkFrame(content, fg_color=CARD_COLOR, corner_radius=15)
    left_panel.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
    ctk.CTkLabel(left_panel, text="الحساسية/السرعة", font=FONT_HEADER, text_color=TEXT_WHITE).pack(pady=15)
    ctk.CTkLabel(left_panel, text="سرعة المؤشر", font=FONT_SUB, text_color=TEXT_GREY).pack(anchor="e", padx=20)
    slider_speed = ctk.CTkSlider(left_panel, from_=0, to=100, button_color=TEXT_WHITE, progress_color=ACCENT_COLOR)
    slider_speed.set(settings_data.get("pointer_speed", 50))
    slider_speed.pack(fill="x", padx=20, pady=(5,20))
    ctk.CTkLabel(left_panel, text="حساسية التنعيم", font=FONT_SUB, text_color=TEXT_GREY).pack(anchor="e", padx=20)
    slider_smooth = ctk.CTkSlider(left_panel, from_=0, to=100, button_color=TEXT_WHITE, progress_color=ACCENT_COLOR)
    slider_smooth.set(settings_data.get("smoothing", 50))
    slider_smooth.pack(fill="x", padx=20, pady=(5,20))

    # إعدادات المظهر
    ctk.CTkLabel(left_panel, text="الوضع الليلي/الفاتح", font=FONT_SUB, text_color=TEXT_GREY).pack(pady=(10,0))
    switch_appearance = ctk.CTkSwitch(left_panel, text="Night/Dark Mode", progress_color=ACCENT_COLOR)
    switch_appearance.select() if settings_data.get("appearance", "dark") == "dark" else switch_appearance.deselect()
    switch_appearance.pack()
    def on_switch_appearance():
        settings_data["appearance"] = "dark" if switch_appearance.get() == 1 else "light"

    ctk.CTkLabel(left_panel, text="شفافية الخلفية", font=FONT_SUB, text_color=TEXT_GREY).pack(pady=(15,0))
    slider_opacity = ctk.CTkSlider(left_panel, from_=0.7, to=1.0, button_color=TEXT_WHITE, progress_color=ACCENT_COLOR)
    slider_opacity.set(settings_data.get("opacity", 0.95))
    slider_opacity.pack(fill="x", padx=20, pady=(3, 10))
    def on_opacity_change(val):
        settings_data["opacity"] = float(slider_opacity.get())

    bottom_panel = ctk.CTkFrame(frame, fg_color=CARD_COLOR, corner_radius=15)
    bottom_panel.pack(fill="x", pady=10, padx=10)
    ctk.CTkLabel(bottom_panel, text=":لون مؤشر التتبع", font=FONT_SUB, text_color=TEXT_WHITE).pack(side="right", padx=20, pady=15)
    colors = ["#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6"]
    color_var = settings_data.get("accent_color", "#3b82f6")
    def set_color(color):
        settings_data["accent_color"] = color

    for color in colors:
        btn = ctk.CTkButton(
            bottom_panel,
            text="",
            width=30, height=30,
            fg_color=color, corner_radius=15,
            hover_color=color,
            command=lambda c=color: set_color(c)
        )
        btn.pack(side="right", padx=5, pady=15)
    def on_save():
        settings_data["pointer_speed"] = slider_speed.get()
        settings_data["smoothing"] = slider_smooth.get()
        settings_data["appearance"] = "dark" if switch_appearance.get() == 1 else "light"
        settings_data["opacity"] = float(slider_opacity.get())
        apply_settings()

    ctk.CTkButton(bottom_panel, text="حفظ الإعدادات", font=FONT_HEADER, fg_color=ACCENT_COLOR, corner_radius=15, command=on_save).pack(side="left", padx=20, pady=15)
    switch_appearance.configure(command=on_switch_appearance)
    slider_opacity.configure(command=on_opacity_change)

    return frame