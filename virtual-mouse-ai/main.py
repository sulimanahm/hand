import customtkinter as ctk
from theme import *
from navigation import switch_page
from pages.team_page import create_team_page
from pages.guide_page import create_guide_page
from pages.settings_page import create_settings_page
from pages.camera_page import create_camera_page
from pages.welcome_screen import create_welcome_screen

class VirtualMouseApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Virtual Mouse AI")
        self.geometry("1100x700")
        self.configure(fg_color=BG_COLOR)
        ctk.set_appearance_mode("dark")
        self.attributes("-alpha", 0.95)
        self.frames = {}
        self.sidebar_frame = None
        self.main_content = None
        self.nav_buttons = []
        self.settings_data = {  # سيتم تحديثه من شاشة الإعدادات
            "appearance": "dark",
            "opacity": 0.95,
            "accent_color": ACCENT_COLOR
        }
        self.show_welcome()

    def get_settings(self):
        # هذه الدالة يستدعيها hand_control عند كل إطار أو كل حدث: تُعادّل بإعداداتك الفعلية
        return self.settings_data
    
    def show_welcome(self):
        def finish():
            self.welcome_frame.destroy()
            self.show_main_layout()
        self.welcome_frame = create_welcome_screen(self, on_finish=finish)
        self.welcome_frame.pack(fill="both", expand=True)

    def show_main_layout(self):
        # القائمة الجانبية
        self.sidebar_frame = ctk.CTkFrame(self, width=250, fg_color=CARD_COLOR, corner_radius=0)
        self.sidebar_frame.pack(side="left", fill="y")
        self.sidebar_frame.pack_propagate(False)
        logo_label = ctk.CTkLabel(self.sidebar_frame, text="🖱️ AI Mouse", font=FONT_TITLE, text_color=ACCENT_COLOR)
        logo_label.pack(pady=30)
        self.nav_buttons = []
        menus = [
            ("الكاميرا والتتبع", "camera"),
            ("إعدادات التحكم", "settings"),
            ("دليل المستخدم", "guide"),
            ("فريق العمل", "team"),
        ]
        for text, name in menus:
            btn = ctk.CTkButton(self.sidebar_frame, text=f"{text}  ➔", font=FONT_HEADER,
                fg_color="transparent", text_color=TEXT_GREY, hover_color=BG_COLOR,
                anchor="e", corner_radius=10, command=lambda n=name: self.switch_page(n))
            btn.pack(pady=10, padx=20, fill="x")
            self.nav_buttons.append((btn, name))
        self.main_content = ctk.CTkFrame(self, fg_color=BG_COLOR, corner_radius=0)
        self.main_content.pack(side="right", fill="both", expand=True)
        self.frames["team"] = create_team_page(self.main_content)
        self.frames["guide"] = create_guide_page(self.main_content)
        self.frames["camera"] = create_camera_page(self.main_content, self.get_settings)
        self.frames["settings"] = create_settings_page(self.main_content, self.settings_data, self.apply_settings)
        self.switch_page("team")

    def apply_settings(self):
        # يُستدعى لتطبيق الإعدادات مباشرة حسب الحاجة من شاشة الإعدادات
        ctk.set_appearance_mode(self.settings_data.get("appearance", "dark"))
        self.attributes("-alpha", float(self.settings_data.get("opacity", 0.95)))
        # يمكن التوسع لدعم المزيد كالألوان
        # example: تغيير ACCENT_COLOR عبر theme

    def switch_page(self, page_name):
        switch_page(self.frames, page_name, self.nav_buttons)

if __name__ == "__main__":
    app = VirtualMouseApp()
    app.mainloop()