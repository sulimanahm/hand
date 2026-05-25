import customtkinter as ctk
from theme import CARD_COLOR, ACCENT_COLOR, TEXT_WHITE, TEXT_GREY, FONT_HEADER, FONT_TITLE, FONT_SUB
from assets.images import DEVS_IMAGES, get_image_path

DEVS = [
    "بدرالدين أبكر",
    "سليمان أحمد آدم",
    "التجاني محمد",
    "مهدي الخليفة",
    "التوم عبدالعزيز"
]

def create_team_page(parent):
    frame = ctk.CTkFrame(parent, fg_color="transparent")
    title = ctk.CTkLabel(frame, text="فريق العمل والمطورين", font=FONT_TITLE, text_color=TEXT_WHITE)
    title.pack(anchor="e", pady=(0,30))
    cards_frame = ctk.CTkFrame(frame, fg_color="transparent")
    cards_frame.pack(fill="x", expand=True)
    for i, dev in enumerate(DEVS):
        cards_frame.columnconfigure(i, weight=1)
        card = ctk.CTkFrame(cards_frame, fg_color=CARD_COLOR, corner_radius=25, height=220)
        card.grid(row=0, column=i, padx=10, sticky="nsew")
        card.pack_propagate(False)
        img_placeholder = ctk.CTkFrame(card, width=70, height=70, corner_radius=35, fg_color="#333", border_width=2, border_color=ACCENT_COLOR)
        img_placeholder.pack(pady=(20, 10))
        ctk.CTkLabel(img_placeholder, text="👤", font=("Cairo", 24)).place(relx=0.5, rely=0.5, anchor="center")
        dev_name = ctk.CTkLabel(card, text=dev, font=FONT_HEADER, text_color=TEXT_WHITE)
        dev_name.pack(pady=5)
        role = ctk.CTkLabel(card, text="مطور ذكاء اصطناعي", font=FONT_SUB, text_color=TEXT_GREY)
        role.pack()
        def on_enter(_): card.configure(fg_color="#23263A")
        def on_leave(_): card.configure(fg_color=CARD_COLOR)
        card.bind("<Enter>", on_enter)
        card.bind("<Leave>", on_leave)
        for widget in (dev_name, role):
            widget.bind("<Enter>", lambda e, card=card: card.configure(fg_color="#23263A"))
            widget.bind("<Leave>", lambda e, card=card: card.configure(fg_color=CARD_COLOR))
    tech_frame = ctk.CTkFrame(frame, fg_color=CARD_COLOR, corner_radius=15)
    tech_frame.pack(fill="x", side="bottom", pady=20)
    ctk.CTkLabel(tech_frame, text="Powered by: Python | OpenCV | MediaPipe | PyAutoGUI | CustomTkinter", 
                 font=("Arial", 14, "bold"), text_color=ACCENT_COLOR).pack(pady=15)
    return frame