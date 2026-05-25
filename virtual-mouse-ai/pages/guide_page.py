import customtkinter as ctk
from theme import CARD_COLOR, BG_COLOR, FONT_TITLE, FONT_HEADER, FONT_SUB, TEXT_GREY, TEXT_WHITE

instructions = [
    ("فرد الإصبع السبابة للتحكم في مؤشر الماوس.", "🖕"),
    ("قبضة اليد لإغلاق البرنامج.", "✊"),
    ("تقريب الأصبعين الإبهام والسبابة للتحكم في مستوى الصوت. كلما اقتربا قل المستوى.", "🤏"),
    ("تقريب الأصبعين الوسطى والسبابة لعمل رجوع.", "✌️"),
    ("فرد الأصابع الأربعة عدا الإبهام لعمل كليك أيمن.", "🖐️"),
    ("رفع وتنزيل السبابة بسرعة لعمل دبل كليك.", "👆👇"),
]

def create_guide_page(parent):
    frame = ctk.CTkFrame(parent, fg_color="transparent")
    title = ctk.CTkLabel(frame, text="دليل التحكم باليد (إرشادات الحركة)", font=FONT_TITLE, text_color=TEXT_WHITE)
    title.pack(pady=(0, 20))
    content = ctk.CTkFrame(frame, fg_color=CARD_COLOR, corner_radius=15)
    content.pack(fill="both", expand=True, padx=10, pady=10)
    content.columnconfigure(0, weight=0)
    content.columnconfigure(1, weight=2)
    for i, (text, icon) in enumerate(instructions):
        text_lbl = ctk.CTkLabel(content, text=f"{i+1}. {text}", font=FONT_HEADER, text_color=TEXT_GREY, justify="right")
        text_lbl.grid(row=i, column=1, sticky="e", padx=30, pady=15)
        icon_frame = ctk.CTkFrame(content, width=60, height=60, fg_color=BG_COLOR, corner_radius=15)
        icon_frame.grid(row=i, column=0, sticky="e", padx=15, pady=10)
        icon_frame.pack_propagate(False)
        ctk.CTkLabel(icon_frame, text=icon, font=("Cairo", 28)).place(relx=0.5, rely=0.5, anchor="center")
        # hover تفاعل
        text_lbl.bind("<Enter>", lambda e, lbl=text_lbl: lbl.configure(text_color="#77c6f7"))
        text_lbl.bind("<Leave>", lambda e, lbl=text_lbl: lbl.configure(text_color=TEXT_GREY))
    return frame