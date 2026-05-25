def switch_page(frames, page_name, nav_buttons):
    # إخفاء جميع الصفحات ثم إظهار المختارة + إظهار الزر الحالي مميز
    for frame in frames.values():
        frame.pack_forget()
    frames[page_name].pack(fill="both", expand=True, padx=20, pady=20)
    for btn, name in nav_buttons:
        if name == page_name:
            btn.configure(fg_color="#3b82f6", text_color="#fff")
        else:
            btn.configure(fg_color="transparent", text_color="#a0aec0")