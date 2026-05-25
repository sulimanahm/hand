# يحوي مسارات صور أعضاء الفريق. عند إضافة صور حقيقية, غيّر أسماء الملفات بما يناسب

DEVS_IMAGES = {
    "بدرالدين أبكر": "dev1.png",
    "سليمان أحمد آدم": "dev2.png",
    "التجاني محمد": "dev3.png",
    "مهدي الخليفة": "dev4.png",
    "التوم عبدالعزيز": "dev5.png"
}

def get_image_path(dev_name):
    # ارجع المسار الكامل للصورة (مبدئيًا صور وهمية)
    from pathlib import Path
    return str(Path(__file__).parent / DEVS_IMAGES.get(dev_name, "default.png"))
# Inside assets/images.py
