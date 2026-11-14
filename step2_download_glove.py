import gensim.downloader as api

print("=" * 50)
print("مرحله ۲: دانلود GloVe با gensim")
print("=" * 50)

try:
    print("در حال دانلود glove-wiki-gigaword-300...")
    # این مدل دقیقاً GloVe 300d آموزش‌دیده روی Wikipedia + Gigaword هست
    model = api.load("glove-wiki-gigaword-300")
    print("\n✅ GloVe دانلود شد!")
    print(f"تعداد کلمات: {len(model.key_to_index)}")
    print(f"ابعاد بردار: {model.vector_size}")
    
    # تست یه کلمه
    print("\n--- تست بردار کلمه 'earthquake' ---")
    print(model['earthquake'][:10])  # ۱۰ عدد اول
    
except Exception as e:
    print(f"\n❌ خطا: {e}")
    print("\n💡 راه‌حل جایگزین: مسیر ۲ (Zenodo) رو امتحان کن.")