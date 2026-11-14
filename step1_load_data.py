from datasets import load_dataset
import pandas as pd

print("=" * 50)
print("مرحله ۱: دانلود دیتاست CrisisBench")
print("=" * 50)

# دانلود دیتاست
dataset = load_dataset("QCRI/CrisisBench-english", "informativeness")

print("\n--- ساختار دیتاست ---")
print(dataset)

# بررسی اینکه چه بخش‌هایی وجود دارن
print("\n--- بخش‌های موجود ---")
for split_name in dataset.keys():
    print(f"  - {split_name}: {len(dataset[split_name])} نمونه")

# ذخیره‌ی هر بخش به CSV
for split_name in dataset.keys():
    df = pd.DataFrame(dataset[split_name])
    df.to_csv(f'{split_name}.csv', index=False)
    print(f"✅ {split_name}.csv ذخیره شد ({len(df)} نمونه)")

# نمایش نمونه
print("\n--- نمونه‌ی داده ---")
first_split = list(dataset.keys())[0]
print(dataset[first_split][0])

print("\n" + "=" * 50)
print("مرحله ۱ با موفقیت انجام شد!")
print("=" * 50)