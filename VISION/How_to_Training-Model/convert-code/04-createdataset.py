import os
import shutil
import random

# مسیر ورودی (merged)
input_dir = 'merged'
input_images = os.path.join(input_dir, 'images')
input_labels = os.path.join(input_dir, 'labels')

# مسیرهای خروجی
train_dir = 'train'
val_dir = 'val'
train_images = os.path.join(train_dir, 'images')
train_labels = os.path.join(train_dir, 'labels')
val_images = os.path.join(val_dir, 'images')
val_labels = os.path.join(val_dir, 'labels')

# ساخت پوشه‌ها
os.makedirs(train_images, exist_ok=True)
os.makedirs(train_labels, exist_ok=True)
os.makedirs(val_images, exist_ok=True)
os.makedirs(val_labels, exist_ok=True)

# جمع‌آوری جفت‌های معتبر (تصویر + لیبل غیرخالی)
pairs = []
for img_name in os.listdir(input_images):
    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        base_name = os.path.splitext(img_name)[0]
        label_name = f"{base_name}.txt"
        
        img_path = os.path.join(input_images, img_name)
        label_path = os.path.join(input_labels, label_name)
        
        # فقط اگر لیبل وجود داشته باشد و خالی نباشد
        if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
            pairs.append((img_path, label_path))

print(f"تعداد جفت‌های معتبر پیدا شده: {len(pairs)}")

if len(pairs) == 0:
    print("خطا: هیچ جفت تصویر و لیبل معتبری پیدا نشد!")
    exit()

# شافل کردن کاملاً تصادفی
random.shuffle(pairs)

# تقسیم 80% train و 20% val
train_count = int(len(pairs) * 0.8)
val_count = len(pairs) - train_count

train_pairs = pairs[:train_count]
val_pairs = pairs[train_count:]

print(f"تقسیم داده‌ها: {train_count} برای train و {val_count} برای val")

# تابع برای کپی و تغییر نام ترتیبی
def copy_and_rename(pairs_list, img_dest, label_dest, start_idx=0):
    for idx, (img_path, label_path) in enumerate(pairs_list, start=start_idx):
        # نام جدید با 6 رقم (مثل 000123.jpg و 000123.txt)
        new_name = f"{idx:06d}"
        
        img_ext = os.path.splitext(img_path)[1]  # حفظ فرمت اصلی تصویر
        new_img_name = new_name + img_ext
        new_label_name = new_name + ".txt"
        
        shutil.copy(img_path, os.path.join(img_dest, new_img_name))
        shutil.copy(label_path, os.path.join(label_dest, new_label_name))

# کپی و شماره‌گذاری برای train
copy_and_rename(train_pairs, train_images, train_labels, start_idx=0)

# کپی و شماره‌گذاری برای val (ادامه شماره از آخر train)
copy_and_rename(val_pairs, val_images, val_labels, start_idx=train_count)

print("✅ تمام شد!")
print(f"   📁 train: {train_count} نمونه (images و labels با نام‌گذاری 000000 تا {train_count-1:06d})")
print(f"   📁 val:   {val_count} نمونه (images و labels با نام‌گذاری {train_count:06d} به بعد)")