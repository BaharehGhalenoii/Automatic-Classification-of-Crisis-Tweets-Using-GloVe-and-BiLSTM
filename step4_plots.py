import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, roc_curve, auc,
                             accuracy_score, precision_score, recall_score, f1_score)
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import re
import nltk
from nltk.corpus import stopwords

# ---------- تنظیمات ----------
MAX_LEN = 50

print("=" * 50)
print("مرحله ۴: رسم نمودارها")
print("=" * 50)

# ---------- ۱. بارگذاری داده و مدل ----------
print("\n[۱/۴] بارگذاری داده و مدل...")
df = pd.read_csv('train.csv')
df['label'] = df['class_label'].apply(lambda x: 1 if x == 'informative' else 0)

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

model = load_model('bilstm_model.h5')
print("مدل و توکنایزر لود شدند.")

# ---------- ۲. پیش‌پردازش ----------
print("\n[۲/۴] پیش‌پردازش...")
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join([w for w in text.split() if w not in stop_words])
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_text'] = df['text'].apply(preprocess_text)
sequences = tokenizer.texts_to_sequences(df['clean_text'])
X = pad_sequences(sequences, maxlen=MAX_LEN, padding='post')
y = df['label'].values

# تقسیم به train/test برای رسم نمودار
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"تعداد تست: {len(y_test)}")

# ---------- ۳. پیش‌بینی ----------
print("\n[۳/۴] پیش‌بینی...")
y_pred_prob = model.predict(X_test, verbose=0)
y_pred = (y_pred_prob > 0.5).astype(int)

# محاسبه معیارها
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}")

# ---------- ۴. رسم نمودارها ----------
print("\n[۴/۴] رسم نمودارها...")

# ۴-۱. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['غیراطلاع‌رسان', 'اطلاع‌رسان'],
            yticklabels=['غیراطلاع‌رسان', 'اطلاع‌رسان'])
plt.title('Confusion Matrix - GloVe + BiLSTM', fontsize=14)
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
plt.close()
print("✅ confusion_matrix.png ذخیره شد.")

# ۴-۲. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - GloVe + BiLSTM', fontsize=14)
plt.legend(loc="lower right", fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=300)
plt.close()
print("✅ roc_curve.png ذخیره شد.")

# ۴-۳. Training History (اختیاری، اگه دوباره آموزش بدی)
print("\n✅ همه نمودارها ذخیره شدند.")
print("=" * 50)