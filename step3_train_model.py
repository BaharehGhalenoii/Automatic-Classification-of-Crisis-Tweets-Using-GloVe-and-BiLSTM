import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import gensim.downloader as api
import nltk
from nltk.corpus import stopwords
import re
import pickle

# ---------- تنظیمات ----------
MAX_VOCAB = 15000
MAX_LEN = 50
EMBEDDING_DIM = 300
BATCH_SIZE = 128
EPOCHS = 5
DROPOUT_RATE = 0.4

print("=" * 50)
print("مرحله ۳: آموزش مدل GloVe + BiLSTM")
print("=" * 50)

# ---------- ۱. بارگذاری دیتاست ----------
print("\n[۱/۶] بارگذاری دیتاست...")
df = pd.read_csv('train.csv')
print(f"تعداد کل نمونه‌ها: {len(df)}")
print(f"توزیع کلاس‌ها:\n{df['class_label'].value_counts()}")

# تبدیل برچسب به عدد
df['label'] = df['class_label'].apply(lambda x: 1 if x == 'informative' else 0)

# ---------- ۲. پیش‌پردازش ----------
print("\n[۲/۶] پیش‌پردازش متن...")
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
print("نمونه‌ی متن پاک‌شده:")
print(df['clean_text'].iloc[0][:200])

# ---------- ۳. توکن‌سازی ----------
print("\n[۳/۶] توکن‌سازی...")
tokenizer = Tokenizer(num_words=MAX_VOCAB)
tokenizer.fit_on_texts(df['clean_text'])
sequences = tokenizer.texts_to_sequences(df['clean_text'])
X = pad_sequences(sequences, maxlen=MAX_LEN, padding='post')
y = df['label'].values
print(f"شکل داده‌ها: X={X.shape}, y={y.shape}")

# ---------- ۴. بارگذاری GloVe ----------
print("\n[۴/۶] بارگذاری GloVe...")
glove_model = api.load("glove-wiki-gigaword-300")
print(f"تعداد کلمات GloVe: {len(glove_model.key_to_index)}")

# ساخت ماتریس جاسازی
embedding_matrix = np.zeros((MAX_VOCAB, EMBEDDING_DIM))
found = 0
for word, i in tokenizer.word_index.items():
    if i < MAX_VOCAB:
        try:
            embedding_matrix[i] = glove_model[word]
            found += 1
        except KeyError:
            pass
print(f"تعداد کلماتی که بردار GloVe پیدا کردن: {found}")

# ---------- ۵. ساخت مدل ----------
print("\n[۵/۶] ساخت مدل...")
def build_model():
    model = Sequential([
        Embedding(MAX_VOCAB, EMBEDDING_DIM,
                  weights=[embedding_matrix],
                  input_length=MAX_LEN,
                  trainable=False),
        Bidirectional(LSTM(256, return_sequences=True)),
        Dropout(DROPOUT_RATE),
        Bidirectional(LSTM(128)),
        Dropout(DROPOUT_RATE),
        Dense(64, activation='relu'),
        Dropout(DROPOUT_RATE),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer='RMSprop',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

model = build_model()
model.summary()

# ---------- ۶. آموزش با ۵-Fold CV ----------
print("\n[۶/۶] آموزش با Stratified 5-Fold CV...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []}

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    print(f"\n===== Fold {fold}/5 =====")
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = build_model()
    model.fit(X_train, y_train,
              epochs=EPOCHS,
              batch_size=BATCH_SIZE,
              validation_split=0.1,
              verbose=1)

    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_pred_prob)

    results['accuracy'].append(acc)
    results['precision'].append(prec)
    results['recall'].append(rec)
    results['f1'].append(f1)
    results['auc'].append(auc)

    print(f"Fold {fold} → Acc: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

# ---------- ۷. نتایج نهایی ----------
print("\n" + "=" * 50)
print("نتایج نهایی (میانگین ۵ فولد)")
print("=" * 50)
for metric, values in results.items():
    print(f"{metric}: {np.mean(values):.4f} ± {np.std(values):.4f}")

# ذخیره‌ی نتایج
with open('results.txt', 'w') as f:
    for metric, values in results.items():
        f.write(f"{metric}: {np.mean(values):.4f} ± {np.std(values):.4f}\n")

# ذخیره‌ی مدل
model.save('bilstm_model.h5')
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

print("\n✅ نتایج در results.txt ذخیره شد.")
print("✅ مدل در bilstm_model.h5 ذخیره شد.")
print("=" * 50)