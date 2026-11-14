import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Embedding, Bidirectional, GRU, LSTM,
                                     Dense, Dropout, Conv1D, GlobalMaxPooling1D)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)
import gensim.downloader as api
import nltk
from nltk.corpus import stopwords
import re
import matplotlib.pyplot as plt
import seaborn as sns

# ================== SEED (مورد ۵) ==================
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ================== تنظیمات ==================
MAX_VOCAB = 15000
MAX_LEN = 50
EMBEDDING_DIM = 300
BATCH_SIZE = 128
EPOCHS = 5
DROPOUT_RATE = 0.4
DECISION_THRESHOLD = 0.5   # مورد ۵: آستانه تصمیم

print("=" * 60)
print("پایپ‌لاین نهایی (بدون Data Leakage)")
print("=" * 60)
print(f"TensorFlow: {tf.__version__}")
print(f"NumPy: {np.__version__}")
print(f"Pandas: {pd.__version__}")
print(f"Seed: {SEED}, Threshold: {DECISION_THRESHOLD}")

# ================== ۱. داده ==================
df = pd.read_csv('train.csv')
df['label'] = df['class_label'].apply(lambda x: 1 if x == 'informative' else 0)

# مورد ۲: بررسی توییت تکراری
n_before = len(df)
df = df.drop_duplicates(subset=['text']).reset_index(drop=True)
n_after = len(df)
print(f"\n[مورد ۲] توییت تکراری حذف شد: {n_before - n_after} (قبل: {n_before}, بعد: {n_after})")

# ================== ۲. پیش‌پردازش ==================
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
    return re.sub(r'\s+', ' ', text).strip()

df['clean_text'] = df['text'].apply(preprocess_text)
texts = df['clean_text'].values
y = df['label'].values
events = df['event'].values if 'event' in df.columns else None

# ================== ۳. GloVe ==================
print("\n[۳] بارگذاری GloVe...")
glove_model = api.load("glove-wiki-gigaword-300")

def build_embedding_matrix(tokenizer):
    """ماتریس جاسازی رو فقط از واژگان توکنایزر همین Fold می‌سازه (مورد ۱)"""
    matrix = np.zeros((MAX_VOCAB, EMBEDDING_DIM))
    found = 0
    for word, i in tokenizer.word_index.items():
        if i < MAX_VOCAB:
            try:
                matrix[i] = glove_model[word]
                found += 1
            except KeyError:
                pass
    return matrix, found

# ================== ۴. مدل‌ها ==================
def build_bilstm(emb_matrix):
    m = Sequential([
        Embedding(MAX_VOCAB, EMBEDDING_DIM, weights=[emb_matrix],
                  input_length=MAX_LEN, trainable=False),
        Bidirectional(LSTM(256, return_sequences=True)),
        Dropout(DROPOUT_RATE),
        Bidirectional(LSTM(128)),
        Dropout(DROPOUT_RATE),
        Dense(64, activation='relu'),
        Dropout(DROPOUT_RATE),
        Dense(1, activation='sigmoid')
    ])
    m.compile(optimizer='RMSprop', loss='binary_crossentropy', metrics=['accuracy'])
    return m

def build_bigru_lstm(emb_matrix):
    """مورد ۳: جزئیات کامل مدل دوم"""
    m = Sequential([
        Embedding(MAX_VOCAB, EMBEDDING_DIM, weights=[emb_matrix],
                  input_length=MAX_LEN, trainable=False),
        Conv1D(filters=128, kernel_size=5, activation='relu'),   # ← فیلتر=128، کرنل=5
        Bidirectional(GRU(100, return_sequences=True)),           # ← 100 واحد
        Dropout(DROPOUT_RATE),
        LSTM(100, return_sequences=True),                         # ← 100 واحد
        Dropout(DROPOUT_RATE),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dropout(DROPOUT_RATE),
        Dense(1, activation='sigmoid')
    ])
    m.compile(optimizer='RMSprop', loss='binary_crossentropy', metrics=['accuracy'])
    return m

# ================== ۵. آموزش با CV (بدون Data Leakage) ==================
def run_cv(model_builder, model_name):
    print(f"\n{'=' * 60}\n{model_name}\n{'=' * 60}")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    res = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []}
    oof_probs = np.zeros(len(texts))  # مورد ۴: پیش‌بینی خارج از Fold

    for fold, (tr_idx, te_idx) in enumerate(skf.split(texts, y), 1):
        print(f"\n--- Fold {fold}/5 ---")

        # مورد ۱: توکنایزر فقط روی داده‌ی آموزشی این Fold
        tok = Tokenizer(num_words=MAX_VOCAB)
        tok.fit_on_texts(texts[tr_idx])

        X_tr = pad_sequences(tok.texts_to_sequences(texts[tr_idx]),
                             maxlen=MAX_LEN, padding='post')
        X_te = pad_sequences(tok.texts_to_sequences(texts[te_idx]),
                             maxlen=MAX_LEN, padding='post')

        emb, found = build_embedding_matrix(tok)
        print(f"  واژگان Fold: {len(tok.word_index)}, بردار GloVe: {found}")

        model = model_builder(emb)
        model.fit(X_tr, y[tr_idx], epochs=EPOCHS, batch_size=BATCH_SIZE,
                  validation_split=0.1, verbose=1)

        probs = model.predict(X_te, verbose=0).ravel()
        oof_probs[te_idx] = probs  # ذخیره برای نمودار OOF
        preds = (probs >= DECISION_THRESHOLD).astype(int)

        res['accuracy'].append(accuracy_score(y[te_idx], preds))
        res['precision'].append(precision_score(y[te_idx], preds, zero_division=0))
        res['recall'].append(recall_score(y[te_idx], preds, zero_division=0))
        res['f1'].append(f1_score(y[te_idx], preds, zero_division=0))
        res['auc'].append(roc_auc_score(y[te_idx], probs))

        print(f"  Acc={res['accuracy'][-1]:.4f}, F1={res['f1'][-1]:.4f}, "
              f"AUC={res['auc'][-1]:.4f}")

    print(f"\n=== {model_name} (میانگین ۵ فولد) ===")
    for k, v in res.items():
        print(f"{k}: {np.mean(v):.4f} ± {np.std(v):.4f}")

    return res, oof_probs

# ================== ۶. اجرا ==================
results_bilstm, oof_bilstm = run_cv(build_bilstm, "GloVe + BiLSTM")
results_bigru, oof_bigru = run_cv(build_bigru_lstm, "GloVe + BiGRU-LSTM")

# ================== ۷. نمودار OOF (مورد ۴) ==================
print("\n[۴] رسم نمودارهای OOF...")
for name, oof in [("BiLSTM", oof_bilstm), ("BiGRU-LSTM", oof_bigru)]:
    preds = (oof >= DECISION_THRESHOLD).astype(int)

    cm = confusion_matrix(y, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Info', 'Info'],
                yticklabels=['Non-Info', 'Info'])
    plt.title(f'Confusion Matrix (OOF) - {name}')
    plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f'cm_oof_{name}.png', dpi=300); plt.close()

    fpr, tpr, _ = roc_curve(y, oof)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'AUC={roc_auc_score(y, oof):.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title(f'ROC (OOF) - {name}'); plt.legend()
    plt.tight_layout()
    plt.savefig(f'roc_oof_{name}.png', dpi=300); plt.close()
    print(f"  ✅ cm_oof_{name}.png و roc_oof_{name}.png ذخیره شد")

print("\n✅ همه کارها تموم شد.")