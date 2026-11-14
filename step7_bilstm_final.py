import os, random
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

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

MAX_VOCAB=15000; MAX_LEN=50; EMBEDDING_DIM=300
BATCH_SIZE=128; EPOCHS=3; DROPOUT_RATE=0.4

df = pd.read_csv('train.csv')
df['label'] = df['class_label'].apply(lambda x: 1 if x == 'informative' else 0)
df = df.drop_duplicates(subset=['text']).reset_index(drop=True)
print(f"تعداد نمونه بعد از حذف تکراری: {len(df)}")

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

def preprocess(text):
    if pd.isna(text): return ""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join([w for w in text.split() if w not in stop_words])
    return re.sub(r'\s+', ' ', text).strip()

df['clean_text'] = df['text'].apply(preprocess)
texts = df['clean_text'].values
y = df['label'].values

print("بارگذاری GloVe...")
glove = api.load("glove-wiki-gigaword-300")

def build_matrix(tok):
    m = np.zeros((MAX_VOCAB, EMBEDDING_DIM))
    for word, i in tok.word_index.items():
        if i < MAX_VOCAB:
            try: m[i] = glove[word]
            except KeyError: pass
    return m

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
res = {'accuracy':[], 'precision':[], 'recall':[], 'f1':[], 'auc':[]}

for fold, (tr, te) in enumerate(skf.split(texts, y), 1):
    print(f"\n--- Fold {fold}/5 ---")
    tok = Tokenizer(num_words=MAX_VOCAB)
    tok.fit_on_texts(texts[tr])  # فقط روی train هر Fold

    X_tr = pad_sequences(tok.texts_to_sequences(texts[tr]), maxlen=MAX_LEN, padding='post')
    X_te = pad_sequences(tok.texts_to_sequences(texts[te]), maxlen=MAX_LEN, padding='post')

    emb = build_matrix(tok)
    model = Sequential([
        Embedding(MAX_VOCAB, EMBEDDING_DIM, weights=[emb], trainable=False),
        Bidirectional(LSTM(256, return_sequences=True)),
        Dropout(DROPOUT_RATE),
        Bidirectional(LSTM(128)),
        Dropout(DROPOUT_RATE),
        Dense(64, activation='relu'),
        Dropout(DROPOUT_RATE),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='RMSprop', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_tr, y[tr], epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1, verbose=1)

    p = model.predict(X_te, verbose=0).ravel()
    pred = (p >= 0.5).astype(int)

    res['accuracy'].append(accuracy_score(y[te], pred))
    res['precision'].append(precision_score(y[te], pred, zero_division=0))
    res['recall'].append(recall_score(y[te], pred, zero_division=0))
    res['f1'].append(f1_score(y[te], pred, zero_division=0))
    res['auc'].append(roc_auc_score(y[te], p))

    print(f"  Acc={res['accuracy'][-1]:.4f}, F1={res['f1'][-1]:.4f}")

print("\n" + "="*50)
print("GloVe + BiLSTM (بدون Data Leakage)")
print("="*50)
for k, v in res.items():
    print(f"{k}: {np.mean(v):.4f} ± {np.std(v):.4f}")

with open('results_bilstm_final.txt', 'w', encoding='utf-8') as f:
    f.write("GloVe + BiLSTM (بدون Data Leakage)\n")
    for k, v in res.items():
        f.write(f"{k}: {np.mean(v):.4f} ± {np.std(v):.4f}\n")
print("\n✅ ذخیره شد: results_bilstm_final.txt")