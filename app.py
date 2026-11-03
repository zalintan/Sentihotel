import os
import re
import ast
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# ── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SentiHotel — Analisis Sentimen Hotel Indonesia",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=Sora:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}
h1, h2, h3, .stTitle {
    font-family: 'Sora', sans-serif;
}

/* Hide Streamlit default elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display:none;}

/* Main background */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Header Hero */
.hero-container {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f2942 100%);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    border: 1px solid rgba(99, 179, 237, 0.2);
    position: relative;
    overflow: hidden;
}
.hero-container::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(99,179,237,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Sora', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: #f0f9ff;
    margin: 0 0 0.5rem 0;
    letter-spacing: -0.5px;
}
.hero-subtitle {
    color: #93c5fd;
    font-size: 1rem;
    margin: 0;
    font-weight: 400;
}
.hero-badge {
    display: inline-block;
    background: rgba(99,179,237,0.15);
    color: #93c5fd;
    border: 1px solid rgba(99,179,237,0.3);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin-bottom: 1rem;
    text-transform: uppercase;
}

/* Sentiment Cards */
.sentiment-card {
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin: 0.4rem 0;
    border-left: 4px solid;
    display: flex;
    align-items: center;
    gap: 0.8rem;
}
.sentiment-pos {
    background: rgba(16, 185, 129, 0.08);
    border-color: #10b981;
}
.sentiment-neg {
    background: rgba(239, 68, 68, 0.08);
    border-color: #ef4444;
}
.sentiment-neu {
    background: rgba(148, 163, 184, 0.08);
    border-color: #94a3b8;
}
.aspect-name {
    font-weight: 600;
    font-size: 0.95rem;
    color: #1e293b;
    min-width: 90px;
}
.sentiment-label {
    font-size: 0.85rem;
    font-weight: 500;
}
.prob-bar-bg {
    flex: 1;
    height: 6px;
    background: #e2e8f0;
    border-radius: 3px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.5s ease;
}

/* Info box */
.info-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin: 1rem 0;
}

/* Metric chip */
.metric-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #f1f5f9;
    border-radius: 8px;
    padding: 8px 16px;
    margin: 4px;
    font-size: 0.88rem;
    font-weight: 500;
    color: #334155;
}

/* Sidebar */
.sidebar-section {
    background: #f8fafc;
    border-radius: 10px;
    padding: 1rem;
    margin: 0.5rem 0;
    border: 1px solid #e2e8f0;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# KONFIGURASI & PREPROCESSING (sama persis seperti notebook)
# ══════════════════════════════════════════════════════════════════════════════

ASPECTS = ['Pelayanan', 'Kamar', 'Fasilitas', 'Makanan', 'Lokasi', 'Suasana', 'Harga']

# Label akhir yang ditampilkan ke user (dipakai di REVERSE_LABEL & COLOR_MAP)
# 0 = Negatif, 1 = Tidak Dibahas, 2 = Positif
REVERSE_LABEL = {0: 'Negatif', 1: 'Tidak Dibahas', 2: 'Positif'}

panjang_maksimal = 100

# ── FOLDER STRUKTUR (1 folder model, 1 folder tokenizer — isi 2 set file) ──
# Tahap 1 (Deteksi Aspek)   : saved_models/t1_deteksi_{slug}.keras
#                              saved_tokenizers/tokenizer_t1_{slug}.json
# Tahap 2 (Sentimen Polar.) : saved_models/t2_sentimen_{slug}.keras
#                              saved_tokenizers/tokenizer_t2_{slug}.json
MODEL_DIR      = "models" # satu folder berisi file t1_ dan t2_
TOKENIZER_DIR  = "tokenizers" # satu folder berisi file tokenizer_t1_ dan tokenizer_t2_

NORM_DICT = {
    'yg': 'yang', 'dgn': 'dengan', 'dg': 'dengan', 'utk': 'untuk',
    'krn': 'karena', 'karna': 'karena', 'krna': 'karena', 'sdh': 'sudah',
    'udh': 'sudah', 'udah': 'sudah', 'bs': 'bisa', 'bsa': 'bisa',
    'sy': 'saya', 'aku': 'saya', 'jgn': 'jangan', 'kl': 'kalau',
    'klo': 'kalau', 'kalo': 'kalau', 'klu': 'kalau', 'dtg': 'datang',
    'lg': 'lagi', 'lgi': 'lagi', 'tp': 'tapi', 'tpi': 'tapi',
    'hrs': 'harus', 'msh': 'masih', 'msih': 'masih', 'bgt': 'banget',
    'bngt': 'banget', 'bgtt': 'banget',
    'aja': 'saja', 'aj': 'saja', 'deh': '', 'dong': '', 'sih': '',
    'mah': '', 'kok': '', 'sich': '', 'doong': '',
    'ga': 'tidak', 'gak': 'tidak', 'ngak': 'tidak', 'nggak': 'tidak',
    'ngga': 'tidak', 'gk': 'tidak', 'tak': 'tidak', 'tdk': 'tidak',
    'enggak': 'tidak', 'nggk': 'tidak', 'bkn': 'bukan', 'blm': 'belum',
    'blum': 'belum', 'blom': 'belum', 'belom': 'belum',
    'emg': 'memang', 'emang': 'memang', 'kayak': 'seperti', 'kyk': 'seperti',
    'gimana': 'bagaimana', 'gmn': 'bagaimana', 'gitu': 'begitu',
    'gini': 'begini', 'gt': 'begitu', 'abis': 'habis', 'smpe': 'sampai',
    'ampe': 'sampai', 'sampe': 'sampai', 'cuma': 'hanya', 'cm': 'hanya',
    'krg': 'kurang', 'kurg': 'kurang', 'sgt': 'sangat', 'bener': 'benar',
    'bner': 'benar', 'lbh': 'lebih', 'lbih': 'lebih',
    'staffnya': 'staf', 'stafnya': 'staf', 'staffs': 'staf',
    'chekout': 'checkout', 'checkin': 'check in',
    'ac': 'pendingin ruangan', 'wfi': 'wifi', 'wfe': 'wifi',
    'good': 'bagus', 'nice': 'bagus', 'great': 'bagus', 'best': 'terbaik',
    'clean': 'bersih', 'ok': 'oke', 'okay': 'oke', 'overall': 'semua',
    'harga_jutaan': 'harga mahal', 'harga_ribuan': 'harga terjangkau',
    'harga_nominal': 'harga',
}

STOPWORDS = {
    'saya', 'kami', 'kita', 'mereka', 'dia', 'anda', 'kamu', 'ia',
    'di', 'ke', 'dari', 'pada', 'dalam', 'untuk', 'dan', 'atau', 'juga',
    'dengan', 'karena', 'seperti', 'sehingga', 'oleh', 'tentang', 'bahwa',
    'hingga', 'sampai', 'sejak', 'antara', 'terhadap', 'selama', 'setelah',
    'sebelum', 'ketika', 'agar', 'supaya', 'walau', 'walaupun', 'meski',
    'meskipun', 'apabila', 'bagi', 'kalau', 'tapi', 'namun',
    'ini', 'itu', 'sini', 'situ', 'sana', 'tersebut',
    'adalah', 'ialah',
    'akan', 'sudah', 'telah', 'bisa', 'dapat', 'boleh', 'harus',
    'perlu', 'mau', 'ingin', 'punya', 'sedang', 'masih', 'lagi',
    'pun', 'lah', 'kah', 'ya', 'yah', 'oh', 'ah',
    'apa', 'siapa', 'mana', 'kapan', 'kenapa', 'mengapa', 'bagaimana',
    'yg', 'yang', 'nya', 'ada',
}

# ── Preprocessing Functions ──────────────────────────────────────────────────

def case_folding(text):
    return text.lower()

def text_cleaning(text):
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'#\w+|@\w+', '', text)
    text = re.sub(r'rp\.?\s*\d+[\.,]?\d*', ' harga nominal ', text, flags=re.IGNORECASE)
    text = re.sub(r'\d+\s*(juta|jt)\b', ' harga mahal ', text, flags=re.IGNORECASE)
    text = re.sub(r'\d+\s*(ribu|rb|k)\b', ' harga terjangkau ', text, flags=re.IGNORECASE)
    bintang_map = {'1': 'sangat jelek', '2': 'jelek', '3': 'standar', '4': 'bagus', '5': 'mewah'}
    text = re.sub(
        r'bintang\s*(\d)',
        lambda m: f" {bintang_map.get(m.group(1), 'hotel berbintang')} ",
        text, flags=re.IGNORECASE
    )
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\b\d+\b', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenization(text):
    return text.split()

def normalization(tokens):
    normalized = [NORM_DICT.get(word, word) for word in tokens]
    return [w for w in normalized if w.strip()]

def stopword_removal(tokens):
    return [t for t in tokens if t not in STOPWORDS]

def join_tokens(tokens):
    if isinstance(tokens, str):
        try:
            tokens = ast.literal_eval(tokens)
        except (ValueError, SyntaxError):
            pass
    if isinstance(tokens, list):
        return ' '.join(tokens)
    return str(tokens)

def preprocess_input(teks):
    teks = case_folding(teks)
    teks = text_cleaning(teks)
    tokens = tokenization(teks)
    tokens = normalization(tokens)
    tokens = stopword_removal(tokens)
    return join_tokens(tokens)

def aspect_slug(aspek):
    return aspek.lower().replace(' ', '_')


# ── Load Models (cached) ─────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_models():
    """
    Load semua model keras dan tokenizer dari disk.

    Struktur file (satu folder model, satu folder tokenizer):
        saved_models/
            t1_deteksi_{slug}.keras      ← Tahap 1: Tidak Dibahas vs Dibahas (binary)
            t2_sentimen_{slug}.keras     ← Tahap 2: Negatif vs Positif (binary)
        saved_tokenizers/
            tokenizer_t1_{slug}.json     ← Tokenizer Tahap 1
            tokenizer_t2_{slug}.json     ← Tokenizer Tahap 2
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        from tensorflow.keras.preprocessing.text import tokenizer_from_json

        class AttentionLayer(tf.keras.layers.Layer):
            def __init__(self, **kwargs):
                super(AttentionLayer, self).__init__(**kwargs)
        
            def build(self, input_shape):
                self.W = self.add_weight(
                    name='attention_weight',
                    shape=(input_shape[-1], 1),
                    initializer='glorot_uniform',   
                    trainable=True
                )
                self.b = self.add_weight(
                    name='attention_bias',
                    shape=(1,),                    
                    initializer='zeros',
                    trainable=True
                )
                super(AttentionLayer, self).build(input_shape)
        
            def call(self, x):
                e = tf.keras.activations.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
                a = tf.keras.activations.softmax(e, axis=1)
                return tf.reduce_sum(x * a, axis=1)

            def compute_output_shape(self, input_shape):
                return (input_shape[0], input_shape[-1])

            def get_config(self):
                return super(AttentionLayer, self).get_config()

        models_t1, tokenizers_t1 = {}, {}
        models_t2, tokenizers_t2 = {}, {}
        missing = []

        for aspek in ASPECTS:
            slug = aspect_slug(aspek)

            # ── Tahap 1 (Deteksi) ──────────────────────────────────────
            m1_path = f"{MODEL_DIR}/t1_deteksi_{slug}.keras"
            tk1_path = f"{TOKENIZER_DIR}/tokenizer_t1_{slug}.json"

            if os.path.exists(m1_path) and os.path.exists(tk1_path):
                models_t1[aspek] = load_model(
                    m1_path, custom_objects={'AttentionLayer': AttentionLayer},
                    compile=False
                )
                with open(tk1_path, 'r', encoding='utf-8') as f:
                    tokenizers_t1[aspek] = tokenizer_from_json(f.read())
            else:
                missing.append(f"T1/{aspek}")

            # ── Tahap 2 (Sentimen) ─────────────────────────────────────
            m2_path = f"{MODEL_DIR}/t2_sentimen_{slug}.keras"
            tk2_path = f"{TOKENIZER_DIR}/tokenizer_t2_{slug}.json"

            if os.path.exists(m2_path) and os.path.exists(tk2_path):
                models_t2[aspek] = load_model(
                    m2_path, custom_objects={'AttentionLayer': AttentionLayer},
                    compile=False
                )
                with open(tk2_path, 'r', encoding='utf-8') as f:
                    tokenizers_t2[aspek] = tokenizer_from_json(f.read())
            else:
                missing.append(f"T2/{aspek}")

        if missing:
            return models_t1, tokenizers_t1, models_t2, tokenizers_t2, \
                   f"File tidak ditemukan: {', '.join(missing)}"

        return models_t1, tokenizers_t1, models_t2, tokenizers_t2, None

    except ImportError as e:
        return {}, {}, {}, {}, f"TensorFlow tidak terinstall: {e}"
    except Exception as e:
        return {}, {}, {}, {}, str(e)


# ══════════════════════════════════════════════════════════════════════════════
# ATTENTION WEIGHT EXTRACTION (TANPA retrain / TANPA ubah arsitektur)
# ══════════════════════════════════════════════════════════════════════════════
# Ide dasarnya: AttentionLayer yang sudah dilatih (method .call()) menghitung
# bobot attention `a` di dalam dirinya sendiri, tapi cuma me-return context
# vector (hasil weighted-sum), bukan `a`-nya. Supaya `a` bisa ditampilkan tanpa
# menyentuh arsitektur/bobot yang sudah dilatih, kita:
#   1) Bikin sub-model baru yang outputnya adalah tensor SEBELUM masuk ke
#      Attention layer (reuse layer & bobot yang sama persis, tidak ada
#      layer/bobot baru yang dibuat atau dilatih).
#   2) Ambil W & b yang SUDAH dilatih dari layer Attention (`get_weights()`).
#   3) Hitung ulang e = tanh(x·W + b) dan a = softmax(e) secara manual pakai
#      numpy — rumus ini PERSIS sama dengan yang ada di AttentionLayer.call().
# Sudah divalidasi: hasil numpy ini identik (selisih ~1e-9) dengan hasil
# AttentionLayer asli yang dipanggil langsung lewat TensorFlow.

def get_attention_weights(model, padded_sequence):
    """
    Bobot attention (softmax, sepanjang panjang_maksimal) untuk satu input
    yang sudah di-pad, dari sebuah model yang sudah dilatih & di-load.
    """
    import tensorflow as tf

    attention_layer = None
    for layer in model.layers:
        if layer.__class__.__name__ == 'AttentionLayer':
            attention_layer = layer
            break
    if attention_layer is None:
        raise ValueError("Layer AttentionLayer tidak ditemukan pada model ini.")

    # Tensor yang masuk ke Attention layer = output BiLSTM, apapun namanya
    sub_model = tf.keras.Model(inputs=model.input, outputs=attention_layer.input)
    pre_attention = sub_model.predict(padded_sequence, verbose=0)  # (1, T, hidden)

    W, b = attention_layer.get_weights()  # W: (hidden, 1), b: (1,)

    e = np.tanh(np.tensordot(pre_attention, W, axes=1) + b)  # (1, T, 1)
    e = e[..., 0]                                             # (1, T)
    e = e - np.max(e, axis=1, keepdims=True)                  # stabilitas numerik
    exp_e = np.exp(e)
    a = exp_e / np.sum(exp_e, axis=1, keepdims=True)          # softmax, sama seperti call()

    return a[0]  # (T,)


def decode_sequence_to_words(tokenizer, seq):
    """Petakan id token -> kata asli sesuai kosakata tokenizer (tanpa menebak)."""
    index_word = tokenizer.index_word
    return [index_word.get(idx, '?') for idx in seq if idx != 0]


def get_word_attention(model, padded_sequence, tokenizer, raw_seq, maxlen=panjang_maksimal):
    """
    Bobot attention untuk kata-kata ASLI dalam satu ulasan (padding dibuang).
    Karena padding di sini pakai padding='post', kata-kata asli selalu berada
    di posisi awal, jadi tinggal dipotong sepanjang jumlah kata asli.
    Dinormalisasi ulang supaya jumlah bobot antar-kata = 1 (model dilatih
    tanpa attention masking, jadi posisi padding ikut kebagian sedikit bobot;
    ini dibuang lalu sisanya dinormalisasi ulang khusus untuk tampilan).
    """
    try:
        raw_seq = raw_seq[:maxlen]  # samakan dgn efek truncating='post'
        full_weights = get_attention_weights(model, padded_sequence)
        words = decode_sequence_to_words(tokenizer, raw_seq)
        n = len(words)
        word_weights = full_weights[:n]
        total = float(word_weights.sum())
        if total > 0:
            word_weights = word_weights / total
        return words, word_weights.tolist()
    except Exception:
        return [], []


def attention_html(words, weights, hex_color="#6366f1"):
    """Render kata-kata dengan warna latar sesuai bobot attention-nya."""
    if not words:
        return "<i style='color:#94a3b8; font-size:0.85rem'>Tidak ada kata untuk ditampilkan.</i>"
    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
    w_arr = np.array(weights, dtype=float)
    w_max = w_arr.max() if w_arr.max() > 0 else 1.0
    spans = []
    for word, w in zip(words, w_arr / w_max):
        alpha = 0.10 + 0.70 * float(w)
        spans.append(
            f'<span style="background: rgba({r},{g},{b},{alpha:.2f}); '
            f'padding: 3px 7px; border-radius: 6px; margin: 2px; display:inline-block; '
            f'font-size:0.9rem; color:#1e293b">{word}</span>'
        )
    return '<div style="line-height:2.3">' + ' '.join(spans) + '</div>'


def plot_attention_bar(words, weights, hex_color="#6366f1"):
    """Bar chart bobot attention per kata — versi presisi utk pelengkap highlight teks."""
    fig, ax = plt.subplots(figsize=(max(4, len(words) * 0.5), 2.4))
    x = np.arange(len(words))
    ax.bar(x, weights, color=hex_color, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(words, rotation=40, ha='right', fontsize=8)
    ax.set_ylabel('Bobot', fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_facecolor('#f8fafc')
    fig.patch.set_facecolor('#f8fafc')
    plt.tight_layout()
    return fig


# ── Prediction Function ──────────────────────────────────────────────────────

def predict_review(teks_asli, models_t1, tokenizers_t1, models_t2, tokenizers_t2):
    """
    Pipeline 2-tahap:
      Tahap 1 → apakah aspek dibahas? (0=Tidak Dibahas, 1=Dibahas)
      Tahap 2 → kalau dibahas, apa sentimennya? (0=Negatif, 1=Positif)

    Label akhir yang dikembalikan mengikuti REVERSE_LABEL:
      0 = Negatif, 1 = Tidak Dibahas, 2 = Positif

    Probabilitas komposit (jumlah = 1):
      prob_neu = P(Tidak Dibahas dari T1)
      prob_neg = P(Dibahas dari T1) × P(Negatif dari T2)
      prob_pos = P(Dibahas dari T1) × P(Positif dari T2)

    Setiap hasil aspek juga menyertakan 'attention_words' & 'attention_weights':
    bobot attention per kata dari layer Attention yang SUDAH dilatih (lihat
    get_word_attention), dipakai untuk menampilkan kata mana yang paling
    berpengaruh terhadap keputusan model di tiap aspek.
    """
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    teks_bersih = preprocess_input(teks_asli)
    results = {}

    for aspek in ASPECTS:
        # Kedua model & tokenizer harus tersedia
        if aspek not in models_t1 or aspek not in models_t2:
            continue

        # ── Tahap 1: Deteksi Aspek ──────────────────────────────────────
        tok1 = tokenizers_t1[aspek]
        seq1 = tok1.texts_to_sequences([teks_bersih])
        pad1 = pad_sequences(seq1, maxlen=panjang_maksimal, padding='post', truncating='post')
        prob_t1 = models_t1[aspek].predict(pad1, verbose=0)[0]
        # prob_t1[0] = P(Tidak Dibahas), prob_t1[1] = P(Dibahas)
        p_tidak_dibahas = float(prob_t1[0])
        p_dibahas       = float(prob_t1[1])
        t1_pred         = int(np.argmax(prob_t1))   # 0 atau 1

        if t1_pred == 0:
            # Aspek tidak dibahas — tidak perlu lanjut ke T2
            att_words, att_weights = get_word_attention(models_t1[aspek], pad1, tok1, seq1[0])
            results[aspek] = {
                'label':     1,              # 1 = Tidak Dibahas (sesuai REVERSE_LABEL)
                'label_str': 'Tidak Dibahas',
                'prob_neg':  0.0,
                'prob_neu':  p_tidak_dibahas,
                'prob_pos':  0.0,
                # Detail T1 (opsional, untuk debugging)
                'p_dibahas': p_dibahas,
                'attention_stage':   'Tahap 1 — Deteksi Aspek',
                'attention_words':   att_words,
                'attention_weights': att_weights,
            }
        else:
            # ── Tahap 2: Klasifikasi Sentimen ───────────────────────────
            tok2 = tokenizers_t2[aspek]
            seq2 = tok2.texts_to_sequences([teks_bersih])
            pad2 = pad_sequences(seq2, maxlen=panjang_maksimal, padding='post', truncating='post')
            prob_t2 = models_t2[aspek].predict(pad2, verbose=0)[0]
            # prob_t2[0] = P(Negatif), prob_t2[1] = P(Positif)
            t2_pred = int(np.argmax(prob_t2))

            # Probabilitas komposit
            prob_neg = p_dibahas * float(prob_t2[0])
            prob_pos = p_dibahas * float(prob_t2[1])

            # Label akhir: 0=Negatif, 2=Positif (sesuai REVERSE_LABEL)
            final_label = 0 if t2_pred == 0 else 2

            att_words, att_weights = get_word_attention(models_t2[aspek], pad2, tok2, seq2[0])
            results[aspek] = {
                'label':     final_label,
                'label_str': REVERSE_LABEL[final_label],
                'prob_neg':  prob_neg,
                'prob_neu':  p_tidak_dibahas,
                'prob_pos':  prob_pos,
                'p_dibahas': p_dibahas,
                'attention_stage':   'Tahap 2 — Sentimen',
                'attention_words':   att_words,
                'attention_weights': att_weights,
            }

    return results, teks_bersih


# ══════════════════════════════════════════════════════════════════════════════
# UI LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

# ── Hero Header ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-container">
  <div class="hero-badge">🏨 NLP · Analisis Sentimen · 2 Tahap</div>
  <div class="hero-title">SentiHotel Indonesia</div>
  <p class="hero-subtitle">
    Analisis sentimen ulasan hotel secara otomatis menggunakan pipeline 2-tahap BiLSTM + Attention
    dengan FastText embedding Bahasa Indonesia — 7 aspek sekaligus.
  </p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Pengaturan")

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    show_proba        = st.toggle("Tampilkan probabilitas", value=True)
    show_preprocessed = st.toggle("Tampilkan teks setelah preprocessing", value=False)
    show_chart        = st.toggle("Tampilkan grafik probabilitas", value=True)
    show_attention    = st.toggle("Tampilkan bobot attention", value=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📋 Aspek yang Dianalisis")
    aspect_icons = {
        'Pelayanan': '👥', 'Kamar': '🛏️', 'Fasilitas': '🏊',
        'Makanan': '🍽️', 'Lokasi': '📍', 'Suasana': '✨', 'Harga': '💰'
    }
    for asp in ASPECTS:
        st.markdown(f"{aspect_icons.get(asp, '•')} **{asp}**")

    st.markdown("---")
    st.markdown("### ℹ️ Info Model")
    st.markdown("""
    **Pipeline 2 Tahap:**
    - **Tahap 1** — Deteksi Aspek  
      *(Tidak Dibahas vs Dibahas)*  
    - **Tahap 2** — Sentimen Polaritas  
      *(Negatif vs Positif)*  

    ---
    - **Arsitektur**: BiLSTM + Attention  
    - **Embedding**: FastText ID  
    """)

    st.markdown("---")
    st.markdown("### 📁 Struktur Folder")
    st.code("""\
📁 models/
   ├── t1_deteksi_pelayanan.keras
   ├── t2_sentimen_pelayanan.keras
   └── ... (×7 aspek)
📁 tokenizers/
   ├── tokenizer_t1_pelayanan.json
   ├── tokenizer_t2_pelayanan.json
   └── ... (×7 aspek)""", language=None)

# ── Main Content ─────────────────────────────────────────────────────────────
col_input, col_result = st.columns([1, 1], gap="large")

with col_input:
    st.markdown("#### ✍️ Masukkan Ulasan Hotel")

    contoh = {
        "Pilih contoh ulasan...": "",
        "⭐ Ulasan Campuran":   "air kolam renangnya kotor, sarapannya terlalu asin, tapi kasurnya empuk dan nyaman, juga lokasinya dekat kemana mana. pemandangannya bagus, stafnya jutek",
        "✅ Ulasan Positif":    "sarapannya super enak, sangat worth it karena harga murah tapi bisa dapat kamar yang bagus dan fasilitas lengkap",
        "❌ Ulasan Negatif":    "kamarnya kotor, suasananya horor, dan stafnya tidak ramah. sangat mengecewakan",
        "🔀 Ulasan Detail":     "pelayanan cepat tanggap, sarapannya enak sekali, kamarnya luas dan fasilitas mewah. untuk lokasinya memang jauh dari pusat kota jadi aksesnya sulit, juga harganya lumayan mahal tapi dari kamar bisa liat pemandangan yang indah",
    }

    selected     = st.selectbox("Atau pilih contoh:", list(contoh.keys()))
    default_text = contoh[selected]

    user_input = st.text_area(
        "Teks ulasan:",
        value=default_text,
        height=150,
        placeholder="Contoh: Kamarnya sangat bersih dan nyaman, staf ramah, tapi makanannya biasa saja...",
        label_visibility="collapsed"
    )

    analyze_btn = st.button("🔍 Analisis Sentimen", type="primary", use_container_width=True)

    if not user_input:
        st.info("💡 Ketik ulasan hotel dalam Bahasa Indonesia, lalu klik **Analisis Sentimen**.")

with col_result:
    st.markdown("#### 📊 Hasil Analisis")

    if not analyze_btn or not user_input.strip():
        st.markdown("""
        <div class="info-box" style="text-align:center; color:#94a3b8; padding: 3rem;">
            <div style="font-size:3rem">🏨</div>
            <div style="margin-top:1rem; font-size:0.95rem">
                Hasil analisis akan muncul di sini<br>setelah kamu mengklik <strong>Analisis Sentimen</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Load models
        with st.spinner("Memuat model..."):
            models_t1, tokenizers_t1, models_t2, tokenizers_t2, error = load_models()

        if error and not models_t1:
            st.error(f"❌ Error memuat model: {error}")
            st.info("Pastikan folder `saved_models/` dan `saved_tokenizers/` ada dan berisi file yang tepat.")
        elif not models_t1 and not models_t2:
            st.warning("⚠️ Tidak ada model yang ditemukan. Pastikan struktur folder sudah benar:")
            st.code("""\
📁 saved_models/
   ├── t1_deteksi_pelayanan.keras
   ├── t2_sentimen_pelayanan.keras
   └── ... (×7 aspek)
📁 saved_tokenizers/
   ├── tokenizer_t1_pelayanan.json
   ├── tokenizer_t2_pelayanan.json
   └── ... (×7 aspek)""")
        else:
            # Tampilkan warning parsial jika ada aspek yang file-nya tidak lengkap
            if error:
                st.warning(f"⚠️ Beberapa file tidak ditemukan: {error}")

            with st.spinner("Menganalisis ulasan (2 tahap)..."):
                results, teks_bersih = predict_review(
                    user_input, models_t1, tokenizers_t1, models_t2, tokenizers_t2
                )
            with col_input:
                # Probability chart
                if show_chart and results:
                    st.markdown("---")
                    st.markdown("**📈 Distribusi Probabilitas per Aspek**")
    
                    fig, ax = plt.subplots(figsize=(8, 4))
                    aspek_list = [a for a in ASPECTS if a in results]
                    x = np.arange(len(aspek_list))
                    w = 0.25
    
                    neg_vals = [results[a]['prob_neg'] for a in aspek_list]
                    neu_vals = [results[a]['prob_neu'] for a in aspek_list]
                    pos_vals = [results[a]['prob_pos'] for a in aspek_list]
    
                    ax.bar(x - w, neg_vals, w, label='Negatif',       color='#ef4444', alpha=0.85)
                    ax.bar(x,     neu_vals, w, label='Tidak Dibahas', color='#94a3b8', alpha=0.85)
                    ax.bar(x + w, pos_vals, w, label='Positif',       color='#10b981', alpha=0.85)
    
                    ax.set_xticks(x)
                    ax.set_xticklabels(aspek_list, fontsize=9)
                    ax.set_ylim(0, 1.1)
                    ax.set_ylabel('Probabilitas', fontsize=9)
                    ax.legend(fontsize=8, loc='upper right')
                    ax.spines[['top', 'right']].set_visible(False)
                    ax.set_facecolor('#f8fafc')
                    fig.patch.set_facecolor('#f8fafc')
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)

            if show_preprocessed:
                with st.expander("🔧 Teks setelah preprocessing"):
                    st.code(teks_bersih, language=None)

            # Summary chips
            pos_count = sum(1 for r in results.values() if r['label'] == 2)
            neg_count = sum(1 for r in results.values() if r['label'] == 0)
            neu_count = sum(1 for r in results.values() if r['label'] == 1)

            st.markdown(
                f'<div style="margin-bottom:1rem">'
                f'<span class="metric-chip">✅ Positif: {pos_count}</span>'
                f'<span class="metric-chip">❌ Negatif: {neg_count}</span>'
                f'<span class="metric-chip">➖ Tidak Dibahas: {neu_count}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

            # Per-aspect results
            COLOR_MAP = {
                2: ('#10b981', 'sentiment-pos', '✅'),   # Positif
                0: ('#ef4444', 'sentiment-neg', '❌'),   # Negatif
                1: ('#94a3b8', 'sentiment-neu', '➖'),   # Tidak Dibahas
            }

            for aspek in ASPECTS:
                if aspek not in results:
                    continue
                r = results[aspek]
                color, css_class, icon = COLOR_MAP[r['label']]
                conf = max(r['prob_neg'], r['prob_neu'], r['prob_pos'])

                st.markdown(f"""
                <div class="sentiment-card {css_class}">
                  <span style="font-size:1.3rem">{aspect_icons.get(aspek, '•')}</span>
                  <span class="aspect-name">{aspek}</span>
                  <span class="sentiment-label" style="color:{color}; min-width:120px">
                    {icon} {r['label_str']}
                  </span>
                  {'<span style="color:#94a3b8; font-size:0.8rem">(' + f"{conf:.0%}" + ')</span>' if show_proba else ''}
                </div>
                """, unsafe_allow_html=True)

                if show_attention and r.get('attention_words'):
                    with st.expander(f"🔍 Bobot attention — {r['attention_stage']}"):
                        st.caption(
                            "Warna lebih pekat = bobot attention lebih besar "
                            "(kata itu lebih berpengaruh ke keputusan model). "
                            "Kata di bawah adalah teks setelah preprocessing."
                        )
                        st.markdown(
                            attention_html(r['attention_words'], r['attention_weights'], hex_color=color),
                            unsafe_allow_html=True
                        )
                        fig_att = plot_attention_bar(r['attention_words'], r['attention_weights'], hex_color=color)
                        st.pyplot(fig_att, use_container_width=True)
                        plt.close(fig_att)

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#94a3b8; font-size:0.82rem'>"
    "SentiHotel · Pipeline 2-Tahap · BiLSTM + Attention · FastText ID · Dibangun dengan Streamlit"
    "</div>",
    unsafe_allow_html=True
)
