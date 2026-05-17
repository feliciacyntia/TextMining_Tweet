import re
import string
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# --- PENGATURAN HALAMAN (LAYOUT WIDE MAKSIMAL) ---
st.set_page_config(
    page_title="Analisis Sentimen Twitter",
    page_icon="🐦",
    layout="wide" 
)

# --- DOWNLOAD RESOURCE NLTK OTOMATIS ---
@st.cache_resource
def load_nltk_resources():
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)

load_nltk_resources()

# --- DEFINISI VARIABEL GLOBAL & SASTRAWI ---
NEGATION_WORDS = {'tidak', 'bukan', 'ga', 'gak', 'ngga', 'nggak', 'jangan', 'belum', 'tanpa'}
stop_words_id = set(stopwords.words('indonesian')) - NEGATION_WORDS

@st.cache_resource
def get_stemmer():
    factory = StemmerFactory()
    return factory.create_stemmer()

stemmer = get_stemmer()

# --- KAMUS SLANG LENGKAP NOTEBOOK ---
SLANG_DICT = {
    # --- Kata ganti ---
    'gue':'saya','gw':'saya','gua':'saya','aku':'saya',
    'sy':'saya','aq':'saya',
    'lo':'kamu','lu':'kamu','loe':'kamu','elo':'kamu',
    'km':'kamu','kmu':'kamu',
    'dy':'dia','doi':'dia','dya':'dia',
    'mrk':'mereka',
    # --- Konjungsi dan partikel Indonesia ---
    'yg':'yang','dgn':'dengan','dg':'dengan',
    'tp':'tapi','tpi':'tapi',
    'krn':'karena','karna':'karena','kalo':'kalau',
    'klu':'kalau','kl':'kalau',
    'utk':'untuk','utuk':'untuk','tuk':'untuk',
    'jg':'juga','jga':'juga',
    'aja':'saja','aj':'saja',
    'sm':'sama','ama':'sama',
    # --- Waktu ---
    'skrg':'sekarang','skg':'sekarang','skrang':'sekarang',
    'ntar':'nanti','tar':'nanti','nnti':'nanti',
    'kmrn':'kemarin',
    # --- Kata kerja & kondisi ---
    'mo':'mau',
    'bs':'bisa','bsa':'bisa',
    'hrs':'harus','hrus':'harus',
    'sdh':'sudah','udh':'sudah','udah':'sudah',
    'blm':'belum','blum':'belum',
    'lg':'lagi','lgi':'lagi',
    'dpt':'dapat','dapet':'dapat',
    'msh':'masih','masi':'masih',
    'pke':'pakai','pake':'pakai',
    'emg':'memang','emang':'memang',
    # --- Kata sifat & emosi ---
    'bgt':'banget','bangt':'banget','bgtt':'banget',
    'bgs':'bagus',
    'keren':'bagus','kerenn':'bagus','kren':'bagus',
    'mantap':'bagus','mantep':'bagus',
    'jos':'bagus','joss':'bagus',
    'jelek':'buruk',
    'parah':'buruk',
    'ancur':'buruk','hancur':'buruk',
    'kzl':'kesal','kesel':'kesal',
    'seneng':'senang','senengg':'senang',
    'happy':'senang','bahagia':'senang',
    'sad':'sedih',
    'susah':'sulit',
    'capek':'lelah','cape':'lelah',
    'cepet':'cepat',
    'lelet':'lambat','lemot':'lambat',
    # --- Ekspresi noise ---
    'wkwk':'','wkwkwk':'','haha':'','hahaha':'',
    'huhu':'','huhuh':'','wkwkwkwk':'',
    # --- Terima kasih ---
    'makasih':'terima kasih','makasi':'terima kasih',
    'mksh':'terima kasih','thx':'terima kasih',
    'thanks':'terima kasih','tq':'terima kasih',
    # --- Ekspresi umum Indonesia ---
    'nih':'ini',
    'tuh':'itu',
    'kyk':'seperti','kayak':'seperti','kyak':'seperti',
    'ngerasa':'merasa','kerasa':'terasa',
    'pengen':'ingin','pengin':'ingin','pgn':'ingin',
    'gitu':'begitu','gini':'begini',
    'knp':'kenapa',
    'ok':'oke','okey':'oke',
    'anj':'anjing','anjir':'anjing'
}

LABEL_MAP = {-1: 'negatif', 0: 'netral', 1: 'positif'}

# --- FUNGSI PREPROCESSING ---
def remove_emoji(text):
    emoji_pattern = re.compile(
        "[" u"\U0001F600-\U0001F64F" u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF" u"\U0001F1E0-\U0001F1FF"
        u"\U00002702-\U000027B0" u"\U000024C2-\U0001F251" "]+",
        flags=re.UNICODE)
    return emoji_pattern.sub('', text)

def remove_twitter_artifacts(text):
    text = re.sub(r'http\S+|https\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'\bRT\b', '', text)
    return text

def normalize_slang(text, slang_dict):
    words = text.split()
    normalized = [slang_dict.get(word, word) for word in words]
    return ' '.join(w for w in normalized if w.strip() != '')

def preprocess_tweet(text):
    if not isinstance(text, str) or not text.strip():
        return ''
    text = text.lower()                                      
    text = remove_emoji(text)                                
    text = remove_twitter_artifacts(text)                    
    text = re.sub(r'\d+', '', text)                          
    text = text.translate(str.maketrans('','',string.punctuation)) 
    text = normalize_slang(text, SLANG_DICT)                 
    text = re.sub(r'\s+', ' ', text).strip()                 
    tokens = word_tokenize(text)                             
    tokens = [t for t in tokens if len(t) > 1]               
    
    filtered_tokens = [w for w in tokens if w not in stop_words_id or w in NEGATION_WORDS]
    
    return ' '.join([stemmer.stem(w) for w in filtered_tokens])

# --- FUNGSI TRAINING MODEL ---
@st.cache_resource
def get_model_and_data(file_path):
    model_path = 'svm_model_3label.pkl'
    vec_path = 'tfidf_vectorizer_3label.pkl'
    
    df_raw = pd.read_csv(file_path, sep='\t')
    if df_raw.shape[1] == 1:
        col = df_raw.columns[0]
        if ';' in col:
            split_data = df_raw[col].str.split(';', n=1, expand=True)
            df_raw['sentimen'] = pd.to_numeric(split_data[0], errors='coerce')
            df_raw['Tweet'] = split_data[1]
        df_raw = df_raw[['sentimen', 'Tweet']]
    df_raw = df_raw.dropna(subset=['sentimen']).reset_index(drop=True)
    df_raw['sentimen'] = df_raw['sentimen'].astype(int)

    if os.path.exists(model_path) and os.path.exists(vec_path):
        model = joblib.load(model_path)
        vectorizer = joblib.load(vec_path)
        df_clean = df_raw.copy() 
        return model, vectorizer, df_raw, df_clean
    else:
        df_clean = df_raw.copy()
        df_clean['tweet_clean'] = df_clean['Tweet'].apply(preprocess_tweet)
        df_clean = df_clean[df_clean['tweet_clean'].str.strip() != ''].reset_index(drop=True)
        
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 1),
            max_features=10000,
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            analyzer='word',
            token_pattern=r'\b\w{2,}\b'
        )
        X = vectorizer.fit_transform(df_clean['tweet_clean'])
        y = df_clean['sentimen']
        
        model = LinearSVC(
            C=1.0,
            max_iter=5000,
            random_state=42,
            loss='squared_hinge',
            multi_class='ovr',
            class_weight='balanced'
        )
        model.fit(X, y)
        
        joblib.dump(model, model_path)
        joblib.dump(vectorizer, vec_path)
        
        return model, vectorizer, df_raw, df_clean

# --- REVISI NAMA FILE DATASET ---
dataset_name = 'Indonesian Sentiment Twitter Dataset Labeled.csv'

if not os.path.exists(dataset_name):
    st.error(f"⚠️ File dataset `{dataset_name}` tidak ditemukan. Pastikan file CSV dtaruh satu folder dengan script app.py ini.")
    st.stop()

with st.spinner("Sedang memproses basis data eksperimen, mohon tunggu..."):
    model, vectorizer, df_raw, df_clean = get_model_and_data(dataset_name)

# --- SIDEBAR CLEAN ---
st.sidebar.header("🔍 Navigasi & Sampel")

filter_label = st.sidebar.selectbox("Lihat contoh tweet dari dataset:", ["Pilih Label Sentimen", "Negatif", "Netral", "Positif"])
if filter_label != "Pilih Label Sentimen":
    label_code = {"Negatif": -1, "Netral": 0, "Positif": 1}[filter_label]
    samples = df_raw[df_raw['sentimen'] == label_code]['Tweet'].head(4).tolist()
    st.sidebar.write(f"**Contoh Teks {filter_label}:**")
    for i, sample in enumerate(samples, 1):
        st.sidebar.caption(f"{i}. \"{sample[:110]}...\"")

# --- TAMPILAN UTAMA FULL-WIDTH (SUDAH TIDAK DIBATASI KOLOM) ---
st.title("Analisis Sentimen Tweet Bahasa Indonesia")
st.write("Sistem klasifikasi sentimen 3 kelas (Positif, Netral, Negatif) menggunakan pendekatan ekstraksi fitur **TF-IDF Unigram** dan algoritma **Linear Support Vector Machine (LinearSVC)**. Silakan masukkan teks atau tweet pada kolom di bawah untuk melakukan pengujian secara realtime.")
st.write("---")

user_input = st.text_area("Masukkan kalimat atau tweet yang ingin dianalisis:", placeholder="Ketik di sini untuk menguji sentimen...")

if st.button("Analisis Sentimen", type="primary"):
    if not user_input.strip():
        st.warning("Silakan isi teks terlebih dahulu sebelum menekan tombol!")
    else:
        cleaned_text = preprocess_tweet(user_input)
        vec_text = vectorizer.transform([cleaned_text])
        
        pred_code = model.predict(vec_text)[0]
        
        decision_scores = model.decision_function(vec_text)[0]
        exp_scores = np.exp(decision_scores - np.max(decision_scores))
        probabilities = exp_scores / np.sum(exp_scores)
        
        prob_dict = dict(zip(model.classes_, probabilities))
        prob_neg = prob_dict.get(-1, 0.0) * 100
        prob_net = prob_dict.get(0, 0.0) * 100
        prob_pos = prob_dict.get(1, 0.0) * 100
        
        st.write("### Hasil Klasifikasi:")
        if pred_code == -1:
            st.error("🔴 **Sentimen Dominan: NEGATIF**")
        elif pred_code == 0:
            st.warning("⚪ **Sentimen Dominan: NETRAL**")
        else:
            st.success("🔵 **Sentimen Dominan: POSITIF**")
            
        st.write("#### Tingkat Keyakinan Model per Kelas:")
        
        st.write(f"🔵 **Positif:** {prob_pos:.1f}%")
        st.progress(prob_pos / 100)
        
        st.write(f"⚪ **Netral:** {prob_net:.1f}%")
        st.progress(prob_net / 100)
        
        st.write(f"🔴 **Negatif:** {prob_neg:.1f}%")
        st.progress(prob_neg / 100)
        st.write("")
            
        with st.expander("Lihat Hasil Pembersihan Teks (Preprocessing)"):
            st.write(f"**Teks Asli:** `{user_input}`")
            st.write(f"**Hasil Penyaringan Kata:** `{cleaned_text if cleaned_text else '[Teks kosong setelah difilter]'}`")

st.write("\n---")

# --- EXPANDER SEKARANG IKUT MELEBAR FULL KE KANAN ---
with st.expander("📊 Lihat Detail Distribusi Informasi Dataset", expanded=False):
    col_data1, col_data2 = st.columns(2)
    with col_data1:
        st.write("**Sampel 5 Data Teratas:**")
        st.dataframe(df_raw[['Tweet', 'sentimen']].head(5), use_container_width=True)
    with col_data2:
        st.write("**Komposisi Sentimen Dataset:**")
        counts = df_raw['sentimen'].map(LABEL_MAP).value_counts()
        st.bar_chart(counts)