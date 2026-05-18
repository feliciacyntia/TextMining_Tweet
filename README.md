# 📊 Analisis Sentimen Tweet Bahasa Indonesia
## Implementasi TF-IDF Unigram + Linear Support Vector Machine

Proyek ini membangun sistem klasifikasi sentimen otomatis tiga kelas (Negatif, Netral, Positif) terhadap data tweet berbahasa Indonesia dan campuran Melayu kasual. Proyek dilengkapi dengan pipeline preprocessing teks NLP yang ketat serta antarmuka pengujian interaktif berbasis Streamlit.

## Anggota Kelompok (Kelompok 10):
1. Keisha Hernantya Zahra (103052330063)
2. Felicia Cyntia Febriani (103052300086)

## Fitur Utama
* **Pipeline Preprocessing Modular:** Pembersihan teks terstruktur mulai dari cleaning text, normalisasi slang manual, penghapusan stopword, hingga stemming Sastrawi.
* **Optimasi Fitur TF-IDF:** Pengaturan hyperparameter pembobotan kata tingkat tinggi untuk mengontrol dimensi teks yang sparse.
* **Model LinearSVM Stabil:** Menggunakan strategi One-vs-Rest (OvR) dengan kompensasi bobot data tidak seimbang.
* **Analisis Kesalahan Komprehensif:** Modul Error Analysis untuk melacak bias dan pola salah tebak antar kelas.
* **Live Demo Streamlit:** Antarmuka pengujian real-time berbasis web untuk menguji teks kotor/noisy.

## Ringkasan Dataset & Distribusi Data
* **Sumber Data:** Indonesian_Sentiment_Twitter_Dataset_Labeled.csv
* **Data Awal:** ~10.806 tweet.
* **Data Bersih (Pasca-Deduplikasi):** 10.016 tweet (Didominasi oleh kelas Netral).
* **Data Latih (Train Set - 80%):** 8.012 tweet
* **Data Uji (Test Set - 20%):** 2.004 tweet
