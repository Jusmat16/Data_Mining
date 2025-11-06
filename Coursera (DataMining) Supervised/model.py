# --- Cell 1: Imports ---
# Import library dasar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Import untuk Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import 3 Model Klasifikasi Berbeda
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Import Metrik Evaluasi
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# Mengatur agar plot terlihat bagus
sns.set(style="whitegrid")
warnings.filterwarnings("ignore")

print("Library untuk Supervised Learning (Klasifikasi) berhasil di-import.")


# --- Cell 2: Memuat dan Membersihkan Data ---
print("Memulai proses memuat data...")
try:
    df = pd.read_csv('SpotifyFeatures.csv', low_memory=False)
    print(f"Data berhasil dimuat. Jumlah baris awal: {len(df)}")
except FileNotFoundError:
    print("Error: File 'SpotifyFeatures.csv' tidak ditemukan.")
    # Kita buat dataframe dummy jika file tidak ada
    df = pd.DataFrame({
        'track_id': ['1', '2', '3', '4', '5', '6', '7', '8'],
        'name': ['Song A', 'Song B', 'Song C', 'Song D', 'Song E', 'Song F', 'Song G', 'Song H'],
        'popularity': [20, 80, 30, 75, 40, 90, 10, 60],
        'acousticness': [0.1, 0.8, 0.2, 0.7, 0.3, 0.9, 0.1, 0.5],
        'danceability': [0.8, 0.2, 0.7, 0.3, 0.6, 0.2, 0.9, 0.4],
        'energy': [0.9, 0.3, 0.8, 0.2, 0.7, 0.1, 0.8, 0.3],
        'instrumentalness': [0.0, 0.7, 0.1, 0.8, 0.0, 0.9, 0.0, 0.1],
        'liveness': [0.1, 0.3, 0.1, 0.4, 0.2, 0.5, 0.1, 0.2],
        'loudness': [-5.0, -10.0, -6.0, -9.0, -7.0, -12.0, -5.5, -9.5],
        'speechiness': [0.05, 0.4, 0.06, 0.3, 0.07, 0.3, 0.05, 0.06],
    })
    print("Menggunakan data dummy untuk demonstrasi.")

# Ambil sampel 20.000 data
if len(df) > 20000:
    df_sample = df.sample(n=20000, random_state=42)
    print(f"Mengambil sampel 20.000 lagu. Ukuran data: {len(df_sample)}")
else:
    df_sample = df.copy()
    print(f"Menggunakan semua data. Ukuran data: {len(df_sample)}")

# Pembersihan (sama seperti sebelumnya)
df_cleaned = df_sample.drop_duplicates(subset=['track_id'])
audio_features = [
    'acousticness', 'danceability', 'energy', 'instrumentalness', 
    'liveness', 'loudness', 'speechiness'
]
df_cleaned = df_cleaned.dropna(subset=audio_features + ['popularity'])
print(f"Data setelah dibersihkan (NaN/Duplikat): {len(df_cleaned)}")


# --- Cell 3: Feature Engineering (Target Variable) & Train-Test Split ---
print("Memulai Feature Engineering dan Split Data...")

# 1. Tentukan Fitur (X)
X = df_cleaned[audio_features]
print(f"Bentuk X (fitur): {X.shape}")

# 2. Tentukan Variabel Target (y)
# Kita definisikan "Populer" adalah lagu dengan skor popularitas > 65
# Anda bisa mengubah ambang batas (threshold) ini
POPULARITY_THRESHOLD = 65 
y = (df_cleaned['popularity'] > POPULARITY_THRESHOLD).astype(int) 
# y akan berisi 1 (jika populer) atau 0 (jika tidak populer)
print(f"Bentuk y (target): {y.shape}")

# 3. Cek Keseimbangan Kelas (Class Imbalance) - SANGAT PENTING
print("\n--- Distribusi Kelas Target ---")
print(y.value_counts(normalize=True))
print("---------------------------------")
# Ini akan menunjukkan apakah data kita tidak seimbang (misal: 90% tidak populer, 10% populer)
# Ini adalah 'kelemahan model' yang penting untuk laporan

# 4. Bagi Data (Train-Test Split)
# 80% data untuk melatih model, 20% untuk menguji model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Ukuran X_train: {X_train.shape}")
print(f"Ukuran X_test: {X_test.shape}")

# 5. Standardisasi (Scaling) - WAJIB
print("\nMelakukan Standardisasi (Scaling) pada data...")
scaler = StandardScaler()
# Fit HANYA pada data latih (X_train)
X_train_scaled = scaler.fit_transform(X_train)
# Transform pada data latih dan data uji
X_test_scaled = scaler.transform(X_test)

print("Persiapan data selesai.")


# --- Cell 4: Eksperimen 1 - Logistic Regression (Baseline) ---
print("\n--- Melatih Model 1: Logistic Regression ---")
# 'class_weight='balanced'' membantu menangani data tidak seimbang
log_reg = LogisticRegression(random_state=42, class_weight='balanced')
log_reg.fit(X_train_scaled, y_train)

# Prediksi pada data uji
y_pred_log = log_reg.predict(X_test_scaled)

# Evaluasi Model
print("Hasil Evaluasi Logistic Regression:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_log):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_log))


# --- Cell 5: Eksperimen 2 - Decision Tree Classifier ---
print("\n--- Melatih Model 2: Decision Tree ---")
# 'class_weight='balanced'' membantu menangani data tidak seimbang
tree_clf = DecisionTreeClassifier(random_state=42, class_weight='balanced', max_depth=10)
tree_clf.fit(X_train_scaled, y_train)

# Prediksi pada data uji
y_pred_tree = tree_clf.predict(X_test_scaled)

# Evaluasi Model
print("Hasil Evaluasi Decision Tree:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_tree):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_tree))


# --- Cell 6: Eksperimen 3 - Random Forest Classifier (Ensemble) ---
print("\n--- Melatih Model 3: Random Forest ---")
# 'class_weight='balanced'' membantu menangani data tidak seimbang
rf_clf = RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=100)
rf_clf.fit(X_train_scaled, y_train)

# Prediksi pada data uji
y_pred_rf = rf_clf.predict(X_test_scaled)

# Evaluasi Model
print("Hasil Evaluasi Random Forest:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_rf))


# --- Cell 7: Perbandingan Model, Temuan Kunci (Feature Importance) & Visualisasi ---
print("\n--- Analisis Hasil dan Temuan Kunci ---")

# Model Random Forest biasanya yang terkuat dan paling baik untuk 'feature importance'
# Kita akan gunakan model ini untuk "Temuan Kunci"

# 1. Temuan Kunci (Key Drivers / Feature Importance)
print("Mengekstrak Fitur Terpenting (Key Drivers) dari Random Forest...")
importances = rf_clf.feature_importances_
feature_names = X.columns

# Membuat DataFrame untuk visualisasi
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print(feature_importance_df)

# Plot Feature Importance
plt.figure(figsize=(12, 7))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
plt.title('Fitur Terpenting untuk Memprediksi Popularitas Lagu')
plt.xlabel('Tingkat Kepentingan (Importance)')
plt.ylabel('Fitur Audio')
plt.grid(True)
plt.savefig('feature_importance_plot.png')
print("\nPlot Feature Importance disimpan sebagai 'feature_importance_plot.png'.")

# 2. Visualisasi Model Terbaik (Confusion Matrix)
# Mari kita asumsikan Random Forest adalah model terbaik
print("Membuat Confusion Matrix untuk Model Random Forest...")
cm = confusion_matrix(y_test, y_pred_rf, labels=rf_clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Tidak Populer (0)', 'Populer (1)'])

fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(ax=ax, cmap='Blues')
plt.title('Confusion Matrix - Random Forest (K={POPULARITY_THRESHOLD})')
plt.savefig('confusion_matrix_plot.png')
print("Plot Confusion Matrix disimpan sebagai 'confusion_matrix_plot.png'.")

print("\n--- Analisis Selesai ---")
print("Anda sekarang memiliki:")
print("1. Metrik (Accuracy, F1-Score, dll) untuk 3 model berbeda.")
print("2. 'feature_importance_plot.png' - Untuk bagian 'Temuan Kunci'.")
print("3. 'confusion_matrix_plot.png' - Visual evaluasi model terbaik.")
