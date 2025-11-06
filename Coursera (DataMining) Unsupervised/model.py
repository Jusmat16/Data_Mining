import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings

# Mengatur agar plot terlihat bagus
sns.set(style="whitegrid")
warnings.filterwarnings("ignore")

print("Memulai proses analisis...")

# --- STEP 3: EKSPLORASI, PEMBERSIHAN, & PERSIAPAN DATA ---

# 1. Memuat Data
# Ganti 'tracks.csv' dengan nama file Anda jika berbeda
try:
    # Menggunakan 'low_memory=False' terkadang membantu saat membaca file CSV besar/kompleks
    df = pd.read_csv('SpotifyFeatures.csv', low_memory=False) 
    print(f"Data berhasil dimuat. Jumlah baris awal: {len(df)}")
except FileNotFoundError:
    print("Error: File 'SpotifyFeatures.csv' tidak ditemukan.")

# 2. Pembersihan Data
# Menghapus duplikat berdasarkan 'track_id' (sesuai gambar)
df_cleaned = df.drop_duplicates(subset=['track_id'])
print(f"Data setelah hapus duplikat: {len(df_cleaned)}")

# 3. Pemilihan Fitur (Feature Selection)
# **DISESUAIKAN** berdasarkan kolom di gambar Anda
audio_features = [
    'acousticness', 'danceability', 'energy', 'instrumentalness', 
    'liveness', 'loudness', 'speechiness' 
]

# Menghapus baris yang memiliki nilai NaN di kolom fitur audio
df_cleaned = df_cleaned.dropna(subset=audio_features)
print(f"Data setelah hapus NaN: {len(df_cleaned)}")

# Menyimpan data fitur audio dalam variabel baru
X = df_cleaned[audio_features]
# Menyimpan info lagu untuk analisis akhir
track_info = df_cleaned[['track_name', 'artist_name', 'genre']] 

print(f"Menggunakan {len(audio_features)} fitur audio untuk analisis.")

# 4. Standardisasi Fitur (Feature Scaling)
# WAJIB untuk PCA dan K-Means
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Data berhasil dibersihkan dan distandardisasi.")

# --- STEP 4: EKSPERIMEN MODEL (PCA & CLUSTERING) ---

# --- A. Dimension Reduction (PCA) ---
# Kita reduksi 7 fitur audio menjadi 2 "Komponen Utama" untuk visualisasi
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Menampilkan seberapa banyak variasi yang dijelaskan oleh 2 komponen ini
print(f"PCA - 2 Komponen menjelaskan {pca.explained_variance_ratio_.sum() * 100:.2f}% variasi data.")

# Membuat DataFrame baru dari hasil PCA untuk plotting
df_pca = pd.DataFrame(data=X_pca, columns=['PCA_1', 'PCA_2'])

# --- B. Clustering (K-Means) & Eksperimen ---

# 1. Eksperimen 1: Elbow Method untuk mencari K
print("Menjalankan Eksperimen 1: Elbow Method (mencari K)...")
wcss = [] # Within-Cluster Sum of Squares
k_range = range(1, 11)

for k in k_range:
    kmeans_elbow = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    kmeans_elbow.fit(X_pca)
    wcss.append(kmeans_elbow.inertia_)

# Plot Elbow Method
plt.figure(figsize=(10, 6))
plt.plot(k_range, wcss, marker='o', linestyle='--')
plt.title('Elbow Method untuk Menentukan K Optimal')
plt.xlabel('Jumlah Cluster (K)')
plt.ylabel('WCSS (Inertia)')
plt.grid(True)
plt.savefig('elbow_plot.png')
print("Plot Elbow Method disimpan sebagai 'elbow_plot.png'.")
print("Lihat plot 'elbow_plot.png' untuk menentukan 'siku' (elbow) secara visual. Misal kita pilih K=5.")

# 2. Eksperimen 2, 3, 4: Melatih 3 Variasi Model K-Means
# Berdasarkan Elbow Method, kita pilih 3 nilai K di sekitar "siku"
K_VARIATIONS = [4, 5, 6] # Anda bisa ganti ini berdasarkan plot elbow Anda
models = {}
silhouette_scores = {}

print(f"Menjalankan Eksperimen 2, 3, 4: Melatih K-Means dengan K={K_VARIATIONS}...")

for k_val in K_VARIATIONS:
    # Membuat & melatih model
    kmeans_model = KMeans(n_clusters=k_val, init='k-means++', random_state=42, n_init=10)
    cluster_labels = kmeans_model.fit_predict(X_pca)
    
    # Menyimpan model dan labelnya
    models[k_val] = kmeans_model
    df_pca[f'cluster_k{k_val}'] = cluster_labels
    
    # Menghitung Silhouette Score (metrik evaluasi clustering)
    score = silhouette_score(X_pca, cluster_labels)
    silhouette_scores[k_val] = score
    print(f"  Model K={k_val} - Silhouette Score: {score:.4f}")

# --- PEMILIHAN MODEL FINAL & VISUALISASI ---

# Memilih model terbaik berdasarkan Silhouette Score tertinggi
best_k = max(silhouette_scores, key=silhouette_scores.get)
final_model = models[best_k]
final_labels = df_pca[f'cluster_k{best_k}']

print(f"\nModel Final Dipilih: K = {best_k} (Silhouette Score tertinggi)")

# Menambahkan hasil cluster final ke DataFrame PCA
df_pca['cluster_label'] = final_labels

# Menggabungkan hasil cluster kembali ke DataFrame asli (df_cleaned)
# Penting: Pastikan index-nya lurus
df_pca.index = track_info.index
df_final_analysis = pd.concat([track_info, df_pca], axis=1)

# --- Visualisasi Hasil (Sangat penting untuk laporan) ---
print("Membuat visualisasi cluster...")
plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=df_pca, 
    x='PCA_1', 
    y='PCA_2', 
    hue='cluster_label', 
    palette='viridis',  # Anda bisa ganti palet warna
    s=50, 
    alpha=0.7
)
plt.title(f'Segmentasi Lagu Spotify (K={best_k}) berdasarkan 2 Komponen PCA')
plt.xlabel('Komponen PCA 1 (DNA Lagu 1)')
plt.ylabel('Komponen PCA 2 (DNA Lagu 2)')
plt.legend(title='Cluster')
plt.grid(True)
plt.savefig(f'spotify_cluster_pca_k{best_k}.png')
print(f"Visualisasi cluster disimpan sebagai 'spotify_cluster_pca_k{best_k}.png'.")

# --- Analisis Awal (Bonus) ---
# Ini akan SANGAT membantu Anda menulis bagian "Key Findings"
print("\n--- Analisis Awal Karakteristik Cluster ---")
# Menggabungkan fitur audio asli dengan label cluster
df_final_analysis = pd.concat([df_final_analysis, X], axis=1)

# Menghitung rata-rata fitur audio untuk setiap cluster
cluster_profile = df_final_analysis.groupby('cluster_label')[audio_features].mean()
print(cluster_profile)

# Menyimpan profil cluster ke file
cluster_profile.to_csv('cluster_profile.csv')
print("Profil cluster (rata-rata fitur) disimpan di 'cluster_profile.csv'")

print("\n--- Analisis Awal Selesai ---")
print("Anda sekarang memiliki:")
print("1. 'elbow_plot.png' - Untuk bagian 'Ringkasan Pelatihan Model'.")
print(f"2. 'spotify_cluster_pca_k{best_k}.png' - Visual utama untuk 'Temuan Kunci'.")
print(f"3. 'cluster_profile.csv' - Tabel berisi 'DNA' dari setiap cluster. Ini adalah 'Wawasan' Anda!")