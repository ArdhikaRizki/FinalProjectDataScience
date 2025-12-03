import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ganti 'nama_file.csv' dengan nama file dataset kamu
df = pd.read_csv('datasets/dataset.csv')

# --- Contoh DataFrame (jika kamu belum punya file) ---
# data = {
#     'komentar': [
#         'promo link gacor hari ini',
#         'keren banget videonya kak!',
#         'Ayo daftar di situs kami sekarang juga',
#         'keren banget videonya kak!', # Duplikat
#         'cuma 1 kata',
#         'ini adalah contoh komentar yang sangat panjang sekali dibuat hanya untuk tujuan demonstrasi agar bisa dideteksi sebagai outlier berdasarkan panjang katanya oleh program analisis data',
#         None, # Data hilang
#         'jangan lupa mampir ya'
#     ],
#     'label': [1, 0, 1, 0, 0, 1, 0, 0]
# }
# df = pd.DataFrame(data)
# ----------------------------------------------------


print("=============================================")
print("          1. CEK DATA HILANG (MISSING)")
print("=============================================")
# Menghitung jumlah data hilang di setiap kolom
missing_values = df.isnull().sum()
print("Jumlah data hilang per kolom:")
print(missing_values)
print("\n")


print("=============================================")
print("           2. CEK DATA DUPLIKAT")
print("=============================================")
# Menghitung jumlah baris data yang terduplikasi
duplicate_rows = df.duplicated().sum()
print(f"Total baris data duplikat ditemukan: {duplicate_rows} baris\n")

# Opsional: Menampilkan baris yang duplikat (jika ada)
if duplicate_rows > 0:
    print("Contoh baris data yang duplikat:")
    print(df[df.duplicated(keep=False)]) # 'keep=False' akan menampilkan semua baris duplikat
print("\n")


print("=============================================")
print("            3. CEK OUTLIER (Teks)")
print("=============================================")
# Untuk data teks, outlier bisa dideteksi dari panjangnya (jumlah kata atau karakter)
# Kita akan membuat kolom baru untuk menyimpan panjang komentar dalam jumlah kata

# Pastikan tidak ada nilai NaN di kolom 'komentar' sebelum split
df['panjang_komentar'] = df['komentar'].dropna().str.split().str.len()

print("Statistik deskriptif untuk panjang komentar:")
# .describe() akan memberikan gambaran statistik (rata-rata, min, max, kuartil)
print(df['panjang_komentar'].describe())
print("\n")

# Visualisasi untuk melihat distribusi panjang komentar
plt.figure(figsize=(10, 6))
sns.histplot(df['panjang_komentar'], bins=30, kde=True)
plt.title('Distribusi Panjang Komentar (Jumlah Kata)')
plt.xlabel('Jumlah Kata')
plt.ylabel('Frekuensi')
plt.show()

# Membuat boxplot untuk melihat outlier dengan lebih jelas
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['panjang_komentar'])
plt.title('Boxplot Panjang Komentar')
plt.xlabel('Jumlah Kata')
plt.show()