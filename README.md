# Deteksi Makanan Indonesia

Proyek ini dibuat oleh **Akbar Aditiya Sugianto (Bengkel Coding)**.

Aplikasi berbasis **Streamlit** untuk mendeteksi makanan pada gambar menggunakan model **YOLO** dari Ultralytics dan menampilkan estimasi informasi nutrisi melalui **Google Gemini**.

## Fitur

- Deteksi satu gambar atau beberapa gambar sekaligus (batch).
- Dukungan gambar JPG, JPEG, dan PNG.
- Tampilan kotak deteksi, nama kelas makanan, dan confidence score.
- Pengaturan confidence threshold dan IoU threshold melalui sidebar.
- Estimasi kalori, protein, lemak, karbohidrat, serat, dan natrium, beserta manfaat dan kekurangan makanan.
- Penyimpanan hasil deteksi, galeri hasil, unduhan, dan penghapusan gambar.
- Daftar kelas makanan yang diambil langsung dari model.

## Struktur Proyek

```text
DeteksiMakanan/
├── app.py       # Aplikasi Streamlit, deteksi, dan analisis nutrisi
├── best.pt      # Bobot model YOLO
├── Tes/         # Contoh gambar untuk pengujian manual
├── hasil/       # Gambar hasil deteksi
├── README.md
└── LICENSE
```

Folder `hasil/` dibuat otomatis jika belum tersedia. Proyek ini belum menyertakan kode pelatihan model, dataset pelatihan, atau pengujian otomatis.

## Instalasi

Siapkan Python dan pip. Jalankan perintah berikut dari direktori proyek.

### 1. Buat virtual environment

```powershell
python -m venv .venv
```

Aktifkan di Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Untuk Linux atau macOS:

```bash
source .venv/bin/activate
```

### 2. Pasang dependensi

```bash
python -m pip install --upgrade pip
python -m pip install streamlit ultralytics Pillow opencv-python numpy SpeechRecognition pandas google-generativeai
```

`SpeechRecognition` perlu dipasang karena masih diimpor oleh `app.py`, meskipun antarmuka saat ini belum menyediakan fitur input suara. Versi dependensi belum dikunci dalam proyek ini.

### 3. Periksa konfigurasi

Pastikan `best.pt` tersedia di direktori proyek. Konfigurasi berikut berada di bagian awal `app.py`:

```python
RESULT_FOLDER = "hasil"
MODEL_PATH = "best.pt"
GEMINI_API_KEY = "API_KEY_ANDA"
```

Untuk analisis nutrisi, isi `GEMINI_API_KEY` dengan kunci milik Anda dari [Google AI Studio](https://aistudio.google.com/). Implementasi saat ini membaca kunci langsung dari `app.py`; variabel lingkungan dan Streamlit secrets belum digunakan. Jangan membagikan kunci API di repositori publik. Jika kunci yang tersimpan dalam kode pernah dibagikan, ganti atau cabut kunci tersebut.

Kode menggunakan model Gemini `gemini-2.5-flash`. Akses internet dan akses API yang valid dibutuhkan untuk permintaan analisis nutrisi. Deteksi gambar dilakukan oleh model YOLO secara lokal.

### 4. Jalankan aplikasi

```bash
python -m streamlit run app.py
```

Buka alamat yang ditampilkan di terminal, biasanya `http://localhost:8501`. Jalankan dari direktori proyek karena lokasi model dan hasil menggunakan path relatif.

## Cara Menggunakan

1. Periksa status model pada sidebar dan buka **Lihat Daftar Kelas** untuk mengetahui makanan yang didukung.
2. Gunakan **Mode Otomatis** (confidence `0.5`, IoU `0.45`) atau atur parameter secara manual.
3. Pada tab **Deteksi Single**, unggah satu gambar lalu klik **Deteksi Makanan**.
4. Pada tab **Deteksi Batch**, unggah beberapa gambar lalu klik **Deteksi Semua Gambar** untuk melihat ringkasannya.
5. Aktifkan **Simpan hasil deteksi** agar gambar beranotasi disimpan ke `hasil/`.
6. Buka **Hasil Tersimpan** untuk melihat, mengunduh, atau menghapus gambar. Tombol **Hapus Semua** langsung menghapus seluruh gambar hasil dalam folder tersebut.

Gambar contoh tersedia di folder `Tes/`. Nama hasil mengikuti pola `<nama_gambar>_<YYYYMMDD_HHMMSS>_detected.jpg`.

## Batasan

- Deteksi hanya mencakup kelas yang tersedia dalam `best.pt`. Akurasi bergantung pada model, pencahayaan, sudut pengambilan, dan kualitas gambar.
- Informasi nutrisi merupakan estimasi Gemini berdasarkan nama makanan dan permintaan porsi 100–150 gram; aplikasi tidak mengukur berat atau bahan makanan dari gambar.
- Jika permintaan Gemini atau parsing respons gagal, aplikasi menggunakan angka nutrisi umum yang sama untuk setiap makanan. Jika inisialisasi Gemini gagal, analisis nutrisi dilewati.
- Nama sumber data pada hasil berasal dari respons Gemini dan tidak diverifikasi oleh aplikasi.
- Asal, metrik evaluasi, dan lisensi bobot model serta gambar contoh belum didokumentasikan dalam proyek.

## Pemecahan Masalah

| Masalah | Langkah pemeriksaan |
| --- | --- |
| Model tidak dapat dimuat | Pastikan `best.pt` tersedia, path sesuai, dan berkas kompatibel dengan Ultralytics yang terpasang. |
| Tidak ada makanan terdeteksi | Periksa daftar kelas, gunakan gambar lebih jelas, atau turunkan confidence threshold. |
| Analisis nutrisi gagal | Periksa kunci API, koneksi internet, akses model, dan kuota Gemini. |
| Modul Python tidak ditemukan | Aktifkan virtual environment yang benar dan pasang seluruh dependensi di atas. |
| Hasil tidak tersimpan | Pastikan opsi penyimpanan aktif dan direktori proyek dapat ditulis. |

## Lisensi

Kode asli proyek ini menggunakan [MIT License](LICENSE). Teks lisensi mengikuti [Open Source Initiative](https://opensource.org/license/mit).

Dependensi, bobot model `best.pt`, dan gambar pihak ketiga mengikuti lisensi masing-masing pemiliknya. Lisensi MIT proyek tidak mengubah ketentuan lisensi aset atau dependensi tersebut.
