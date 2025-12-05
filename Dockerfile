# 1. Gunakan runtime Python resmi sebagai citra induk
FROM python:3.11-slim

# 2. Tetapkan direktori kerja di dalam kontainer
WORKDIR /app

# 3. Instal dependensi sistem yang diperlukan untuk OpenCV/GL
#    Memastikan pembaruan, instalasi, dan pembersihan cache dilakukan dalam satu langkah
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# 4. Salin requirements.txt dan instal paket
#    Langkah ini akan memanfaatkan caching layer Docker.
#    Jika requirements.txt tidak berubah, Docker tidak akan menjalankan kembali instalasi pip.
COPY requirements.txt .

# 5. Instal paket yang diperlukan, membersihkan cache pip
RUN pip install --no-cache-dir -r requirements.txt \
    # Opsional: Hapus pip cache setelah instalasi untuk citra yang lebih kecil
    && rm -rf /root/.cache/pip

# 6. Salin konten direktori saat ini (kode aplikasi) ke dalam kontainer di /app
#    Langkah ini dilakukan terakhir karena kode aplikasi lebih sering berubah
COPY . .

# 7. Jalankan face_recognition_app.py saat kontainer diluncurkan
CMD ["python", "main.py"]