# Penerapan Program Face Tracking Menggunakan Algoritma Cascade Classifier dari Library OpenCV

## Anggota Kelompok Bubadibako:

- Naufaldy
- Fadilah
- Yudha
- Bunga
- Fauzan
- Dicky
- Nabil

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/d/d2/OpenCV_logo_black.svg/907px-OpenCV_logo_black.svg.png" alt="opencv" width="20%">
<img src="https://www.mygreatlearning.com/blog/wp-content/uploads/2020/08/dp.png" alt="face tracking" width="30%">

# **Apa itu Face Tracking?**

Face tracking  atau pelacakan wajah adalah teknologi yang dapat mendeteksi dan mengikuti posisi wajah seseorang secara real time melalui kamera. 

Teknologi ini banyak digunakan dalam:

- Sistem keamanan otomatis
- Absensi berbasi wajah
- Efek kamera

Salah satu cara untuk membuat face tracking adalah dengan algoritma Cascade Classifier dari Library OpenCV di Python.

OpenCV (Open Source Computer Vision Library) adalah library open source untuk pengolahan  gambar atau video, serta pengembangan sistem visi komputer (computer vision) dan pembelajaran mesin (machine learning).

## **Algoritma Cascade Classifier**

<center>
</center>

Casecade Classifier adalah algoritma deteksi objek yang dikembangkan oleh Viola dan Jones pada tahun 2001. Disebut Cascade karena prosesnya bertingkat, mulai dari penyaringan sederhana sampai kompleks.

Algoritma dari cascade classifier berjalan mengunakan library OpenCV yang melalui proses 4 tahap yaitu :

1. **Selecting Haar-like features :**

<center>
<img src="https://i.sstatic.net/ssXnC.png" alt="haar-like" width="40%">
</center>
    
Merupakan proses memilih pola-pola sederhana (fitur Haar) yang digunakan untuk mengenali ciri khas suatu objek, seperti perbedaan terang dan gelap pada wajah. Fitur ini membantu mendeteksi bagian-bagian penting dari objek, seperti mata, hidung, atau tepi wajah

Tentu. Rumus tersebut adalah inti dari cara kerja Haar-like Features, yang merupakan fondasi dari algoritma deteksi wajah Viola-Jones.

Secara sederhana, rumus ini bekerja dengan membandingkan total kecerahan (intensitas) piksel di area hitam dengan total kecerahan piksel di area putih pada sebuah templat fitur.

#### **Keterangan Algoritma**

Berikut ini adalah keterangan dari setiap komponen pada rumus:

### **$$f(x, y) = \sum_{i \in R_b} p(i) - \sum_{i \in R_w} p(i)$$**

$$\text{Nilai Fitur} = \text{Jumlah Intensitas Piksel di Area Hitam} - \text{Jumlah Intensitas Piksel di Area Putih}$$

- **$f(x, y)$** : Ini adalah nilai akhir atau skor dari fitur tersebut ketika ditempatkan pada posisi (x, y) di gambar. Hasilnya adalah satu angka tunggal.

- **$\sum$** (Sigma) : Ini adalah simbol matematika untuk iterasi. Artinya summation atau "jumlahkan semua..."

- **$i \in R_b$** : p berarti pixel intensity (nilai kecerahan piksel, biasanya 0 untuk hitam pekat dan 255 untuk putih pekat).
Subskrip b berarti black (hitam).
Jadi, Σ **$i \in R_b$**  berarti "total jumlah nilai kecerahan dari semua piksel yang berada di bawah area hitam pada templat."

- **$i \in R_w$**: p berarti pixel intensity.
Subskrip w berarti white (putih).
Jadi, Σ **$i \in R_w$** berarti "total jumlah nilai kecerahan dari semua piksel yang berada di bawah area putih pada templat."

- **$p(i)$** Intensitas (nilai kecerahan) piksel pada indeks $i$. 

2. **Creating an integral image :**

Tahap ini membuat representasi gambar baru di mana setiap titik menyimpan jumlah total nilai piksel dari sudut kiri atas hingga titik tersebut. Teknik ini digunakan untuk mempercepat perhitungan fitur Haar di berbagai area gambar.

  <center>
      <img src="https://miro.medium.com/v2/resize:fit:1100/format:webp/0*H_2BKqQKmrI8uAeY" alt="integral image" width="40%">
  </center>
      
Gambar integral memberikan cara cepat dan sederhana untuk menghitung nilai fitur Haar-like feature. Hal ini dilakukan dengan memanfaatkan Summed Area Table. Nilai untuk lokasi (x, y) pada gambar integral adalah jumlah piksel di atas dan di kiri (x, y) pada gambar asli ditambah dirinya sendiri. Misalnya, untuk menghitung (2,2) = 16 = 5+2+3+6

Setelah membangun gambar integral, gambar tersebut digunakan untuk menghitunng diinginkan.

  <center>
      <img src="https://miro.medium.com/v2/resize:fit:1100/format:webp/0*EJqS55ys5KY8Rfip" alt="integral image2" width="40%">    
  </center>
      
Untuk mendapatkan jumlah persegi panjang biru, cukup tambahkan nilai hijau dan kurangi nilai merah pada gambar integral. 1+21–11–3 = 8.

  <center>
      <img src="https://i.sstatic.net/V7x9c.png" alt="integral image after" width="40%">    
  </center>

3. **Running AdaBoost training :**
Pada tahap ini, digunakan algoritma pembelajaran AdaBoost untuk memilih dan menggabungkan fitur-fitur terbaik (yang paling mampu membedakan antara objek dan bukan objek) menjadi sebuah model klasifikasi yang kuat. Algoritma ini juga memberi bobot lebih pada kesalahan, agar model terus membaik pada setiap iterasi.
<center>
    <img src="https://i.sstatic.net/K1TpT.png" alt="adaboost" width="40%">
</center>

4. **Creating classifier cascades :**
Proses menyusun beberapa classifier secara bertahap, dari yang paling sederhana hingga yang lebih kompleks. Setiap tahap bertugas menyaring area gambar jika sebuah area gagal pada tahap awal, maka langsung dianggap bukan objek dan tidak dilanjutkan ke tahap berikutnya. Pendekatan ini membuat proses deteksi menjadi jauh lebih cepat dan efisien.
<center>
<img src="https://www.mygreatlearning.com/blog/wp-content/uploads/2020/08/dp.png" alt="face tracking" width="30%">
</center>

### **Cara Kerja Sederhana Algoritma Cascade Classifier**

- Gambar wajah dibaca frame per frame dari kamera.
- Algoritma memeriksa tiap bagian gambar menggunakan fitur Haar-Like.
- Tiap tahap menyaring bagian yang tidak mirip wajah.
- Bagian yang lolos dari semua tahap dianggap wajah.

### **Kenapa Dipakai di OpenCV**

- Cepat, karena tidak perlu memproses seluruh piksel.
- Sudah tersedia dalam bentuk model XML yang bisa langsung di gunakan, contohnya “haarcascade_frontalfcae_default.xml”.

# Kelebihan dan Kekurangan

### Kelebihan:

- Proses cepat dan ringan untuk real time.
- Mudah diimplementasikan.
- Tidak butuh data training tambahan.

### Kekurangan:

- Kurang akurat pada pencahayan rendah.
- Tidak bisa mengenali wajah individu (hanya mendeteksi wajah).
- Sulit mendeteksi wajah dari samping tau miring.

# Kesimpulan

Face tracking menggunakan Cascade Classifier dari OpenCV itu salah satu cara untuk membuat komputer mendeteksi wajah secara realtime. Walau hasilnya masih banyak kekurangannya, tetapi ini adalah salah satu metode pembelajaran yang mudah untuk mengenali wajah secara realtime. Dengan pehamanan dasar ini bisa dikembangkan lagi untuk membuat sistem keamanan ataupun robot yang bergerak jika mendeteksi wajah.
