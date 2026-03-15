# 🤟 SIBI-INSIGN

**Intelligent Sign Gesture Recognition for SIBI**

Aplikasi web berbasis **deep learning** untuk mengenali gerakan bahasa isyarat **SIBI (Sistem Isyarat Bahasa Indonesia)** secara **real-time** menggunakan kamera atau gambar yang diunggah. Sistem ini menggunakan model **ResNet50** yang dijalankan langsung di browser menggunakan **TensorFlow.js**.

---

# 🚀 Deskripsi Proyek

Proyek ini merupakan implementasi **end-to-end machine learning pipeline**, mulai dari **training model**, evaluasi beberapa arsitektur CNN, hingga **deployment model dalam aplikasi web berbasis browser**.

Pipeline sistem:

Dataset → Training CNN → Model `.h5` → TensorFlow SavedModel → TensorFlow.js → Web Application

Model terbaik yang dipilih adalah **ResNet50** karena memiliki performa paling tinggi dibanding arsitektur lainnya.

---

# ✨ Fitur Aplikasi

🎥 **Real-time Detection**
Deteksi dan klasifikasi gestur tangan langsung dari webcam.

🖼 **Upload & Analyze**
Upload gambar untuk analisis gestur secara offline.

✋ **Hand Detection**
Deteksi otomatis area tangan menggunakan **HandPose model**.

📊 **Top-5 Predictions**
Menampilkan 5 prediksi kelas dengan confidence score.

📸 **Screenshot Result**
Menyimpan hasil deteksi dengan metadata.

---

# 🧠 Arsitektur Model

Tiga arsitektur CNN diuji dalam proyek ini:

* **ResNet50**
* **VGG16**
* **MobileNetV2**

Model dilatih menggunakan **transfer learning** dengan pretrained weights dari ImageNet. Setelah training, model disimpan dalam format **`.h5`**, kemudian dikonversi menjadi **TensorFlow SavedModel**, dan akhirnya dikonversi ke **TensorFlow.js** untuk digunakan pada aplikasi web.

---

# 📊 Performa Model

| Model           | Validation Accuracy | Test Accuracy |
| --------------- | ------------------- | ------------- |
| **ResNet50**    | 0.9955              | 0.9933        |
| **VGG16**       | 0.9933              | 0.9910        |
| **MobileNetV2** | 0.9933              | 0.9865        |

Berdasarkan hasil evaluasi, **ResNet50** dipilih sebagai **model final** karena memberikan performa terbaik.

---

# 🛠 Teknologi yang Digunakan

🧠 **TensorFlow / Keras** — training deep learning model
🌐 **TensorFlow.js** — menjalankan model di browser
✋ **HandPose** — hand landmark detection
🎨 **HTML5 Canvas** — visualisasi real-time
💻 **JavaScript (ES6)** — logic aplikasi
🎨 **Modern CSS** — desain UI dengan efek glassmorphism

---

# 📂 Struktur Repository

```
sibi-sign-language-recognition
│
├── training/           # Pipeline training model
│
├── web_app/            # Aplikasi web untuk inference
│   ├── index.html
│   ├── style.css
│   ├── script.js
│   ├── labels.json
│   └── model_web
│       └── model.json
│
└── README.md
```

---

# 🚀 Cara Menjalankan Aplikasi

### 1️⃣ Clone repository

```
git clone https://github.com/anggapradanaa/sibi-sign-language-recognition.git
```

### 2️⃣ Masuk ke folder aplikasi

```
cd web_app
```

### 3️⃣ Jalankan aplikasi

Buka file berikut di browser:

```
index.html
```

Gunakan browser modern seperti:

* Google Chrome
* Microsoft Edge
* Mozilla Firefox

---

# 🎥 Mode Penggunaan

## Mode Kamera Real-time

1. Klik **Mulai Kamera**
![Mulai-Kamera](https://raw.githubusercontent.com/anggapradanaa/sibi-sign-language-recognition/main/images/Mulai%20Kamera.png)
3. Izinkan akses webcam
4. Tunjukkan gestur tangan ke kamera
5. Sistem menampilkan prediksi secara real-time
![Hasil Prediksi Real-Time](https://raw.githubusercontent.com/anggapradanaa/sibi-sign-language-recognition/main/images/Hasil%20Prediksi%20Real-Time.png)

## Mode Upload Gambar

1. Pilih tab **Upload Gambar**
![Klik Upload Gambar](https://github.com/anggapradanaa/sibi-sign-language-recognition/main/images/Klik%20Upload%20Gambar.png)
2. Upload gambar gestur
![Upload Gambar](https://github.com/anggapradanaa/sibi-sign-language-recognition/main/images/Upload%20Gambar.png)
3. Klik **Analisis**
![Klik Analisis](https://github.com/anggapradanaa/sibi-sign-language-recognition/main/images/Klik%20Analisis.png)
4. Sistem menampilkan **Top-5 prediksi**
![Hasil 1](https://github.com/anggapradanaa/sibi-sign-language-recognition/main/images/Hasil%20Upload%20Gambar%20(2).png)
![Hasil 2](https://github.com/anggapradanaa/sibi-sign-language-recognition/main/images/Hasil%20Upload%20Gambar%20(1).png)
---

# ⚙️ Konfigurasi Model

Di dalam `script.js`:

```
const MODEL_PATH = './model_web/model.json';
const LABELS_PATH = './labels.json';
const CONFIDENCE_THRESHOLD = 0.80;
const STABILIZER_FRAMES = 3;
```

---

# 📋 Requirements

Browser modern dengan dukungan:

* WebGL 2.0
* getUserMedia API
* ES6 JavaScript

Untuk mode real-time diperlukan **webcam**.

---

# 📝 Catatan

* Model menggunakan arsitektur **ResNet50**
* Input size: **224 × 224 × 3**
* Output: **Softmax probabilities untuk kelas SIBI**
* Confidence threshold default: **80%**

---

# 👨‍💻 Author

Proyek ini dibuat sebagai implementasi **Computer Vision dan Deep Learning** untuk sistem pengenalan bahasa isyarat berbasis web.
