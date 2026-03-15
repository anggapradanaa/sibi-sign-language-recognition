# SIBI-INSIGN

**Intelligent Sign Gesture Recognition for SIBI**

Real-time web application untuk mengenali gerakan bahasa isyarat SIBI (Sistem Isyarat Bahasa Indonesia) menggunakan TensorFlow.js dan ResNet50.

## ✨ Fitur

- **Real-time Detection**: Deteksi dan klasifikasi gestur tangan langsung dari webcam
- **Upload & Analyze**: Upload gambar untuk analisis offline
- **Hand Detection**: Deteksi otomatis region tangan menggunakan HandPose model
- **Top-5 Predictions**: Menampilkan 5 prediksi teratas dengan confidence score
- **Screenshot**: Simpan hasil deteksi dengan metadata

## 🛠️ Teknologi

- **TensorFlow.js** v4.11.0 - Deep learning inference di browser
- **ResNet50** - Model klasifikasi pre-trained
- **HandPose** - Hand landmark detection
- **HTML5 Canvas** - Real-time visualization
- **Modern CSS** - Glassmorphism design dengan animasi

## 📁 Struktur File

```
├── index.html          # Main HTML structure
├── style.css           # Styling dengan glassmorphism theme
├── script.js           # Logic deteksi & prediksi
├── model_web/          # TensorFlow.js model files
│   └── model.json
└── labels.json         # Class labels
```

## 🚀 Cara Menggunakan

### 1. Setup
Pastikan file model tersedia:
- `model_web/model.json` - TensorFlow.js GraphModel
- `labels.json` - Array label kelas SIBI

### 2. Jalankan
Buka `index.html` di browser modern (Chrome/Edge/Firefox recommended)

### 3. Mode Kamera Real-time
1. Klik **Mulai Kamera**
![Mulai-Kamera](https://raw.githubusercontent.com/anggapradanaa/sibi-sign-language-recognition/main/images/Mulai%20Kamera.png)
3. Izinkan akses webcam
4. Tunjukkan gestur tangan ke kamera
5. Sistem menampilkan prediksi secara real-time
![Hasil Prediksi Real-Time](https://raw.githubusercontent.com/anggapradanaa/sibi-sign-language-recognition/main/images/Hasil%20Prediksi%20Real-Time.png)

## Mode Upload Gambar

1. Pilih tab **Upload Gambar**
![Klik Upload Gambar](https://raw.githubusercontent.com/anggapradanaa/sibi-sign-language-recognition/main/images/Klik%20Upload%20Gambar.png)
2. Upload gambar gestur
![Upload Gambar](https://raw.githubusercontent.com/anggapradanaa/sibi-sign-language-recognition/main/images/Upload%20Gambar.png)
3. Klik **Analisis**
![Klik Analisis](https://raw.githubusercontent.com/anggapradanaa/sibi-sign-language-recognition/main/images/Klik%20Analisis.png)
4. Sistem menampilkan **Top-5 prediksi**
![Hasil 1](https://raw.githubusercontent.com/anggapradanaa/sibi-sign-language-recognition/main/images/Hasil%20Upload%20Gambar%20(2).png)
![Hasil 2](https://raw.githubusercontent.com/anggapradanaa/sibi-sign-language-recognition/main/images/Hasil%20Upload%20Gambar%20(1).png)

## ⚙️ Konfigurasi

```javascript
// di script.js
const MODEL_PATH = './model_web/model.json';
const LABELS_PATH = './labels.json';
const CONFIDENCE_THRESHOLD = 0.80;  // Threshold minimum
const STABILIZER_FRAMES = 3;        // Frame buffer untuk stabilisasi
```

## 🎯 Fitur Utama

### Preprocessing
- Resize otomatis ke 224x224
- BGR mean subtraction: [103.939, 116.779, 123.68]
- ROI cropping dengan margin adaptif

### Stabilisasi Prediksi
- Voting system dari buffer 3 frame
- Average confidence untuk mengurangi flickering
- Update UI hanya jika perubahan > 2%

### Hand Detection
- Automatic bounding box dengan margin 40px
- Square ROI untuk konsistensi
- Pixel-perfect cropping menggunakan ImageData API

## 📋 Requirements

- Browser modern dengan support:
  - WebGL 2.0
  - getUserMedia API
  - ES6+ JavaScript
- Webcam (untuk mode real-time)

## 🎨 UI Features

- **Dark theme** dengan gradient purple-blue
- **Glassmorphism** effect
- **Smooth animations** pada semua interaksi
- **Responsive design** untuk mobile & desktop
- **Visual feedback** dengan corner bounding box

## 📝 Catatan

- Model menggunakan ResNet50 architecture
- Input size: 224x224x3 (RGB)
- Output: Softmax probabilities untuk semua kelas SIBI
- Confidence threshold default: 80%

## 🔧 Troubleshooting

**Model tidak load:**
- Pastikan path model benar
- Check console untuk error details

**Tangan tidak terdeteksi:**
- Pastikan pencahayaan cukup
- Jarak ideal: 30-100cm dari kamera
- Background kontras dengan warna kulit

**Prediksi tidak stabil:**
- Tingkatkan `STABILIZER_FRAMES`
- Adjust `CONFIDENCE_THRESHOLD`

---

**Framework:** TensorFlow.js | **Model:** ResNet50 | **Detection:** HandPose
