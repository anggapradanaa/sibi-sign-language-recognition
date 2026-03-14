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
1. Klik **"Mulai Kamera"**
2. Izinkan akses webcam
3. Tunjukkan gestur tangan SIBI ke kamera
4. Lihat prediksi real-time dengan bounding box

### 4. Mode Upload Gambar
1. Pilih tab **"Upload Gambar"**
2. Drag & drop atau klik untuk upload
3. Klik **"Analisis"**
4. Lihat hasil dengan top-5 predictions

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