# DOKUMENTASI INTEGRASI MODEL SIBI KE WEB

## A. Tentang Model di Web

Sistem ini menggunakan model ResNet50 yang telah dikonversi ke format TensorFlow.js. File model yang dimuat adalah `model.json` yang terletak di folder `./model_web/`, disertai dengan file weights biner yang direferensikan di dalamnya. Library yang digunakan adalah **TensorFlow.js versi 4.11.0**, yang dimuat melalui CDN. 

Proses pemuatan model dilakukan melalui fungsi `loadModel()` yang menggunakan method `tf.loadGraphModel()` karena model merupakan GraphModel hasil konversi dari format SavedModel Python. Bersamaan dengan model utama, sistem juga memuat **HandPose model** dari TensorFlow.js Models untuk deteksi tangan, serta file `labels.json` yang berisi daftar 26 kelas gesture (kemungkinan A-Z dalam SIBI).

Citra yang akan diprediksi harus melalui preprocessing terlebih dahulu di fungsi `preprocessImage()`. Proses ini mencakup: konversi elemen gambar menjadi tensor menggunakan `tf.browser.fromPixels()`, resize ke dimensi 224×224 piksel dengan `tf.image.resizeBilinear()`, konversi format RGB ke BGR menggunakan `tf.reverse()` untuk kompatibilitas dengan ResNet50, casting ke tipe `float32`, dan pengurangan nilai mean ImageNet ([103.939, 116.779, 123.68]). Hasil preprocessing ini kemudian di-expand dimensinya menjadi batch size 1 sebelum dikirim ke model.

Model mengeluarkan output berupa tensor probabilitas yang kemudian dikonversi menjadi array JavaScript. Sistem mengambil indeks dengan nilai probabilitas tertinggi untuk mendapatkan kelas prediksi, serta menghitung top-5 prediksi dengan confidence masing-masing dalam persentase.

---

## B. Tentang Arsitektur Web

### HTML Structure
File HTML terdiri dari beberapa elemen utama yang membentuk antarmuka pengguna. Elemen `<video id="webcam">` digunakan untuk menampilkan stream dari webcam secara real-time dengan atribut `autoplay` dan `playsinline`. Elemen `<canvas id="canvas">` diposisikan sebagai overlay di atas video untuk menggambar bounding box dan label hasil deteksi.

Terdapat dua tab navigasi: "Kamera Real-time" dan "Upload Gambar", yang memungkinkan user memilih mode operasi. Untuk mode kamera, tersedia tiga tombol kontrol: "Mulai Kamera", "Stop", dan "Screenshot". Untuk mode upload, terdapat elemen `<input type="file" id="fileInput">` yang tersembunyi untuk pemilihan file, area drag-and-drop, serta elemen `<img id="uploadedImage">` untuk menampilkan gambar yang diupload.

Hasil prediksi ditampilkan dalam section `.prediction-box` yang menampilkan label prediksi, nilai confidence dalam persentase, progress bar visual, dan untuk mode upload juga menampilkan top-5 prediksi alternatif.

### CSS Styling
CSS menggunakan pendekatan modern dengan Google Font "Inter" dan tema dark mode berbasis gradasi warna ungu-biru (`#0a0e27` background dengan radial gradient). Implementasi glassmorphism terlihat pada elemen-elemen utama dengan `backdrop-filter: blur()` dan transparansi.

Layout menggunakan CSS Grid untuk prediction result dengan 2 kolom responsif, serta Flexbox untuk controls dan navigation tabs. Animasi smooth transition diterapkan pada semua interaksi button dengan `cubic-bezier(0.4, 0, 0.2, 1)` dan efek hover dengan `transform: translateY()`.

Bounding box pada canvas menggunakan corner-style border dengan warna dinamis: hijau (#00ff00) untuk confidence ≥70%, kuning (#ffff00) untuk 50-70%, dan oranye (#ffa500) untuk <50%. Progress bar confidence menggunakan gradient linear dari ungu ke biru ke hijau untuk memberikan visualisasi yang menarik.

Responsivitas diatur melalui media query `@media (max-width: 768px)` yang mengubah grid menjadi 1 kolom dan button menjadi full-width pada perangkat mobile.

### JavaScript Functions

**Fungsi Utama Deteksi:**
- `loadModel()` - Memuat model ResNet50, HandPose model, dan labels secara asynchronous
- `startWebcam()` - Mengakses webcam menggunakan `navigator.mediaDevices.getUserMedia()` dengan resolusi ideal 1280×720
- `detectionLoop()` - Loop animasi utama yang berjalan setiap frame menggunakan `requestAnimationFrame()`
- `detectHand()` - Menggunakan HandPose model untuk mendeteksi koordinat 21 landmark tangan dan menghitung bounding box
- `predict()` - Mengirim tensor ke model menggunakan `model.executeAsync()` dan mengolah output

**Fungsi Preprocessing dan ROI:**
- `preprocessImage()` - Melakukan transformasi citra sesuai requirement ResNet50
- `detectHandFromImage()` - Versi deteksi tangan untuk gambar statis yang diupload

**Fungsi Stabilisasi:**
Sistem menggunakan prediction buffer yang menyimpan 3 frame terakhir (STABILIZER_FRAMES = 3) dan menerapkan voting system untuk menghindari flickering. Hasil hanya ditampilkan jika confidence ≥80% (CONFIDENCE_THRESHOLD = 0.80), dengan smoothing update UI yang hanya terjadi jika perbedaan confidence >2%.

**Fungsi Upload:**
- `handleImageUpload()` - Membaca file menggunakan FileReader API dengan validasi ukuran maksimal 5MB
- `analyzeBtn.addEventListener()` - Melakukan crop ROI dari area tangan dan prediksi
- `visualizeHandBox()` - Menggambar bounding box dan label pada gambar yang diupload

**Fungsi UI:**
- `drawUI()` - Menggambar corner-style bounding box dan overlay text pada canvas
- `takeScreenshot()` - Mengambil snapshot dari canvas menggunakan `toDataURL('image/jpeg')`
- Tab switching menggunakan event delegation dengan `dataset.tab`

---

## C. Tentang Alur Kerja Sistem

### Alur Mode Kamera Real-time

1. **Inisialisasi Sistem**
   - User membuka halaman web → `loadModel()` dipanggil otomatis
   - Sistem memuat model ResNet50 dari `./model_web/model.json`
   - HandPose model dimuat dari CDN TensorFlow.js
   - Labels dimuat dari `labels.json`
   - Button "Mulai Kamera" diaktifkan setelah semua model siap

2. **Aktivasi Kamera**
   - User klik button "Mulai Kamera"
   - `startWebcam()` dipanggil → akses webcam via `getUserMedia()`
   - Stream video ditampilkan di elemen `<video>`
   - Canvas di-resize sesuai dimensi video actual
   - `detectionLoop()` dimulai dengan `requestAnimationFrame()`

3. **Proses Deteksi Per Frame**
   - Frame video digambar ke canvas
   - `detectHand()` mengestimasi posisi tangan menggunakan HandPose
   - Jika tangan terdeteksi:
     - Sistem menghitung bounding box dari 21 landmarks
     - Margin 40px ditambahkan ke semua sisi
     - Box diperluas menjadi square dengan sisi = max(width, height)
     - ROI (Region of Interest) di-crop dari frame

4. **Preprocessing dan Prediksi**
   - ROI di-crop menggunakan canvas intermediate untuk pixel-perfect extraction
   - `preprocessImage()` mengubah ROI menjadi tensor 224×224 BGR
   - Mean subtraction ImageNet diterapkan
   - Tensor dikirim ke model via `model.executeAsync()`
   - Model mengembalikan array probabilitas 26 kelas

5. **Stabilisasi dan Visualisasi**
   - Jika confidence ≥80%, hasil masuk ke prediction buffer
   - Buffer menyimpan 3 frame terakhir
   - Voting system menentukan label paling konsisten
   - Confidence di-rata-rata untuk label terpilih
   - UI hanya update jika perbedaan >2% (anti-flickering)
   - `drawUI()` menggambar corner-style bounding box dengan warna sesuai confidence
   - Label dan confidence ditampilkan di overlay canvas dan panel prediksi

6. **Handling Tanpa Deteksi Tangan**
   - Jika HandPose tidak mendeteksi tangan:
   - Prediction buffer di-clear
   - UI menampilkan "Tidak ada tangan terdeteksi"
   - Label dan confidence di-reset ke default

### Alur Mode Upload Gambar

1. **Pemilihan Gambar**
   - User klik area upload atau drag-and-drop gambar
   - File dipilih melalui `<input type="file">` (hidden)
   - `handleImageUpload()` validasi ukuran file (max 5MB)
   - FileReader membaca file sebagai Data URL
   - Gambar ditampilkan di area preview
   - Button "Analisis" dan "Clear" diaktifkan

2. **Proses Analisis**
   - User klik button "Analisis"
   - Button berubah menjadi "⏳ Menganalisis..."
   - `detectHandFromImage()` dipanggil:
     - Gambar digambar ke canvas temporary
     - HandPose mendeteksi tangan dari canvas
     - Bounding box dihitung dengan margin 40px
     - Box diperluas menjadi square

3. **Crop dan Prediksi**
   - ROI di-crop langsung dari gambar asli menggunakan `drawImage()` dengan source rectangle
   - Crop canvas berisi hanya area tangan
   - `predict()` melakukan preprocessing dan inferensi
   - Sistem mendapatkan top-5 prediksi dengan confidence

4. **Visualisasi Hasil**
   - Label utama dan confidence ditampilkan di panel
   - Progress bar diupdate sesuai confidence
   - Top-5 prediksi ditampilkan dalam list dengan bar individual
   - `visualizeHandBox()` menggambar bounding box pada gambar:
     - Gambar asli digambar ke canvas baru
     - Corner-style box digambar dengan warna sesuai confidence
     - Label dan confidence ditampilkan sebagai overlay
     - Gambar hasil visualisasi menggantikan gambar asli di preview

5. **Opsi Clear**
   - User bisa klik "Clear" untuk reset semua
   - Gambar dihapus, placeholder muncul kembali
   - Hasil prediksi di-reset
   - Sistem siap untuk upload baru

### Fitur Screenshot (Mode Kamera)
- User klik button "📸 Screenshot"
- `takeScreenshot()` mengambil current state canvas dengan `toDataURL('image/jpeg')`
- File otomatis di-download dengan nama format: `SIBI_{label}_{confidence}_{timestamp}.jpg`
- Flash effect visual memberikan feedback ke user

---

## Detail Teknis Penting

### Perbedaan GraphModel vs LayersModel
Sistem ini menggunakan `tf.loadGraphModel()` bukan `tf.loadLayersModel()` karena model berasal dari konversi SavedModel Python. GraphModel memerlukan `executeAsync()` untuk inferensi, bukan `predict()`.

### Konversi RGB ke BGR
ResNet50 dilatih dengan Keras preprocessing yang menggunakan format BGR (bukan RGB standar). Oleh karena itu, `tf.reverse(tensor, -1)` diperlukan untuk membalik channel terakhir sebelum mean subtraction.

### Memory Management
Semua operasi tensor dibungkus dalam `tf.tidy()` untuk automatic cleanup, dan tensor hasil prediksi di-dispose secara manual dengan `.dispose()` untuk mencegah memory leak.

### Hand Detection Accuracy
Sistem menggunakan margin 40px yang sama persis dengan implementasi Python untuk konsistensi. Square bounding box memastikan aspect ratio 1:1 sesuai requirement input model 224×224.

### Anti-Flickering Mechanism
Kombinasi dari:
- Confidence threshold 80% untuk filtering
- Buffer 3 frame dengan voting system
- UI update hanya jika delta >2%
- Memberikan hasil yang stabil tanpa jumping label

---

## Limitasi dan Catatan

1. **Browser Compatibility**: Memerlukan browser modern dengan support WebGL dan MediaDevices API
2. **HTTPS Requirement**: Webcam access memerlukan HTTPS di production (kecuali localhost)
3. **Model Size**: Model dan weights perlu di-host di server yang sama (CORS restriction)
4. **Performance**: Deteksi real-time optimal pada device dengan GPU acceleration
5. **Hand Detection**: HandPose model memiliki akurasi lebih rendah dibanding MediaPipe di Python, terutama untuk gesture kompleks