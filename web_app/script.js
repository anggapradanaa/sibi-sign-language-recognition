// ========== KONFIGURASI ==========
const MODEL_PATH = './model_web/model.json';
const LABELS_PATH = './labels.json';
const CONFIDENCE_THRESHOLD = 0.80;
const STABILIZER_FRAMES = 3;

// ========== VARIABEL GLOBAL ==========
let model = null;
let labels = [];
let webcamStream = null;
let isDetecting = false;
let animationId = null;
let handposeModel = null;

// Prediction buffer untuk stabilisasi
let predictionBuffer = [];

// Elements
const webcam = document.getElementById('webcam');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const status = document.getElementById('status');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const screenshotBtn = document.getElementById('screenshotBtn');
const predictedLabel = document.getElementById('predictedLabel');
const confidence = document.getElementById('confidence');
const confidenceBar = document.getElementById('confidenceBar');

// Upload elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const selectFileBtn = document.getElementById('selectFileBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const clearBtn = document.getElementById('clearBtn');
const uploadedImage = document.getElementById('uploadedImage');
const uploadPredictedLabel = document.getElementById('uploadPredictedLabel');
const uploadConfidence = document.getElementById('uploadConfidence');
const uploadConfidenceBar = document.getElementById('uploadConfidenceBar');
const topPredictions = document.getElementById('topPredictions');
const topPredictionsList = document.getElementById('topPredictionsList');

// ========== LOAD MODEL DAN LABELS ==========
async function loadModel() {
    try {
        updateStatus('Memuat model ResNet50...');
        document.getElementById('modelStatus').textContent = 'Memuat model...';
        
        // Load model TensorFlow.js
        model = await tf.loadGraphModel(MODEL_PATH);
        console.log('✓ Model berhasil dimuat!');
        
        // Load labels
        const response = await fetch(LABELS_PATH);
        const data = await response.json();
        labels = data.labels || data;
        console.log('✓ Labels berhasil dimuat:', labels);
        
        // Load handpose model
        updateStatus('Memuat hand detection model...');
        try {
            handposeModel = await handpose.load();
            console.log('✓ HandPose model berhasil dimuat!');
        } catch (e) {
            console.warn('HandPose model gagal dimuat:', e);
        }
        
        updateStatus('Model siap! Klik "Mulai Kamera" untuk memulai.');
        document.getElementById('modelStatus').textContent = `Model aktif | ${labels.length} kelas`;
        startBtn.disabled = false;
        
    } catch (error) {
        console.error('Error loading model:', error);
        updateStatus('Error: Gagal memuat model. Periksa console untuk detail.');
        document.getElementById('modelStatus').textContent = 'Error loading model';
        alert('Gagal memuat model! Pastikan file model.json dan labels.json tersedia di folder yang benar.');
    }
}

// ========== UPDATE STATUS ==========
function updateStatus(message) {
    status.innerHTML = `<p>${message}</p>`;
}

// ========== PREPROCESS IMAGE ==========
function preprocessImage(imageElement) {
    return tf.tidy(() => {
        let tensor = tf.browser.fromPixels(imageElement);

        // Resize ONCE (sama dengan Python)
        tensor = tf.image.resizeBilinear(tensor, [224, 224]);

        // RGB → BGR (WAJIB untuk ResNet50)
        tensor = tf.reverse(tensor, -1);

        tensor = tf.cast(tensor, 'float32');

        // Mean subtraction (ImageNet)
        const mean = tf.tensor1d([103.939, 116.779, 123.68]);
        tensor = tensor.sub(mean);

        return tensor.expandDims(0);
    });
}

// ========== PREDICT ==========
async function predict(imageElement) {
    if (!model) {
        console.error('Model belum dimuat!');
        return null;
    }
    
    try {
        const inputTensor = preprocessImage(imageElement);
        
        // GraphModel pakai executeAsync
        const predictionTensor = await model.executeAsync(inputTensor);
        const predictions = await predictionTensor.data();
        
        // Dispose tensors
        inputTensor.dispose();
        predictionTensor.dispose();
        
        // Get predicted class
        const maxIndex = Array.from(predictions).indexOf(Math.max(...predictions));
        const predictedClass = labels[maxIndex];
        const conf = predictions[maxIndex] * 100;
        
        console.log(`Predicted: ${predictedClass}, Confidence: ${conf.toFixed(1)}%`);
        
        // Get top 5 predictions
        const top5 = Array.from(predictions)
            .map((prob, idx) => ({ label: labels[idx], confidence: prob * 100 }))
            .sort((a, b) => b.confidence - a.confidence)
            .slice(0, 5);
        
        return {
            label: predictedClass,
            confidence: conf,
            top5: top5
        };
    } catch (error) {
        console.error('Error during prediction:', error);
        return null;
    }
}

// ========== DETECT HAND ==========
async function detectHand(video) {
    if (!handposeModel) return null;
    
    try {
        const predictions = await handposeModel.estimateHands(video);
        if (predictions.length > 0) {
            const hand = predictions[0];
            const landmarks = hand.landmarks;
            
            // Get bounding box
            const xs = landmarks.map(p => p[0]);
            const ys = landmarks.map(p => p[1]);

            let xMin = Math.min(...xs);
            let xMax = Math.max(...xs);
            let yMin = Math.min(...ys);
            let yMax = Math.max(...ys);

            // Add margin (40px seperti deploy.py)
            const margin = 40;
            const videoWidth = webcam.videoWidth;
            const videoHeight = webcam.videoHeight;

            // Tambah margin dulu (seperti Python)
            xMin = Math.max(0, xMin - margin);
            yMin = Math.max(0, yMin - margin);
            xMax = Math.min(videoWidth, xMax + margin);
            yMax = Math.min(videoHeight, yMax + margin);

            // Baru bikin square
            const width = xMax - xMin;
            const height = yMax - yMin;
            const size = Math.max(width, height);

            const centerX = (xMin + xMax) / 2;
            const centerY = (yMin + yMax) / 2;

            return {
                x: Math.max(0, centerX - size / 2),
                y: Math.max(0, centerY - size / 2),
                width: Math.min(size, videoWidth),
                height: Math.min(size, videoHeight),
                landmarks: landmarks
            };
        }
    } catch (error) {
        console.warn('Hand detection error:', error);
    }
    
    return null;
}

// ========== DETECT HAND DARI STATIC IMAGE ==========
async function detectHandFromImage(imageElement) {
    if (!handposeModel) return null;

    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    const width = imageElement.naturalWidth;
    const height = imageElement.naturalHeight;

    canvas.width = width;
    canvas.height = height;
    ctx.drawImage(imageElement, 0, 0, width, height);

    const predictions = await handposeModel.estimateHands(canvas);

    if (predictions.length === 0) return null;

    const landmarks = predictions[0].landmarks;
    const xs = landmarks.map(p => p[0]);
    const ys = landmarks.map(p => p[1]);

    let xMin = Math.min(...xs);
    let xMax = Math.max(...xs);
    let yMin = Math.min(...ys);
    let yMax = Math.max(...ys);

    const margin = 40; // SAMA PERSIS PYTHON
    xMin = Math.max(0, xMin - margin);
    yMin = Math.max(0, yMin - margin);
    xMax = Math.min(width, xMax + margin);
    yMax = Math.min(height, yMax + margin);

    const boxW = xMax - xMin;
    const boxH = yMax - yMin;
    const size = Math.max(boxW, boxH);

    const cx = (xMin + xMax) / 2;
    const cy = (yMin + yMax) / 2;

    return {
        x: Math.max(0, cx - size / 2),
        y: Math.max(0, cy - size / 2),
        width: Math.min(size, width),
        height: Math.min(size, height)
    };
}


// ========== DRAW UI ON CANVAS ==========
function drawUI(handBox, label, conf) {
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    if (handBox) {
        // Draw bounding box with corners
        const { x, y, width, height } = handBox;
        const cornerLen = 30;
        const thickness = 3;
        
        // Determine color based on confidence
        let color = conf >= 70 ? '#00ff00' : conf >= 50 ? '#ffff00' : '#ffa500';
        
        ctx.strokeStyle = color;
        ctx.lineWidth = thickness;
        
        // Top-left corner
        ctx.beginPath();
        ctx.moveTo(x, y + cornerLen);
        ctx.lineTo(x, y);
        ctx.lineTo(x + cornerLen, y);
        ctx.stroke();
        
        // Top-right corner
        ctx.beginPath();
        ctx.moveTo(x + width - cornerLen, y);
        ctx.lineTo(x + width, y);
        ctx.lineTo(x + width, y + cornerLen);
        ctx.stroke();
        
        // Bottom-left corner
        ctx.beginPath();
        ctx.moveTo(x, y + height - cornerLen);
        ctx.lineTo(x, y + height);
        ctx.lineTo(x + cornerLen, y + height);
        ctx.stroke();
        
        // Bottom-right corner
        ctx.beginPath();
        ctx.moveTo(x + width - cornerLen, y + height);
        ctx.lineTo(x + width, y + height);
        ctx.lineTo(x + width, y + height - cornerLen);
        ctx.stroke();
        
        // Draw text background
        ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
        ctx.fillRect(x + 10, y + 10, 280, 60);
        
        // Draw text
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 16px Arial';
        ctx.fillText('TANGAN TERDETEKSI', x + 20, y + 35);
        ctx.font = 'bold 18px Arial';
        ctx.fillText(`Prediksi: ${label} (${conf.toFixed(1)}%)`, x + 20, y + 60);
    } else {
        // No hand detected
        ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
        ctx.fillRect(0, canvas.height / 2 - 40, canvas.width, 80);
        
        ctx.fillStyle = '#ff0000';
        ctx.font = 'bold 20px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('Tidak ada tangan terdeteksi', canvas.width / 2, canvas.height / 2 - 10);
        
        ctx.fillStyle = '#ffffff';
        ctx.font = '16px Arial';
        ctx.fillText('Tunjukkan tangan Anda ke kamera', canvas.width / 2, canvas.height / 2 + 20);
        ctx.textAlign = 'left';
    }
}

// ========== DETECTION LOOP ==========
async function detectionLoop() {
    if (!isDetecting) return;
    
    // Draw video frame
    ctx.drawImage(webcam, 0, 0, canvas.width, canvas.height);
    
    try {
        // Detect hand
        const handBox = await detectHand(webcam);
        
        let result = null;
        
        if (handBox) {
            // CROP hand region untuk prediksi (IMPROVED METHOD)
            const { x, y, width, height } = handBox;
            
            // 1. Draw full frame ke canvas dulu
            const fullCanvas = document.createElement('canvas');
            fullCanvas.width = webcam.videoWidth;
            fullCanvas.height = webcam.videoHeight;
            const fullCtx = fullCanvas.getContext('2d', { 
                alpha: false,
                willReadFrequently: true 
            });
            fullCtx.drawImage(webcam, 0, 0);
            
            // 2. Get ImageData dari ROI (pixel-perfect)
            const imageData = fullCtx.getImageData(x, y, width, height);
            
            // 3. Buat canvas untuk ROI
            const roiCanvas = document.createElement('canvas');
            roiCanvas.width = width;
            roiCanvas.height = height;
            const roiCtx = roiCanvas.getContext('2d');
            roiCtx.putImageData(imageData, 0, 0);
            
            // 4. Predict dari cropped canvas
            result = await predict(roiCanvas);
        }
        
        if (result) {
            const { label: rawLabel, confidence: rawConf } = result;
            
            // Stabilization logic
            if (rawConf >= CONFIDENCE_THRESHOLD * 100) {
                predictionBuffer.push({ label: rawLabel, confidence: rawConf });
                
                if (predictionBuffer.length > STABILIZER_FRAMES) {
                    predictionBuffer.shift();
                }
                
                // Voting system (sama seperti sebelumnya)
                const labelCounts = {};
                predictionBuffer.forEach(item => {
                    labelCounts[item.label] = (labelCounts[item.label] || 0) + 1;
                });
                
                let stableLabel = rawLabel;
                let maxCount = 0;
                for (const [lbl, count] of Object.entries(labelCounts)) {
                    if (count > maxCount) {
                        maxCount = count;
                        stableLabel = lbl;
                    }
                }
                
                // Average confidence for stable label
                const confidencesForStableLabel = predictionBuffer
                    .filter(item => item.label === stableLabel)
                    .map(item => item.confidence);
                
                const avgConf = confidencesForStableLabel.length > 0
                    ? confidencesForStableLabel.reduce((a, b) => a + b) / confidencesForStableLabel.length
                    : rawConf;
                
                // ⭐ TAMBAHKAN SMOOTHING: Hanya update UI jika perbedaan > 2%
                const currentConfidence = parseFloat(confidence.textContent) || 0;
                const shouldUpdate = Math.abs(avgConf - currentConfidence) > 2.0 || 
                                    predictedLabel.textContent !== stableLabel;
                
                if (shouldUpdate) {
                    predictedLabel.textContent = stableLabel;
                    confidence.textContent = `${avgConf.toFixed(1)}%`;
                    confidenceBar.style.width = `${avgConf}%`;
                }
                
                drawUI(handBox, stableLabel, avgConf);
            } else {
                // Low confidence, tapi tangan masih terdeteksi
                drawUI(handBox, '?', 0);
            }
        } else if (handBox) {
            // Tangan terdeteksi tapi tidak ada hasil prediksi
            drawUI(handBox, '?', 0);
        } else {
            // ⭐ TIDAK ADA TANGAN TERDETEKSI - RESET SEMUA
            predictionBuffer = [];  // Clear buffer
            predictedLabel.textContent = '-';
            confidence.textContent = '0%';
            confidenceBar.style.width = '0%';
            
            drawUI(null, '?', 0);  // Draw "tidak ada tangan terdeteksi"
        }
    } catch (error) {
        console.error('Detection error:', error);
    }
    
    animationId = requestAnimationFrame(detectionLoop);
}

// ========== START WEBCAM ==========
async function startWebcam() {
    try {
        updateStatus('Mengakses webcam...');
        
        const constraints = {
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 },
                facingMode: 'user'
            }
        };
        
        webcamStream = await navigator.mediaDevices.getUserMedia(constraints);
        webcam.srcObject = webcamStream;
        
        webcam.onloadedmetadata = () => {
            canvas.width = webcam.videoWidth;
            canvas.height = webcam.videoHeight;
            
            status.classList.add('hidden');
            isDetecting = true;
            
            startBtn.disabled = true;
            stopBtn.disabled = false;
            screenshotBtn.disabled = false;
            
            detectionLoop();
        };
        
    } catch (error) {
        console.error('Webcam error:', error);
        updateStatus('Error: Tidak dapat mengakses webcam!');
        alert('Gagal mengakses webcam! Pastikan browser memiliki izin untuk mengakses kamera.');
    }
}

// ========== STOP WEBCAM ==========
function stopWebcam() {
    isDetecting = false;
    
    if (animationId) {
        cancelAnimationFrame(animationId);
    }
    
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcamStream = null;
    }
    
    webcam.srcObject = null;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    status.classList.remove('hidden');
    updateStatus('Kamera dihentikan. Klik "Mulai Kamera" untuk memulai lagi.');
    
    startBtn.disabled = false;
    stopBtn.disabled = true;
    screenshotBtn.disabled = true;
    
    predictedLabel.textContent = '-';
    confidence.textContent = '0%';
    confidenceBar.style.width = '0%';
    
    predictionBuffer = [];
}

// ========== SCREENSHOT ==========
function takeScreenshot() {
    const link = document.createElement('a');
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const filename = `SIBI_${predictedLabel.textContent}_${confidence.textContent}_${timestamp}.jpg`;
    
    link.download = filename;
    link.href = canvas.toDataURL('image/jpeg');
    link.click();
    
    // Flash effect
    canvas.style.opacity = '0.5';
    setTimeout(() => { canvas.style.opacity = '1'; }, 100);
    
    console.log(`📸 Screenshot saved: ${filename}`);
}

// ========== TAB SWITCHING ==========
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const targetTab = btn.dataset.tab;
        
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });
        document.getElementById(`${targetTab}-tab`).classList.add('active');
        
        if (targetTab !== 'camera' && isDetecting) {
            stopWebcam();
        }
    });
});

// ========== UPLOAD FUNCTIONALITY ==========
uploadArea.addEventListener('click', () => fileInput.click());
selectFileBtn.addEventListener('click', () => fileInput.click());

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('drag-over');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('drag-over');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
    
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        handleImageUpload(file);
    }
});

fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        handleImageUpload(file);
    }
});

function handleImageUpload(file) {
    if (file.size > 5 * 1024 * 1024) {
        alert('Ukuran file terlalu besar! Maksimal 5MB.');
        return;
    }
    
    const reader = new FileReader();
    reader.onload = (e) => {
        uploadedImage.src = e.target.result;
        uploadedImage.style.display = 'block';
        uploadArea.querySelector('.upload-placeholder').style.display = 'none';
        
        analyzeBtn.disabled = false;
        clearBtn.disabled = false;
        
        // Reset hasil sebelumnya
        uploadPredictedLabel.textContent = '-';
        uploadConfidence.textContent = '0%';
        uploadConfidenceBar.style.width = '0%';
        topPredictions.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

// ========== ANALYZE UPLOADED IMAGE ==========
analyzeBtn.addEventListener('click', async () => {
    if (!uploadedImage.src) return;

    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = '⏳ Menganalisis...';

    try {
        await new Promise(res => {
            if (uploadedImage.complete) res();
            else uploadedImage.onload = res;
        });

        const handBox = await detectHandFromImage(uploadedImage);
        if (!handBox) throw new Error('Tangan tidak terdeteksi');

        const { x, y, width, height } = handBox;

        // Crop LANGSUNG dari gambar asli
        const cropCanvas = document.createElement('canvas');
        const cropCtx = cropCanvas.getContext('2d');

        cropCanvas.width = width;
        cropCanvas.height = height;

        cropCtx.drawImage(
            uploadedImage,
            x, y, width, height,
            0, 0, width, height
        );

        // Predict (resize dilakukan di preprocessImage)
        const result = await predict(cropCanvas);
        if (!result) throw new Error('Prediksi gagal');

        uploadPredictedLabel.textContent = result.label;
        uploadConfidence.textContent = `${result.confidence.toFixed(1)}%`;
        uploadConfidenceBar.style.width = `${result.confidence}%`;

        topPredictions.style.display = 'block';
        topPredictionsList.innerHTML = result.top5.map(
            (p, i) => `
            <div class="prediction-item">
                <span class="prediction-item-label">${i + 1}. ${p.label}</span>
                <div class="prediction-item-bar">
                    <div class="prediction-item-fill" style="width: ${p.confidence}%"></div>
                </div>
                <span class="prediction-item-confidence">${p.confidence.toFixed(1)}%</span>
            </div>`
        ).join('');

        visualizeHandBox(uploadedImage, handBox, result);

    } catch (e) {
        alert(e.message);
    }

    analyzeBtn.disabled = false;
    analyzeBtn.innerHTML = '🔍 Analisis';
});


// ========== VISUALIZE HAND BOUNDING BOX ==========
function visualizeHandBox(imageElement, handBox, result) {
    // Buat canvas overlay untuk visualisasi
    const visualCanvas = document.createElement('canvas');
    const visualCtx = visualCanvas.getContext('2d');
    
    visualCanvas.width = imageElement.naturalWidth || imageElement.width;
    visualCanvas.height = imageElement.naturalHeight || imageElement.height;
    
    // Draw original image
    visualCtx.drawImage(imageElement, 0, 0);
    
    // Draw bounding box
    const { x, y, width, height } = handBox;
    const cornerLen = 40;
    const thickness = 4;
    
    // Color based on confidence
    const conf = result.confidence;
    const color = conf >= 70 ? '#00ff00' : conf >= 50 ? '#ffff00' : '#ffa500';
    
    visualCtx.strokeStyle = color;
    visualCtx.lineWidth = thickness;
    
    // Draw corners
    // Top-left
    visualCtx.beginPath();
    visualCtx.moveTo(x, y + cornerLen);
    visualCtx.lineTo(x, y);
    visualCtx.lineTo(x + cornerLen, y);
    visualCtx.stroke();
    
    // Top-right
    visualCtx.beginPath();
    visualCtx.moveTo(x + width - cornerLen, y);
    visualCtx.lineTo(x + width, y);
    visualCtx.lineTo(x + width, y + cornerLen);
    visualCtx.stroke();
    
    // Bottom-left
    visualCtx.beginPath();
    visualCtx.moveTo(x, y + height - cornerLen);
    visualCtx.lineTo(x, y + height);
    visualCtx.lineTo(x + cornerLen, y + height);
    visualCtx.stroke();
    
    // Bottom-right
    visualCtx.beginPath();
    visualCtx.moveTo(x + width - cornerLen, y + height);
    visualCtx.lineTo(x + width, y + height);
    visualCtx.lineTo(x + width, y + height - cornerLen);
    visualCtx.stroke();
    
    // Draw label
    visualCtx.fillStyle = 'rgba(0, 0, 0, 0.7)';
    visualCtx.fillRect(x + 10, y + 10, 350, 70);
    
    visualCtx.fillStyle = '#ffffff';
    visualCtx.font = 'bold 20px Arial';
    visualCtx.fillText('TANGAN TERDETEKSI', x + 20, y + 40);
    visualCtx.font = 'bold 24px Arial';
    visualCtx.fillText(`${result.label} (${conf.toFixed(1)}%)`, x + 20, y + 70);
    
    // Replace image dengan visualisasi
    uploadedImage.src = visualCanvas.toDataURL();
    
    console.log('✓ Visualisasi bounding box selesai');
}

// ========== CLEAR UPLOAD ==========
clearBtn.addEventListener('click', () => {
    uploadedImage.src = '';
    uploadedImage.style.display = 'none';
    uploadArea.querySelector('.upload-placeholder').style.display = 'block';
    fileInput.value = '';
    
    uploadPredictedLabel.textContent = '-';
    uploadConfidence.textContent = '0%';
    uploadConfidenceBar.style.width = '0%';
    topPredictions.style.display = 'none';
    
    analyzeBtn.disabled = true;
    clearBtn.disabled = true;
});

// ========== EVENT LISTENERS ==========
startBtn.addEventListener('click', startWebcam);
stopBtn.addEventListener('click', stopWebcam);
screenshotBtn.addEventListener('click', takeScreenshot);

// ========== INITIALIZE ==========
loadModel();