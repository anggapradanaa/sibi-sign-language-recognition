# 🧠 Training Model

Folder ini berisi pipeline pelatihan model untuk proyek **SIBI Sign Language Recognition**. Tiga arsitektur CNN yang diuji dalam penelitian ini adalah **ResNet50**, **VGG16**, dan **MobileNetV2**.

---

## ⚙️ Proses Training Model

Setiap model dilatih menggunakan pendekatan **transfer learning** dengan backbone pretrained dari ImageNet. Setelah proses training selesai, model pertama kali disimpan dalam format **Keras `.h5`**.

Untuk keperluan deployment dan kompatibilitas dengan proses konversi selanjutnya, file `.h5` kemudian dikonversi menjadi format **TensorFlow SavedModel** menggunakan script berikut:

- `mobilenetv2/h5_to_savedmodel_mobilenetv2.py`
- `resnet50/h5_to_savedmodel_resnet50.py`
- `vgg16/h5_to_savedmodel_vgg16.py`

---

## 🌐 Konversi Model untuk Web

Karena aplikasi akhir berjalan di browser menggunakan **TensorFlow.js**, model dalam format SavedModel perlu dikonversi kembali menjadi **format TensorFlow.js**.

Pipeline konversi model pada proyek ini adalah sebagai berikut:

Training → `.h5` model → **TensorFlow SavedModel** → **TensorFlow.js model**

Dari ketiga arsitektur yang diuji, **ResNet50 menghasilkan performa terbaik**, sehingga model ini dipilih sebagai model final dan dikonversi ke format TensorFlow.js untuk digunakan pada aplikasi web.

Proses konversi tersebut dilakukan pada file berikut:

`savedmodel_to_js_resnet50.ipynb`

Model akhir yang digunakan oleh aplikasi web berada pada folder:

`web_app/model_web/`

---

## 📊 Performa Model

| Model | Validation Accuracy | Test Accuracy |
|------|--------------------|---------------|
| **ResNet50** | 0.9955 | 0.9933 |
| **VGG16** | 0.9933 | 0.9910 |
| **MobileNetV2** | 0.9933 | 0.9865 |

---

## 📝 Ringkasan

Tiga arsitektur CNN dilatih dan dievaluasi pada dataset yang sama. **ResNet50** menunjukkan performa terbaik sehingga dipilih sebagai model final dan digunakan pada sistem inferensi berbasis web menggunakan **TensorFlow.js**.
