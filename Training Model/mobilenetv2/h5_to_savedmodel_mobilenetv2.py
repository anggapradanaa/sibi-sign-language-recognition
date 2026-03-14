import numpy as np
import h5py
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalMaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
import os

# ========== KONFIGURASI PATH ==========
OLD_MODEL_PATH = r"D:\Perkuliahan\Semester 7\Skripsi\Sistem Isyarat Bahasa Indonesia (SIBI)\MobileNetV2\sibi_mobilenetv2_model.h5"
NEW_MODEL_PATH = r"D:\Perkuliahan\Semester 7\Skripsi\Sistem Isyarat Bahasa Indonesia (SIBI)\MobileNetV2\sibi_mobilenetv2_savedmodel"
LABELS_PATH = r"D:\Perkuliahan\Semester 7\Skripsi\Sistem Isyarat Bahasa Indonesia (SIBI)\MobileNetV2\labels_list.npy"

print("="*60)
print("REBUILD MODEL MOBILENETV2 DARI WEIGHTS")
print("="*60)

# Load labels untuk mengetahui jumlah kelas
print("\n[1/5] Loading labels...")
try:
    labels_list = np.load(LABELS_PATH, allow_pickle=True)
    num_classes = len(labels_list)
    print(f"✓ Labels berhasil dimuat!")
    print(f"   Jumlah kelas: {num_classes}")
    print(f"   Kelas: {labels_list}")
except Exception as e:
    print(f"✗ Error: {e}")
    exit()

# Rebuild arsitektur model (sama persis dengan script training)
print("\n[2/5] Rebuilding model architecture...")
try:
    base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    
    # Freeze layers seperti di training
    for layer in base_model.layers[:-60]:  
        layer.trainable = False
    
    model = Sequential([
        base_model,
        GlobalMaxPooling2D(),
        BatchNormalization(),
        Dense(1024, activation='relu'),  
        Dropout(0.3),                    
        BatchNormalization(),
        Dense(512, activation='relu'),   
        Dense(num_classes, activation='softmax')
    ])
    
    print("✓ Model architecture berhasil di-rebuild!")
    print(f"   Total layers: {len(model.layers)}")
    
except Exception as e:
    print(f"✗ Error rebuilding model: {e}")
    exit()

# Compile model
print("\n[3/5] Compiling model...")
try:
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print("✓ Model berhasil di-compile!")
except Exception as e:
    print(f"✗ Error: {e}")
    exit()

# Load weights dari file .h5 lama
print("\n[4/5] Loading weights dari model lama...")
try:
    # Buka file h5
    with h5py.File(OLD_MODEL_PATH, 'r') as f:
        # Cek struktur file
        if 'model_weights' in f.keys():
            weight_group = f['model_weights']
        else:
            weight_group = f
        
        print(f"   Ditemukan {len(weight_group.keys())} layer groups")
        
        # Load weights layer by layer
        try:
            model.load_weights(OLD_MODEL_PATH, by_name=True, skip_mismatch=True)
            print("✓ Weights berhasil dimuat menggunakan load_weights!")
        except Exception as e1:
            print(f"   load_weights gagal: {e1}")
            print("   Mencoba metode alternatif...")
            
            # Metode alternatif: manual loading
            try:
                # Load semua weights
                all_weights = []
                def load_weights_recursive(group):
                    for key in group.keys():
                        item = group[key]
                        if isinstance(item, h5py.Group):
                            load_weights_recursive(item)
                        elif isinstance(item, h5py.Dataset):
                            all_weights.append(np.array(item))
                
                load_weights_recursive(weight_group)
                
                # Set weights ke model
                if len(all_weights) == len(model.weights):
                    model.set_weights(all_weights)
                    print(f"✓ Weights berhasil dimuat secara manual!")
                    print(f"   Total weights loaded: {len(all_weights)}")
                else:
                    print(f"⚠ Warning: Jumlah weights tidak match ({len(all_weights)} vs {len(model.weights)})")
                    print("   Model akan menggunakan random weights untuk layer yang tidak match")
                    
            except Exception as e2:
                print(f"✗ Metode alternatif juga gagal: {e2}")
                print("⚠ Model akan disimpan dengan weights dari ImageNet + random weights untuk top layers")
    
except Exception as e:
    print(f"✗ Error loading weights: {e}")
    print("⚠ Model akan disimpan dengan weights default")

# Save model baru
print("\n[5/5] Menyimpan model dalam format SavedModel...")
try:
    # Hapus folder lama jika ada
    if os.path.exists(NEW_MODEL_PATH):
        import shutil
        print(f"   Menghapus folder lama...")
        shutil.rmtree(NEW_MODEL_PATH)
    
    # Save dalam format SavedModel
    model.save(NEW_MODEL_PATH, save_format='tf')
    print(f"✓ Model berhasil disimpan!")
    print(f"   Lokasi: {NEW_MODEL_PATH}")
    
    # Verifikasi model bisa di-load
    print("\n[Verifikasi] Testing load model...")
    test_model = tf.keras.models.load_model(NEW_MODEL_PATH)
    print(f"✓ Model baru berhasil di-load untuk verifikasi!")
    print(f"   Input shape: {test_model.input_shape}")
    print(f"   Output shape: {test_model.output_shape}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    exit()

print("\n" + "="*60)
print("REBUILD SELESAI!")
print("="*60)
print("\n⚠ PENTING:")
print("Jika weights berhasil dimuat dari model lama, model siap digunakan.")
print("Jika weights gagal dimuat, Anda perlu RE-TRAINING model.")
print("\nLangkah selanjutnya:")
print("1. Jalankan: python deploy.py")
print("2. Jika prediksi tidak akurat, RE-TRAIN model dengan script training")
print("="*60)