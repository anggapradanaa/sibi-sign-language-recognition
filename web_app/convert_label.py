import numpy as np
import json

# Path file .npy asli Anda (sesuaikan jika perlu)
LABELS_PATH = r"D:\Perkuliahan\Semester 7\Skripsi\Sistem Isyarat Bahasa Indonesia (SIBI)\ResNet50\labels_list_v4.npy"
OUTPUT_JSON = "labels.json"

try:
    # Load numpy array
    labels_array = np.load(LABELS_PATH, allow_pickle=True)
    
    # Konversi ke list Python standar
    labels_list = labels_array.tolist()
    
    # Simpan sebagai JSON
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(labels_list, f)
        
    print(f"✅ Berhasil membuat {OUTPUT_JSON}")
    print(f"📝 Contoh isi: {labels_list[:5]}...")
    print(f"📊 Total kelas: {len(labels_list)}")

except Exception as e:
    print(f"❌ Error: {e}")