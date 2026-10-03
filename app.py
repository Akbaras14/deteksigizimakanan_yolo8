import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import os
import speech_recognition as sr
from datetime import datetime
import pandas as pd
import google.generativeai as genai
import json

# Konfigurasi halaman
st.set_page_config(
    page_title="Deteksi Makanan Indonesia",
    page_icon="🍜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# KONFIGURASI - EDIT BAGIAN INI
# ============================================================================
RESULT_FOLDER = "hasil"
MODEL_PATH = "best.pt"
GEMINI_API_KEY = ""  # <-- GANTI DENGAN API KEY ANDA
# ============================================================================

# Buat folder hasil jika belum ada
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load model dengan caching
@st.cache_resource
def load_model():
    """Load YOLO11 model"""
    try:
        model = YOLO(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Konfigurasi Gemini API
@st.cache_resource
def init_gemini():
    """Inisialisasi Gemini API"""
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-2.5-flash')
        return model, "✅ Gemini API siap digunakan"
    except Exception as e:
        return None, f"❌ Error: {e}"

# Fungsi untuk konversi gambar ke RGB
def convert_to_rgb(image):
    """Konversi gambar ke RGB (3 channel) jika perlu"""
    if image.mode == 'RGBA':
        # Buat background putih
        background = Image.new('RGB', image.size, (255, 255, 255))
        # Paste gambar dengan alpha channel sebagai mask
        background.paste(image, mask=image.split()[3])
        return background
    elif image.mode != 'RGB':
        return image.convert('RGB')
    return image

# Fungsi untuk mendapatkan info nutrisi dari Gemini
def get_nutrition_info(food_names, gemini_model):
    """Dapatkan informasi nutrisi menggunakan Gemini API"""
    try:
        foods_text = ", ".join(food_names)
        
        prompt = f"""Kamu adalah ahli gizi. Berikan informasi nutrisi untuk makanan berikut: {foods_text}

PENTING: Berikan response HANYA dalam format JSON yang valid, tanpa teks tambahan apapun.

Format JSON yang harus kamu gunakan:
{{
  "makanan": [
    {{
      "nama": "nama makanan",
      "kalori": 250,
      "protein": 12.5,
      "lemak": 8.3,
      "karbohidrat": 35.2,
      "serat": 2.1,
      "natrium": 450,
      "manfaat": ["Tinggi protein untuk pertumbuhan otot", "Sumber energi untuk aktivitas harian", "Mengandung vitamin dan mineral penting"],
      "kekurangan": ["Tinggi lemak jenuh", "Tinggi natrium dapat meningkatkan tekanan darah", "Rendah serat"],
      "sumber_data": "USDA FoodData Central"
    }}
  ]
}}

Berikan data nutrisi per porsi (100-150g) untuk setiap makanan. Berikan minimal 3 manfaat dan 2-3 kekurangan yang spesifik. Gunakan angka desimal, bukan string."""

        response = gemini_model.generate_content(prompt)
        
        # Ambil response text
        response_text = response.text.strip()
        
        # Debug: tampilkan response mentah
        # st.write("Raw response:", response_text[:200])
        
        # Bersihkan markdown code blocks
        if "```json" in response_text:
            # Ekstrak JSON dari code block
            start = response_text.find("```json") + 7
            end = response_text.rfind("```")
            response_text = response_text[start:end].strip()
        elif "```" in response_text:
            start = response_text.find("```") + 3
            end = response_text.rfind("```")
            response_text = response_text[start:end].strip()
        
        # Hapus karakter non-JSON di awal
        response_text = response_text.lstrip()
        if not response_text.startswith("{"):
            # Cari posisi pertama {
            json_start = response_text.find("{")
            if json_start != -1:
                response_text = response_text[json_start:]
        
        # Parse JSON
        nutrition_data = json.loads(response_text)
        
        # Validasi struktur
        if "makanan" not in nutrition_data or not isinstance(nutrition_data["makanan"], list):
            raise ValueError("Format JSON tidak valid")
        
        return nutrition_data
    
    except json.JSONDecodeError as e:
        st.error(f"❌ Error parsing JSON: {e}")
        st.error("Response dari Gemini tidak dalam format JSON yang valid")
        
        # Fallback: buat data manual berdasarkan nama makanan
        return create_fallback_nutrition(food_names)
    
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return create_fallback_nutrition(food_names)

# Fungsi fallback jika Gemini gagal
def create_fallback_nutrition(food_names):
    """Buat data nutrisi fallback sederhana"""
    makanan_list = []
    
    for food in food_names:
        makanan_list.append({
            "nama": food,
            "kalori": 200,
            "protein": 10,
            "lemak": 5,
            "karbohidrat": 30,
            "serat": 2,
            "natrium": 300,
            "manfaat": ["Sumber energi", "Mengandung nutrisi penting", "Mudah dicerna"],
            "kekurangan": ["Data nutrisi tidak lengkap", "Perlu verifikasi lebih lanjut"],
            "sumber_data": "Estimasi umum (Gemini API tidak tersedia)"
        })
    
    return {"makanan": makanan_list}

# Fungsi untuk deteksi objek
def detect_objects(image, model, conf_threshold, iou_threshold):
    """Deteksi objek pada gambar"""
    # PERBAIKAN: Konversi gambar ke RGB terlebih dahulu
    image = convert_to_rgb(image)
    image_np = np.array(image)
    
    results = model.predict(
        image_np,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False
    )
    
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            detection = {
                "class_id": int(box.cls[0]),
                "class_name": model.names[int(box.cls[0])],
                "confidence": float(box.conf[0]),
                "bbox": {
                    "x1": int(box.xyxy[0][0]),
                    "y1": int(box.xyxy[0][1]),
                    "x2": int(box.xyxy[0][2]),
                    "y2": int(box.xyxy[0][3])
                }
            }
            detections.append(detection)
    
    return detections

# Fungsi untuk gambar bounding box
def draw_boxes(image, detections):
    """Gambar bounding box pada image"""
    # PERBAIKAN: Pastikan gambar dalam format RGB
    image = convert_to_rgb(image)
    image_np = np.array(image)
    image_with_boxes = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    for det in detections:
        bbox = det["bbox"]
        x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
        conf = det["confidence"]
        class_name = det["class_name"]
        
        cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        label = f"{class_name}: {conf:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        cv2.rectangle(image_with_boxes, 
                    (x1, y1 - label_size[1] - 10),
                    (x1 + label_size[0], y1),
                    (0, 255, 0), -1)
        
        cv2.putText(image_with_boxes, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    image_with_boxes = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
    return image_with_boxes

# Fungsi untuk simpan hasil
def save_result(image_with_boxes, original_filename):
    """Simpan hasil deteksi ke folder hasil"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    original_name = os.path.splitext(original_filename)[0]
    result_filename = f"{original_name}_{timestamp}_detected.jpg"
    result_path = os.path.join(RESULT_FOLDER, result_filename)
    
    image_bgr = cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR)
    cv2.imwrite(result_path, image_bgr)
    
    return result_filename, result_path

# Fungsi untuk tampilkan info nutrisi
def display_nutrition_info(nutrition_data):
    """Tampilkan informasi nutrisi dengan tampilan menarik"""
    if not nutrition_data or "makanan" not in nutrition_data:
        st.warning("⚠️ Data nutrisi tidak tersedia")
        return
    
    st.markdown("### 🥗 Informasi Nutrisi")
    
    for food in nutrition_data["makanan"]:
        with st.expander(f"📊 {food['nama'].upper()}", expanded=True):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("#### Kandungan Nutrisi (per porsi)")
                
                # Buat dataframe nutrisi
                nutrisi_df = pd.DataFrame([
                    {"Nutrisi": "🔥 Kalori", "Jumlah": f"{food['kalori']} kcal"},
                    {"Nutrisi": "💪 Protein", "Jumlah": f"{food['protein']} g"},
                    {"Nutrisi": "🥑 Lemak", "Jumlah": f"{food['lemak']} g"},
                    {"Nutrisi": "🍚 Karbohidrat", "Jumlah": f"{food['karbohidrat']} g"},
                    {"Nutrisi": "🌾 Serat", "Jumlah": f"{food['serat']} g"},
                    {"Nutrisi": "🧂 Natrium", "Jumlah": f"{food['natrium']} mg"},
                ])
                
                st.dataframe(nutrisi_df, use_container_width=True, hide_index=True)
                
                # Sumber data
                st.caption(f"📚 Sumber: {food['sumber_data']}")
            
            with col2:
                st.markdown("#### 💚 Manfaat")
                for manfaat in food['manfaat']:
                    st.markdown(f"✅ {manfaat}")
                
                st.markdown("#### ⚠️ Kekurangan")
                for kekurangan in food.get('kekurangan', []):
                    st.markdown(f"⚠️ {kekurangan}")

# Main App
def main():
    # Header
    st.title("🍜 Deteksi Makanan Indonesia dengan Analisis Nutrisi")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Pengaturan")
        
        # Load model YOLO
        model = load_model()
        
        if model is None:
            st.error("❌ Model tidak dapat dimuat!")
            st.stop()
        else:
            st.success("✅ Model YOLO berhasil dimuat")
        
        # Load Gemini
        gemini_model, gemini_status = init_gemini()
        
        if gemini_model:
            st.success(gemini_status)
        else:
            st.warning(gemini_status)
        
        st.markdown("---")
        
        # Parameter deteksi
        st.subheader("Parameter Deteksi")
        
        # Penjelasan parameter
        with st.expander("ℹ️ Penjelasan Parameter", expanded=False):
            st.markdown("""
            ### 🎯 Confidence Threshold
            **Apa itu?**  
            Tingkat keyakinan model bahwa objek yang terdeteksi benar-benar ada.
            
            **Contoh:**
            - Nilai **0.5** = Model 50% yakin ada makanan
            - Nilai **0.9** = Model 90% yakin ada makanan
            
            **Kapan diubah?**
            - **Naikkan** (0.6-0.9): Jika terlalu banyak deteksi salah
            - **Turunkan** (0.2-0.4): Jika ada makanan yang terlewat
            
            ---
            
            ### 📐 IoU Threshold (Intersection over Union)
            **Apa itu?**  
            Mengatur seberapa banyak overlap antara 2 kotak deteksi yang dianggap duplikat.
            
            **Contoh:**
            - Nilai **0.45**: Jika 2 kotak overlap >45%, hapus yang confidence-nya lebih rendah
            - Nilai **0.7**: Butuh overlap >70% baru dihapus
            
            **Kapan diubah?**
            - **Naikkan** (0.6-0.8): Jika ada banyak kotak pada 1 objek yang sama
            - **Turunkan** (0.3-0.4): Jika ada objek berbeda yang berdekatan terdeteksi sebagai 1
            
            ---
            
            **💡 Tips:** Gunakan mode otomatis untuk hasil optimal!
            """)
        
        st.markdown("")
        
        # Mode otomatis atau manual
        auto_mode = st.toggle("Mode Otomatis", value=True, help="Gunakan parameter optimal otomatis")
        
        if auto_mode:
            conf_threshold = 0.5
            iou_threshold = 0.45
            
            st.info("🤖 **Mode Otomatis Aktif**")
            st.write(f"• Confidence: **{conf_threshold}** (keyakinan minimal 50%)")
            st.write(f"• IoU: **{iou_threshold}** (standard untuk NMS)")
        else:
            st.info("⚙️ **Mode Manual Aktif**")
            conf_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.1,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="Threshold minimum untuk confidence score"
            )
            
            iou_threshold = st.slider(
                "IoU Threshold",
                min_value=0.1,
                max_value=1.0,
                value=0.45,
                step=0.05,
                help="Threshold untuk Non-Maximum Suppression"
            )
        
        save_results = st.checkbox("Simpan hasil deteksi", value=True)
        
        st.markdown("---")
        
        # Info Model
        st.subheader("📊 Info Model")
        st.write(f"**Total Kelas:** {len(model.names)}")
        
        with st.expander("Lihat Daftar Kelas"):
            for idx, name in model.names.items():
                st.write(f"{idx}: {name}")
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📸 Deteksi Single", "📁 Deteksi Batch", "🗂️ Hasil Tersimpan"])
    
    # Tab 1: Single Detection
    with tab1:
        st.header("Deteksi dari Single Gambar")
        
        uploaded_file = st.file_uploader(
            "Upload gambar makanan",
            type=["jpg", "jpeg", "png"],
            help="Upload gambar makanan Indonesia untuk dideteksi"
        )
        
        if uploaded_file is not None:
            # Load image
            image = Image.open(uploaded_file)
            
            # Display original image
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Gambar Original")
                st.image(image, width=400)
            
            # Tombol deteksi
            if st.button("🔍 Deteksi Makanan", type="primary", use_container_width=True):
                with st.spinner("Sedang mendeteksi..."):
                    # Deteksi
                    detections = detect_objects(image, model, conf_threshold, iou_threshold)
                    
                    if len(detections) > 0:
                        # Gambar bounding box
                        image_with_boxes = draw_boxes(image, detections)
                        
                        with col2:
                            st.subheader("Hasil Deteksi")
                            st.image(image_with_boxes, width=400)
                        
                        # Simpan hasil
                        if save_results:
                            result_filename, result_path = save_result(
                                image_with_boxes, 
                                uploaded_file.name
                            )
                            st.success(f"✅ Hasil disimpan: `{result_filename}`")
                        
                        # Tampilkan hasil deteksi
                        st.markdown("---")
                        st.subheader(f"📋 Ditemukan {len(detections)} objek makanan")
                        
                        # Buat dataframe
                        df_data = []
                        for i, det in enumerate(detections, 1):
                            df_data.append({
                                "No": i,
                                "Makanan": det["class_name"],
                                "Confidence": f"{det['confidence']:.2%}",
                                "Posisi": f"({det['bbox']['x1']}, {det['bbox']['y1']}) - ({det['bbox']['x2']}, {det['bbox']['y2']})"
                            })
                        
                        df = pd.DataFrame(df_data)
                        st.dataframe(df, use_container_width=True, hide_index=True)
                        
                        # Analisis Nutrisi dengan Gemini
                        if gemini_model:
                            st.markdown("---")
                            with st.spinner("🤖 Menganalisis nutrisi dengan Gemini AI..."):
                                food_names = [det["class_name"] for det in detections]
                                nutrition_data = get_nutrition_info(food_names, gemini_model)
                                
                                if nutrition_data:
                                    display_nutrition_info(nutrition_data)
                        else:
                            st.info("💡 Atur GEMINI_API_KEY di code untuk mendapatkan analisis nutrisi")
                        
                    else:
                        st.warning("⚠️ Tidak ada objek makanan terdeteksi. Coba turunkan confidence threshold.")
    
    # Tab 2: Batch Detection
    with tab2:
        st.header("Deteksi dari Multiple Gambar")
        
        uploaded_files = st.file_uploader(
            "Upload multiple gambar",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            help="Upload beberapa gambar sekaligus untuk dideteksi"
        )
        
        if uploaded_files:
            st.info(f"📁 {len(uploaded_files)} gambar telah diupload")
            
            if st.button("🔍 Deteksi Semua Gambar", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                results_summary = []
                all_food_names = []
                
                for idx, uploaded_file in enumerate(uploaded_files):
                    progress = (idx + 1) / len(uploaded_files)
                    progress_bar.progress(progress)
                    
                    image = Image.open(uploaded_file)
                    detections = detect_objects(image, model, conf_threshold, iou_threshold)
                    
                    if len(detections) > 0:
                        image_with_boxes = draw_boxes(image, detections)
                        
                        if save_results:
                            result_filename, result_path = save_result(
                                image_with_boxes,
                                uploaded_file.name
                            )
                        
                        food_names = [d["class_name"] for d in detections]
                        all_food_names.extend(food_names)
                        
                        results_summary.append({
                            "File": uploaded_file.name,
                            "Jumlah Objek": len(detections),
                            "Makanan": ", ".join(food_names)
                        })
                    else:
                        results_summary.append({
                            "File": uploaded_file.name,
                            "Jumlah Objek": 0,
                            "Makanan": "-"
                        })
                
                progress_bar.empty()
                st.success(f"✅ Selesai! {len(uploaded_files)} gambar telah diproses")
                
                # Tampilkan summary
                st.markdown("---")
                st.subheader("📊 Ringkasan Hasil")
                df_summary = pd.DataFrame(results_summary)
                st.dataframe(df_summary, use_container_width=True, hide_index=True)
                
                # Analisis Nutrisi untuk semua makanan
                if gemini_model and all_food_names:
                    st.markdown("---")
                    with st.spinner("🤖 Menganalisis nutrisi dengan Gemini AI..."):
                        unique_foods = list(set(all_food_names))
                        nutrition_data = get_nutrition_info(unique_foods, gemini_model)
                        
                        if nutrition_data:
                            display_nutrition_info(nutrition_data)
    
    # Tab 3: Saved Results
    with tab3:
        st.header("File Hasil Tersimpan")
        
        try:
            files = os.listdir(RESULT_FOLDER)
            image_files = sorted(
                [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))],
                reverse=True
            )
            
            if image_files:
                st.info(f"📁 Total: {len(image_files)} file")
                
                col_btn1, col_btn2 = st.columns([1, 4])
                with col_btn1:
                    if st.button("🗑️ Hapus Semua", type="secondary"):
                        for file in image_files:
                            os.remove(os.path.join(RESULT_FOLDER, file))
                        st.success("✅ Semua file berhasil dihapus!")
                        st.rerun()
                
                st.markdown("---")
                
                cols = st.columns(3)
                for idx, filename in enumerate(image_files):
                    with cols[idx % 3]:
                        file_path = os.path.join(RESULT_FOLDER, filename)
                        image = Image.open(file_path)
                        
                        st.image(image, caption=filename, width=250)
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            with open(file_path, "rb") as file:
                                btn = st.download_button(
                                    label="⬇️ Download",
                                    data=file,
                                    file_name=filename,
                                    mime="image/jpeg",
                                    use_container_width=True
                                )
                        with col_b:
                            if st.button("🗑️", key=f"del_{idx}", use_container_width=True):
                                os.remove(file_path)
                                st.success(f"✅ {filename} dihapus!")
                                st.rerun()
            else:
                st.warning("📭 Belum ada file hasil tersimpan")
        
        except Exception as e:
            st.error(f"❌ Error: {e}")

if __name__ == "__main__":
    main()