import sys
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os

from model import MultimodalAgriNet

# Sınıflar (Dataset'in otomatik olarak sıraladığı hali)
CLASSES = [
    'Tomato___Bacterial_spot', 
    'Tomato___Early_blight', 
    'Tomato___Late_blight', 
    'Tomato___Leaf_Mold', 
    'Tomato___Septoria_leaf_spot', 
    'Tomato___Spider_mites Two-spotted_spider_mite', 
    'Tomato___Target_Spot', 
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 
    'Tomato___Tomato_mosaic_virus', 
    'Tomato___healthy'
]

CLASS_TO_TR = {
    'Tomato___Bacterial_spot': 'Bakteriyel Leke (Bacterial Spot)',
    'Tomato___Early_blight': 'Erken Yanıklık Mantarı (Early Blight)',
    'Tomato___Late_blight': 'Geç Yanıklık Mantarı (Late Blight)',
    'Tomato___Leaf_Mold': 'Yaprak Küfü (Leaf Mold)',
    'Tomato___Septoria_leaf_spot': 'Septoria Yaprak Lekesi (Septoria Leaf Spot)',
    'Tomato___Spider_mites Two-spotted_spider_mite': 'Kırmızı Örümcek Akarları (Spider Mites)',
    'Tomato___Target_Spot': 'Hedef Lekesi (Target Spot)',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Sarı Yaprak Kıvırcık Virüsü (Yellow Leaf Curl Virus)',
    'Tomato___Tomato_mosaic_virus': 'Domates Mozaik Virüsü (Mosaic Virus)',
    'Tomato___healthy': 'Sağlıklı (Healthy)'
}

CLASS_TO_SOLUTION = {
    'Tomato___Bacterial_spot': 'Çözüm: Bakır bazlı fungisitler kullanılmalı, hastalıklı yapraklar temizlenmeli ve sulama yapraktan (yağmurlama) değil, kökten (damlama) yapılmalıdır.',
    'Tomato___Early_blight': 'Çözüm: Etkilenen alt yapraklar budanmalı, hava sirkülasyonu artırılmalı ve klorotalonil veya bakır bazlı tarım ilaçları (fungisit) uygulanmalıdır.',
    'Tomato___Late_blight': 'Çözüm: Hastalıklı bitkiler derhal sökülüp imha edilmelidir (kompost yapılmamalı). Bitkiler arası mesafe artırılmalı ve serin/ıslak hava koşullarında koruyucu ilaçlama yapılmalıdır.',
    'Tomato___Leaf_Mold': 'Çözüm: Özellikle seralarda havalandırma artırılarak nem oranı düşürülmelidir. Alt yapraklar budanarak hava akışı sağlanmalıdır.',
    'Tomato___Septoria_leaf_spot': 'Çözüm: Damlama sulama kullanılmalı, yere düşen hastalıklı yapraklar toplanıp bahçeden uzaklaştırılmalıdır. Mantar ilaçları ile haftalık koruyucu püskürtme yapılabilir.',
    'Tomato___Spider_mites Two-spotted_spider_mite': 'Çözüm: Doğal neem yağı veya spesifik akarisit (akar ilacı) kullanılmalıdır. Çok kuru havaları sevdikleri için ortam nemi hafifçe artırılabilir.',
    'Tomato___Target_Spot': 'Çözüm: Aşırı azotlu gübrelemeden kaçınılmalı, bitkiler hava alacak şekilde aralıklı dikilmeli ve geniş spektrumlu fungisitler kullanılmalıdır.',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Çözüm: Virüsü taşıyan "beyazsinekler" ile mücadele edilmelidir (sarı yapışkan tuzaklar kullanılabilir). Enfekte olan bitkiler maalesef sökülüp atılmalıdır.',
    'Tomato___Tomato_mosaic_virus': 'Çözüm: Tedavisi yoktur. Hastalıklı bitkiler sökülüp yakılmalıdır. Virüs temasla bulaştığı için (özellikle tütün ürünleri içenlerin ellerinden) aletler çamaşır suyu ile dezenfekte edilmelidir.',
    'Tomato___healthy': 'Durum: Bitkiniz harika görünüyor! Düzenli sulama ve besin takviyesi rutininize devam edebilirsiniz.'
}

def predict_single_image(image_path, temp, humidity, model_path="multimodal_agrinet_sgd.pth"):
    print("\n[+] Model yükleniyor...")
    # 1. Cihaz seçimi (CPU/GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Modeli kur ve eğitilmiş ağırlıkları yükle
    model = MultimodalAgriNet(num_classes=10)
    
    if not os.path.exists(model_path):
        print(f"HATA: Model dosyası '{model_path}' bulunamadı. Lütfen train.py'nin bittiğinden emin olun.")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval() # Modeli tahmin moduna al
    
    # 3. Görüntüyü hazırla (Eğitimdeki gibi aynı boyut ve normalizasyon)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    print(f"[+] Görüntü işleniyor: {image_path}")
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"HATA: Görüntü yüklenirken bir sorun oluştu: {e}")
        return
        
    image_tensor = transform(image).unsqueeze(0).to(device) # Batch boyutu (1) eklendi
    
    # 4. Sensör verisini hazırla (Eğitimdeki gibi normalize ediliyor)
    # temp max 50, humidity max 100 olarak normalize edilmişti.
    norm_temp = temp / 50.0
    norm_hum = humidity / 100.0
    sensor_tensor = torch.tensor([[norm_temp, norm_hum]], dtype=torch.float32).to(device)
    
    # 5. Tahmin Yap
    print("[+] Model tahmini yapılıyor...")
    with torch.no_grad():
        output = model(image_tensor, sensor_tensor)
        probabilities = F.softmax(output[0], dim=0)
        
        # En yüksek ihtimalli sınıfı al
        max_prob, predicted_idx = torch.max(probabilities, 0)
        predicted_class = CLASSES[predicted_idx.item()]
        predicted_class_tr = CLASS_TO_TR.get(predicted_class, predicted_class)
        predicted_solution = CLASS_TO_SOLUTION.get(predicted_class, "Bu hastalık için sistemde kayıtlı bir çözüm bulunamadı.")
        
    print("\n================ TAHMİN SONUCU ================")
    print(f"Girdi Görüntüsü: {os.path.basename(image_path)}")
    print(f"Girdi Sensör Verisi: Sıcaklık {temp}°C, Nem %{humidity}")
    print("-----------------------------------------------")
    print(f"Tahmin Edilen Durum: {predicted_class_tr}")
    print(f"Güven Oranı (Confidence): %{max_prob.item() * 100:.2f}")
    print("-----------------------------------------------")
    print(f"Önerilen Müdahale:")
    print(f"{predicted_solution}")
    print("===============================================\n")

if __name__ == "__main__":
    # Kullanım: python test.py <görüntü_yolu> <sıcaklık> <nem>
    if len(sys.argv) == 4:
        img_path = sys.argv[1]
        t = float(sys.argv[2])
        h = float(sys.argv[3])
        predict_single_image(img_path, t, h)
    else:
        # Varsayılan (fallback) test çalıştırılır
        print("Parametre girilmedi. Örnek bir test çalıştırılıyor...\n")
        
        # PlantVillage/Tomato/color içerisindeki sağlıklı bir yaprak için örnek test
        default_image = r"c:\Users\senso\OneDrive\Masaüstü\YapayZeka\Tomato\color\Tomato___healthy\000146ff-92a4-4db6-90ad-8fce2ae4fddd___GH_HL Leaf 259.1.JPG"
        default_temp = 25.0
        default_hum = 60.0
        
        print(f"Kullanım İpucu: Kendi resminizi denemek için terminalde şöyle yazın:")
        print(f"python test.py resim_adi.jpg 28.5 75\n")
        
        predict_single_image(default_image, default_temp, default_hum)
