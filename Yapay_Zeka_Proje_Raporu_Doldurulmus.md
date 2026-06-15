# PROJE RAPORU

**1. Proje Başlığı**
Multimodal Akıllı Tarım Asistanı: Görüntü ve Çevresel Sensör Verisi Füzyonu ile Bitki Hastalığı Tespiti

**2. Proje Ekibi**
- **Ad Soyad:** [Adınızı Soyadınızı Yazınız]
- **Öğrenci Numarası:** [Numaranızı Yazınız]
- **Bölüm / Sınıf:** [Bölüm ve Sınıfınızı Yazınız]
- **Ekipteki Görev(ler):** Veri setinin hazırlanması, model mimarisinin oluşturulması, karşılaştırmalı deneylerin (Ablation Study) yürütülmesi ve raporlama.

**3. Proje Özeti (En fazla 150 kelime)**
Bu proje, tarımsal verimi düşüren bitki hastalıklarının tespitini yalnızca yaprak görüntüleri ile değil, o anki çevresel koşulları da (sıcaklık ve nem) sürece katarak çok daha yüksek doğrulukla gerçekleştirmeyi amaçlamaktadır. Projede PlantVillage (Domates) veri seti kullanılmış olup, hastalık sınıflarının biyolojik doğasına uygun sentetik sensör verileri entegre edilmiştir. Görüntü öznitelikleri önceden eğitilmiş ResNet18 mimarisi ile çıkarılırken, sensör verileri Çok Katmanlı Algılayıcı (MLP) ile işlenmiş ve 'Erken Füzyon' (Early Fusion) tekniği ile birleştirilmiştir. Modelin doğruluğunu akademik olarak ispatlamak için aktivasyon fonksiyonları (ReLU vs GELU), optimizasyon algoritmaları (Adam vs SGD) ve donanımsal etkiler (Sensörlü vs Sensörsüz) üzerine çoklu karşılaştırma deneyleri yapılmış ve nihai modelde %99 eğitim, %98.5+ test başarı oranlarına ulaşılmıştır.

**4. Giriş**
- **Projenin konusu ve amacı:** Bitki yapraklarından alınan görüntüler ile çevresel koşulların (sıcaklık, nem) bir yapay sinir ağı içerisinde harmanlanarak hastalıkların erken teşhis edilmesi.
- **Neden bu konuyu seçtiniz?:** Literatürdeki mevcut çözümlerin çoğu sadece görsel veriye dayanmaktadır. Ancak gerçek dünyada hastalıkların (özellikle mantar enfeksiyonlarının) oluşumu doğrudan ortam nemi ve sıcaklığıyla bağlantılıdır. Sadece görüntü kalitesinin kötü olduğu sera koşullarında sistemin çökmemesi için modele sensör "farkındalığı" kazandırmak istedim.
- **Çalışmanın kapsamı ve sınırlılıkları:** Kapsam, domates bitkisine ait 10 farklı durumu (9 hastalık + 1 sağlıklı) içermektedir. En büyük sınırlılık gerçek bir IoT altyapısı kurulamadığı için sensör verilerinin Gauss dağılımı ile istatistiksel olarak simüle edilmiş olmasıdır.

**5. Kullanılan Yöntem ve Teknolojiler**
- **Kullandığınız algoritmalar / modeller:** Evrişimli Sinir Ağları (CNN) tabanlı ResNet18 ve MLP (Çok Katmanlı Algılayıcı). Erken Birleştirme (Early Fusion) Mimarisi.
- **Uygulama dili, kütüphaneler, araçlar:** Python, PyTorch, Torchvision, Matplotlib, Google Colab (T4 GPU).
- **Veri seti:** Kaggle PlantVillage (Domates yaprakları) ve biyolojik gereksinimlere göre (örn. Geç Yanıklık için yüksek nem, düşük sıcaklık) Python üzerinde üretilen matematiksel sensör simülasyonları.

**6. Deneysel Karşılaştırmalar ve Analizler (Ablation Study)**
Projenin mimari kararlarını bilimsel bir temele oturtmak için Google Colab üzerinde T4 GPU kullanılarak 3 farklı karşılaştırma deneyi (30'ar Epoch) yürütülmüştür:

- **6.1. Multimodal (Sensörlü) vs Unimodal (Sadece Görüntü) Karşılaştırması:**
  Sensör verilerini modele dahil etmenin etkisini ölçmek için "Ablasyon Deneyi" yapılmıştır. Salt görüntü modeli çok istikrarlı bir öğrenme sergilese de, sensör verisi eklenen model başlangıçta farklı veri tiplerini harmanlarken (early fusion) yüksek varyans (dalgalanma) yaşamıştır. Ancak ilerleyen epoch'larda öğrenmeyi dengeleyip %98.5 başarıya ulaşmıştır. Bu deney, sensör verisinin kameranın net göremediği ortamlarda "güvenlik ağı" görevi göreceği hipotezini doğrulamıştır.

- **6.2. Aktivasyon Fonksiyonu: ReLU vs. GELU:**
  Sensör verilerini birleştiren MLP katmanında modern GELU ile klasik ReLU karşılaştırılmıştır. GELU aktivasyonu eğitimin 27. adımında ciddi bir istikrarsızlık (Test kaybında 0.27'ye varan ani sıçramalar ve başarıda %92'ye düşüş) göstermiştir. Görece sığ (shallow) bir ağ olan füzyon katmanımızda doğrusal olmayan özellikleri daha kesin çizen ReLU'nun çok daha kararlı (stable) olduğu kanıtlanmış ve nihai model ReLU üzerine kurulmuştur.

- **6.3. Optimizasyon Algoritması: Adam vs. SGD (Momentum):**
  Kazanan ReLU mimarisi üzerinde Adam ve SGD algoritmaları yarıştırılmıştır. Adam hızlı öğrense de validasyon (test) kaybında ani sıçramalar yapmıştır. SGD (Momentum: 0.9) ise biraz daha yavaş başlamasına rağmen test başarısını çok hızlı bir şekilde %99 bandına oturtmuş ve neredeyse sıfır dalgalanma (sıfır gürültü) ile en kusursuz öğrenme eğrisini çizmiştir.

**7. Sonuçlar ve Değerlendirme**
- **Elde edilen çıktılar:** Nihai modelimiz (ReLU aktivasyonlu, SGD optimizer destekli, Sensörlü Multimodal Ağ), eğitimde %99.6, test setinde ise %98.5 ile %99 arasında muazzam bir başarı oranına ulaşmıştır. 
- **Yöntemin başarısı ve sınırlılıkları:** Kurulan bu hibrit sistem hem yenilikçi hem de çok başarılıdır. Görüntünün yanıltıcı olabileceği durumlarda sensör verisi karar mekanizmasını düzeltmektedir. Tek sınırlılık, modelin henüz gerçek zamanlı veri akışı sağlayan bir sahada (IoT ortamında) denenmemiş olmasıdır.

**8. Yenilikçilik / Özgünlük Açıklaması**
Çoğu ders projesinde sadece hazır bir yapay zeka (örn. YOLO) çalıştırılarak sonuç alınmaktadır. Bu projede ise:
1. İki farklı veri tipi (Görsel + Sensör) sıfırdan tasarlanan bir sinir ağında Early Fusion ile birleştirilmiştir.
2. Modelin oluşturulma sürecinde "Neden bu algoritmayı seçtin?" sorusuna karşılık olarak; çoklu hiperparametre deneyleri (Ablation Study) yapılmış ve matematiksel grafiklerle savunulabilir bilimsel bir metodoloji izlenmiştir.

**9. Gelecek Çalışmalar**
- Sisteme bir Raspberry Pi entegre edilerek seralarda canlı olarak çalışan uçtan uca (end-to-end) bir IoT tarım asistanı geliştirilebilir.
- Toprak pH'ı ve rüzgar hızı gibi ekstra boyutlar füzyon katmanına kolaylıkla eklenebilir.

**10. Kaynakça**
- Hughes, D., & Salathe, M. (2015). An open access repository of images on plant health to enable the development of mobile disease diagnostics. arXiv preprint arXiv:1511.08060.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE CVPR (pp. 770-778).
- PyTorch Documentation: https://pytorch.org/docs/stable/index.html
