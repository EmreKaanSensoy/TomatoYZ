import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt

# Notebook'ta çalıştırırken bu importları silebilir/kapatabilirsiniz
from dataset import MultimodalAgriDataset
from model import MultimodalAgriNet

def main():
    # =========================================================================
    # 1. DENEY AYARLARI (Karşılaştırmalar için burayı değiştirin)
    # =========================================================================
    USE_SENSORS = True        # Sensörü kapatıp Sadece Görüntü (Ablasyon) denemek için False yapın.
    ACTIVATION = 'relu'       # 'relu', 'gelu' veya 'leakyrelu' yapabilirsiniz.
    OPTIMIZER_TYPE = 'adam'   # 'adam' veya 'sgd' yapabilirsiniz.
    
    # Kendi dizininizi buraya yazın (Örn: /content/Tomato/color)
    data_dir = r"/content/Tomato/color" 
    batch_size = 32
    num_epochs = 30
    learning_rate = 0.001
    # =========================================================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- DENEY BAŞLIYOR ---")
    print(f"Cihaz: {device} | Sensörler Açık Mı?: {USE_SENSORS} | Aktivasyon: {ACTIVATION.upper()} | Optimizer: {OPTIMIZER_TYPE.upper()}")

    # 2. Data Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    print("Veri seti yükleniyor...")
    full_dataset = MultimodalAgriDataset(root_dir=data_dir, transform=transform)

    num_classes = len(full_dataset.classes)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # 4. Modeli yeni ayarlarla oluştur
    model = MultimodalAgriNet(num_classes=num_classes, use_sensor=USE_SENSORS, activation=ACTIVATION).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Seçilen Optimizer'ı kullan
    if OPTIMIZER_TYPE == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Grafikler İçin Geçmiş Listeleri
    history_train_loss, history_val_loss = [], []
    history_train_acc, history_val_acc = [], []

    # 5. Eğitim Döngüsü
    print("Eğitim başlatılıyor...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for i, (images, sensors, labels) in enumerate(train_loader):
            images, sensors, labels = images.to(device), sensors.to(device), labels.to(device)

            outputs = model(images, sensors)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Doğruluk ve Kayıp Değerlerini Hesapla
        train_loss_epoch = running_loss / len(train_loader)
        train_acc_epoch = 100 * correct / total

        # --- Validation (Doğrulama) Döngüsü ---
        model.eval()
        val_loss = 0.0
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, sensors, labels in val_loader:
                images, sensors, labels = images.to(device), sensors.to(device), labels.to(device)
                outputs = model(images, sensors)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss_epoch = val_loss / len(val_loader)
        val_acc_epoch = 100 * val_correct / val_total
        
        # Grafik için verileri kaydet
        history_train_loss.append(train_loss_epoch)
        history_val_loss.append(val_loss_epoch)
        history_train_acc.append(train_acc_epoch)
        history_val_acc.append(val_acc_epoch)

        print(f"Epoch [{epoch+1}/{num_epochs}] -> Train Loss: {train_loss_epoch:.4f}, Train Acc: {train_acc_epoch:.2f}% | Val Loss: {val_loss_epoch:.4f}, Val Acc: {val_acc_epoch:.2f}%")

    print("\nEğitim Tamamlandı!")
    
    # -------------------------------------------------------------
    # 6. GRAFİKLERİN ÇİZİLMESİ (Matplotlib)
    # -------------------------------------------------------------
    epochs_range = range(1, num_epochs + 1)
    plt.figure(figsize=(14, 5))
    
    # Acc (Doğruluk) Grafiği
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history_train_acc, label='Eğitim Başarısı (Train Acc)', color='blue', marker='o')
    plt.plot(epochs_range, history_val_acc, label='Test Başarısı (Val Acc)', color='green', marker='s')
    plt.title(f'Model Doğruluğu ({ACTIVATION.upper()} - {OPTIMIZER_TYPE.upper()} - Sensör:{USE_SENSORS})')
    plt.xlabel('Epoch (Adım)')
    plt.ylabel('Başarı (%)')
    plt.legend(loc='lower right')
    plt.grid(True)

    # Loss (Kayıp) Grafiği
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history_train_loss, label='Eğitim Kaybı (Train Loss)', color='red', marker='o')
    plt.plot(epochs_range, history_val_loss, label='Test Kaybı (Val Loss)', color='orange', marker='s')
    plt.title(f'Model Kaybı (Loss Curve)')
    plt.xlabel('Epoch (Adım)')
    plt.ylabel('Kayıp Değeri')
    plt.legend(loc='upper right')
    plt.grid(True)

    plt.tight_layout()
    # Grafiği dosya olarak kaydet
    plt.savefig('egitim_grafikleri.png', dpi=300)
    plt.show() # Colab'da resmi direkt ekranda gösterir
    print("Grafikler 'egitim_grafikleri.png' olarak kaydedildi ve ekrana çizildi!")

    # Model ağırlıklarını yapılan deneye özgü isimle kaydet
    save_name = f"multimodal_agrinet_{ACTIVATION}_{OPTIMIZER_TYPE}_sensor{USE_SENSORS}.pth"
    torch.save(model.state_dict(), save_name)
    print(f"Model başarıyla '{save_name}' olarak kaydedildi.")

if __name__ == "__main__":
    main()
