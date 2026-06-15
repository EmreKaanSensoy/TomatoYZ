import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from PIL import Image

class MultimodalAgriDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        # We use ImageFolder to automatically get class labels and image paths
        self.image_folder = ImageFolder(root=root_dir)
        self.transform = transform
        self.classes = self.image_folder.classes
        
        # Generate synthetic sensor data based on the label.
        # This simulates real-world conditions for different diseases.
        self.sensor_data = self._generate_synthetic_sensor_data()

    def _generate_synthetic_sensor_data(self):
        # Generates (temperature, humidity) arrays based on class index
        # Temperature in Celsius (approx 15-35), Humidity in % (approx 40-95)
        np.random.seed(42) # For reproducibility
        sensor_data = []
        for img_path, label in self.image_folder.samples:
            class_name = self.classes[label]
            
            # Simple heuristic rules for synthetic data (simulating sensor data based on disease type)
            if "Blight" in class_name:
                # Blights often like cooler, very moist conditions
                temp = np.random.normal(20, 2)
                hum = np.random.normal(85, 5)
            elif "Spider_mites" in class_name:
                # Spider mites like hot, dry conditions
                temp = np.random.normal(30, 2)
                hum = np.random.normal(45, 5)
            elif "Mold" in class_name or "Spot" in class_name:
                # Molds and spots like warm and humid
                temp = np.random.normal(26, 3)
                hum = np.random.normal(75, 5)
            else:
                # Healthy or viral: average normal conditions
                temp = np.random.normal(25, 3)
                hum = np.random.normal(60, 5)
                
            # Clip to realistic ranges
            temp = np.clip(temp, 10, 45)
            hum = np.clip(hum, 20, 100)
            
            # Normalize them for the neural network:
            # Let's say temp max 50, hum max 100
            norm_temp = temp / 50.0
            norm_hum = hum / 100.0
            
            sensor_data.append([norm_temp, norm_hum])
            
        return torch.tensor(sensor_data, dtype=torch.float32)

    def __len__(self):
        return len(self.image_folder)

    def __getitem__(self, idx):
        # 1. Get image and label
        img, label = self.image_folder[idx]
        
        # 2. Apply transforms to image
        if self.transform:
            img = self.transform(img)
            
        # 3. Get corresponding sensor data
        sensor_features = self.sensor_data[idx]
        
        return img, sensor_features, label
