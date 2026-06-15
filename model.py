import torch
import torch.nn as nn
import torchvision.models as models

class MultimodalAgriNet(nn.Module):
    def __init__(self, num_classes=10, use_sensor=True, activation='relu'):
        super(MultimodalAgriNet, self).__init__()
        self.use_sensor = use_sensor
        self.activation_type = activation.lower()
        
        # 1. Image Branch (ResNet18 backbone)
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Remove the final classification layer to get the 512-dim feature vector
        self.image_feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
        # Select activation function
        if self.activation_type == 'gelu':
            act_layer = nn.GELU()
        elif self.activation_type == 'leakyrelu':
            act_layer = nn.LeakyReLU()
        else:
            act_layer = nn.ReLU()
            
        if self.use_sensor:
            # 2. Sensor Data Branch (Temperature, Humidity -> 2 inputs)
            self.sensor_mlp = nn.Sequential(
                nn.Linear(2, 16),
                act_layer,
                nn.Linear(16, 32),
                act_layer
            )
            # 3. Fusion Layer (512 from ResNet + 32 from MLP = 544 dimensions)
            self.fusion_classifier = nn.Sequential(
                nn.Linear(512 + 32, 256),
                act_layer,
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )
        else:
            # If we don't use sensor data (Ablation Study), classify directly from 512-dim image features
            self.image_classifier = nn.Sequential(
                nn.Linear(512, 256),
                act_layer,
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )

    def forward(self, image, sensor_data):
        # Extract image features
        img_features = self.image_feature_extractor(image)
        img_features = torch.flatten(img_features, 1) # Flatten from (Batch, 512, 1, 1) to (Batch, 512)
        
        if self.use_sensor:
            # Extract sensor features
            sensor_features = self.sensor_mlp(sensor_data)
            # Concatenate features (Early Fusion approach)
            combined_features = torch.cat((img_features, sensor_features), dim=1)
            # Final Classification
            output = self.fusion_classifier(combined_features)
        else:
            # Ablation study: classify purely based on image
            output = self.image_classifier(img_features)
            
        return output
