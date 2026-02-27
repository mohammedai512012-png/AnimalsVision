from torch import nn
from torchvision import models

class AnimalsVisionModelV0(nn.Module):
    def __init__(self, output_shape):
        super().__init__()
        
        self.backbone = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
        
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, output_shape)
        )

    def forward(self, x):
        return self.backbone(x)
