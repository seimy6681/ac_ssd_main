import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class Resnet50(nn.Module):
    def __init__(self, 
                 pretrained: bool = True,
                 num_classes: int = 2) -> None:
        super().__init__()
        self.num_classes = num_classes
        
        if pretrained:
            self.base_model = resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            self.base_model = resnet50()
        
        del self.base_model.fc
            
        self.classifier = nn.Linear(512 * 4, self.num_classes)
        self.model = nn.Sequential(self.base_model, self.classifier)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = self.base_model(x)
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)

        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)

        x = self.base_model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x