import torch
import torch.nn as nn
# import torch.nn.functional as F
import torchaudio
from transformers import Wav2Vec2Model, Wav2Vec2Config


class Wav2Vec2_Base(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        
        if pretrained:
            self.model =  Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base", return_dict=False)
        else:
            config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-base", return_dict=False)
            self.model = Wav2Vec2Model(config)
        
    
class Base_Classifier(Wav2Vec2_Base):
    def __init__(self, num_classes: int = 2, pretrained=True):
        super().__init__(pretrained)
        
        self.num_classes = num_classes
        
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.Linear(256, self.num_classes),
        )
        
    def forward(self, x):
        x, _ = self.model(x)
        x = torch.mean(x, dim=1)
        x = self.classifier(x)
        
        return x.squeeze(1)

class Base_Regressor(Wav2Vec2_Base):
    def __init__(self, pretrained=True):
        super().__init__(pretrained)
        
        self.regressor = nn.Sequential(
            nn.Linear(768, 256),
            nn.Linear(256, 1),
        )
        
    def forward(self, x):
        x, _ = self.model(x)
        x = torch.mean(x, dim=1)
        x = self.regressor(x)
        
        return x.squeeze(1)