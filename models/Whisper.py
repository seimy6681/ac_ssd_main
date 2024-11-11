import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import WhisperModel, WhisperConfig
import math


class Whisper(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        
        assert "hop_length" in kwargs
        assert "data_length" in kwargs
        
        self.x_value = math.ceil(kwargs['data_length']*16000 // kwargs['hop_length'])
        self.embedding_table_length = math.floor((self.x_value - 1)/2) + 1
    
###########################################
# Whisper base model
###########################################
class Whisper_Base(Whisper):
    def __init__(self, pretrained=True, **kwargs):
        super().__init__(**kwargs)
        
        if pretrained:
            self.model =  WhisperModel.from_pretrained("openai/whisper-base", max_source_positions=self.embedding_table_length, return_dict=False, ignore_mismatched_sizes=True)
        else:
            config = WhisperConfig.from_pretrained("openai/whisper-base", max_source_positions=self.embedding_table_length, return_dict=False)
            self.model = WhisperModel(config)
            
class Base_Classifier(Whisper_Base):
    def __init__(self, num_classes: int = 2, pretrained=True, **kwargs):
        super().__init__(pretrained, **kwargs)
        
        self.num_classes = num_classes
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.Linear(256, self.num_classes),
        )
        
    def forward(self, x):
        x = self.model.encoder(x)
        x = torch.mean(x[0], dim=1)
        x = self.classifier(x)
        
        return x.squeeze(1)

class Base_Regressor(Whisper_Base):
    def __init__(self, pretrained=True, **kwargs):
        super().__init__(pretrained, **kwargs)
        
        self.regressor = nn.Sequential(
            nn.Linear(512, 256),
            nn.Linear(256, 1),
        )
        
    def forward(self, x):
        x = self.model.encoder(x)
        x = torch.mean(x[0], dim=1)
        x = self.regressor(x)
        
        return x.squeeze(1)
    
###########################################
# Whisper large model
###########################################
class Whisper_Large(Whisper):
    def __init__(self, pretrained=True, **kwargs):
        super().__init__(**kwargs)
        
        if pretrained:
            self.model =  WhisperModel.from_pretrained("openai/whisper-large-v3", max_source_positions=self.embedding_table_length, return_dict=False, ignore_mismatched_sizes=True)
        else:
            config = WhisperConfig.from_pretrained("openai/whisper-large-v3", max_source_positions=self.embedding_table_length, return_dict=False)
            self.model = WhisperModel(config)
            
class Large_Classifier(Whisper_Large):
    def __init__(self, num_classes: int = 2, pretrained=True, **kwargs):
        super().__init__(pretrained, **kwargs)
        
        self.num_classes = num_classes
        
        self.classifier = nn.Sequential(
            nn.Linear(1280, 256),
            nn.Linear(256, self.num_classes),
        )
        
    def forward(self, x):
        x = self.model.encoder(x)
        x = torch.mean(x[0], dim=1)
        x = self.classifier(x)
        
        return x.squeeze(1)

class Large_Regressor(Whisper_Large):
    def __init__(self, pretrained=True, **kwargs):
        super().__init__(pretrained, **kwargs)
        
        self.regressor = nn.Sequential(
            nn.Linear(1280, 256),
            nn.Linear(256, 1),
        )
        
    def forward(self, x):
        x = self.model.encoder(x)
        x = torch.mean(x[0], dim=1)
        x = self.regressor(x)
        
        return x.squeeze(1)
    
###########################################
# distil Whisper large model
###########################################
class Distil_Whisper_Large(Whisper):
    def __init__(self, pretrained=True, **kwargs):
        super().__init__(**kwargs)
        
        if pretrained:
            self.model =  WhisperModel.from_pretrained("distil-whisper/distil-large-v2", max_source_positions=self.embedding_table_length, return_dict=False, ignore_mismatched_sizes=True)
        else:
            config = WhisperConfig.from_pretrained("distil-whisper/distil-large-v2", max_source_positions=self.embedding_table_length, return_dict=False)
            self.model = WhisperModel(config)
            
class Distil_Large_Classifier(Distil_Whisper_Large):
    def __init__(self, num_classes: int = 2, pretrained=True, **kwargs):
        super().__init__(pretrained, **kwargs)
        
        self.num_classes = num_classes
        
        self.classifier = nn.Sequential(
            nn.Linear(1280, 256),
            nn.Linear(256, self.num_classes),
        )
        
    def forward(self, x):
        x = self.model.encoder(x)
        x = torch.mean(x[0], dim=1)
        x = self.classifier(x)
        
        return x.squeeze(1)

class Distil_Large_Regressor(Distil_Whisper_Large):
    def __init__(self, pretrained=True, **kwargs):
        super().__init__(pretrained, **kwargs)
        
        self.regressor = nn.Sequential(
            nn.Linear(1280, 256),
            nn.Linear(256, 1),
        )
        
    def forward(self, x):
        x = self.model.encoder(x)
        x = torch.mean(x[0], dim=1)
        x = self.regressor(x)
        
        return x.squeeze(1)
    
###########################################
# distil Whisper medium model
###########################################
class Distil_Whisper_Medium(Whisper):
    def __init__(self, pretrained=True, **kwargs):
        super().__init__(**kwargs)
        
        if pretrained:
            self.model =  WhisperModel.from_pretrained("distil-whisper/distil-medium.en", max_source_positions=self.embedding_table_length, return_dict=False, ignore_mismatched_sizes=True)
        else:
            config = WhisperConfig.from_pretrained("distil-whisper/distil-medium.en", max_source_positions=self.embedding_table_length, return_dict=False)
            self.model = WhisperModel(config)
            
class Distil_Medium_Classifier(Distil_Whisper_Medium):
    def __init__(self, num_classes: int = 2, pretrained=True, **kwargs):
        super().__init__(pretrained, **kwargs)
        
        self.num_classes = num_classes
        self.projector = nn.Linear(1024, 256)
        self.classifier = nn.Linear(256, self.num_classes)
        
        for i in range(kwargs["num_freeze"]):
            self.model.encoder.layers[i].requires_grad_(False)
        
    def forward(self, x):
        x = self.model.encoder(x)
        x = self.projector(x[0])
        x = torch.mean(x, dim=1)
        x = self.classifier(x)
        
        return x.squeeze(1)

#세임 embeding도 함께 리턴하는 모델 생성(contrastive loss계산할때 이용) (5.22)-------------------
class Distil_Medium_Classifier_with_text_embeddings(Distil_Whisper_Medium):
    def __init__(self, num_classes: int = 2, pretrained=True, **kwargs):
        print(kwargs)
        super().__init__(pretrained, **kwargs)
        
        embedding_dim = 512
        hidden_dim = 256
        
        self.num_classes = num_classes
        self.projector = nn.Linear(1024, embedding_dim)
        
        # Add additional layers
        # self.additional_layers = nn.Sequential(
        #     nn.Linear(embedding_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(hidden_dim, embedding_dim),
        #     nn.ReLU()
        # )
        
        self.classifier = nn.Linear(embedding_dim, self.num_classes)
        
        for i in range(kwargs["num_freeze"]):
            self.model.encoder.layers[i].requires_grad_(False)
        
    def forward(self, x):
        x = self.model.encoder(x)
        embeddings = self.projector(x[0])
        pooled_embeddings = torch.mean(embeddings, dim=1)
        logits = self.classifier(pooled_embeddings)
        
        return logits.squeeze(1), pooled_embeddings
        
        # logits = self.classifier(embeddings)
        
        # return logits.squeeze(1), embeddings
    

#### Multi Head Model (5)
# 세임    
class MultiHead_Distil_Medium_Classifier(Distil_Whisper_Medium):
    def __init__(self, num_classes: int = 2, pretrained=True, **kwargs):
        super().__init__(pretrained, **kwargs)
        
        self.num_classes = num_classes
        self.projector = nn.Linear(1024, 256)
        self.classifier = nn.Linear(256, self.num_classes) 
        
        #9개의 classifier 생성
        self.classifiers = nn.ModuleList([nn.Linear(256, self.num_classes) for _ in range(9)])
        
        # encoder layer freezing as specified
        for i in range(kwargs["num_freeze"]):
            self.model.encoder.layers[i].requires_grad_(False)
        
    def forward(self, x):
        age = x[:, -1, 0].int() - 2 # 나이만 떼오기 (batchsize, 맨마지막, 맨첫번째)
        x = x[:, :-1, :] #땐다

        x = self.model.encoder(x) # dimension : (batch_size, 1500, 1024)
        x = self.projector(x[0]) # (batch_size, 1500, 256)
        x = torch.mean(x, dim=1)  # (b, 256)
      
        # x = self.classifier(x) # (b, 2)
        
        #batch가 2개씩 들어가므로 돌면서 각각의 classifier에 넣어준다.
        logits = []
        for index in range(len(x)): #len(x) = 2
            batch = x[index]
             # 해당하는 나이의 classifier를 가져온다   
            logit = self.classifiers[age[index].item()-2](batch) #2-10세이니 2씩 빼준다.  (b,2)
            logit = logit.unsqueeze(0) # 바깥으로 [] 싸주기
            logits.append(logit)
        
        logit_concat = torch.cat(logits, dim=0) # batch 2개 세로로 붙여주기
            
        return logit_concat.squeeze(1)

class Distil_Medium_Regressor(Distil_Whisper_Medium):
    def __init__(self, pretrained=True, **kwargs):
        super().__init__(pretrained, **kwargs)
        
        self.regressor = nn.Sequential(
            nn.Linear(1024, 256),
            nn.Linear(256, 1),
        )
        
    def forward(self, x):
        x = self.model.encoder(x)
        x = torch.mean(x[0], dim=1)
        x = self.regressor(x)
        
        return x.squeeze(1)
    
class Age_Embedding_Whisper(Distil_Whisper_Medium):
    def __init__(self, num_classes: int = 2, pretrained=True, **kwargs):
        super().__init__(pretrained, **kwargs)
        
        assert "device" in kwargs
        
        self.num_classes = num_classes
        self.device = kwargs["device"]
        
        self.embedding = nn.Embedding(9, 1024)
        self.projector = nn.Linear(1024, 256)
        self.classifier = nn.Linear(256, self.num_classes)
        
        for i in range(kwargs["num_freeze"]):
            self.model.encoder.layers[i].requires_grad_(False)
        
    def forward(self, x): # x = spectrogrma
        age_vector = x[:, -1, :1500].int() - 2 # 나이만 떼오기 
        # age_vector = x[:, -1, :1500].int() - 2 # 나이만 떼오기 

        x = x[:, :-1, :]
        
        x = self.model.encoder(x)
        
        embedding_vector = self.embedding(age_vector)
        
        attention_score = torch.matmul(x[0], embedding_vector.transpose(-2, -1)) / math.sqrt(1024)
        attention_weights = F.softmax(attention_score, dim=-1)
        weighted_info = torch.matmul(attention_weights, embedding_vector)
        
        x = self.projector(weighted_info)
        x = torch.mean(x, dim=1)
        x = self.classifier(x)
        
        
        return x.squeeze(1)