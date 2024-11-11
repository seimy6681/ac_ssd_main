import torch
import torch.nn as nn
from torch.utils.data.dataset import ConcatDataset
import torchaudio
import torchaudio.transforms as T
import librosa
from torch.utils.data import Dataset
import pandas as pd
import os
from typing import Any, Union
import re
import math
import json
import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from target_text_list_norm import text_dict


########################################################################################
# 
# Custom Dataset module for pytorch mode
# Input: 
#   - config: config.py module
#   - file: csv file or pandas dataframe object to create dataset 
#           ! must have "audio", config.target in header
#   - data_type: return data type ('wave' or 'spectrogram')
#   - processor: huggingface feature extractor to preprocess waveform
#   - transform: applying SpecAugment 
#                ! data_type should be 'spectrogram'
#
########################################################################################

class CustomDataset(Dataset):
    def __init__(self, 
                 config:Any, 
                 file:Union[str, pd.DataFrame], 
                 data_type:str, 
                 processor:Any=None, 
                 transform:nn.Module=None) -> None:
        super().__init__()
        # configuration arguments
        self.config = config
        self.data_type = data_type
        self.processor = processor
        self.sampling_rate = 16000
        
        self.y_value_map = {
            "resnet50": 224,
            "whisper_base": 80,
            "whisper_large": 128,
            "distil_whisper_large": 80,
            "distil_whisper_medium": 80,
            "age_embedding_whisper": 80,
            "multihead_distil_medium_whisper_classifier": 80
        }
        
        # smoothed label
        self.smoothed_label = {
            2: [[0.5, 0.5], [0.02, 0.98]],
            3: [[0.757, 1-0.757], [0.02, 0.98]],
            4: [[0.864, 1-0.864], [0.02, 0.98]],
            5: [[0.9215, 1-0.9215], [0.02, 0.98]],
            6: [[0.95, 0.05], [0.02, 0.98]],
            7: [[0.98, 0.02], [0.02, 0.98]],
            8: [[0.98, 0.02], [0.02, 0.98]],
            9: [[0.98, 0.02], [0.02, 0.98]],
            10: [[0.98, 0.02], [0.02, 0.98]],
        }
        
        # (7.1) 타겟 단어별 label smoothing (by text_key)
        self.by_text_smoothed_label = {
            -1 : [[1.0 -0.3, 0.3], [0.0, 1-0.0]], #포도
            -0.873 : [[1.0 - 0.3, 0.3], [0.1, 1-0.1]], #컵
            -0.646 : [[1.0 - 0.0, 0.0], [0.3, 1-0.3]], #빗
            #------------------------------------------------
            -0.797 : [[1.0-0.05, 0.5], [0.3, 1-0.3]], #색종이
            -0.899 : [[1.0-0.3, 0.3], [0.0, 1-0.0]], #옥수수
            -0.924 : [[1.0-0.3, 0.3], [0.0, 1-0.0]], #햄버거
        }
        
        self.spec = T.MelSpectrogram(
            n_fft=self.config.n_fft,
            n_mels=self.config.n_mels,
            win_length=self.config.win_length,
            hop_length=215 if self.config.model_name.lower() == 'resnet50' else self.config.hop_length,
            pad=self.config.pad,
            power=self.config.power,
            normalized=self.config.normalized,
            center=self.config.center,
            onesided=self.config.onesided
        )
        
        # .csv file (header: ["audio", config.target, ...])
        if isinstance(file, str):
            self.file = file
            self.df = pd.read_csv(os.path.join(self.config.data_path, self.file))
        else:
            self.df = file
        
        # SpecAugment transform (only used if data_type=='spectrogram')
        self.transform = transform
        
        # target label <-> index(id)
        self.label2id = {}
        self.id2label = {}
        
        # self.age_list = ["2_전", "2_후", "3_전", "3_후", "4_전", "4_후", "5_전", "5_후", "6_전", "6_후", "7", "8", "9", "10"]
        self.age_list = [2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.ts_list = [0, 1]
        
        self.age_id_list = [i for i in range(len(self.age_list))]
        self.normalized_age_list = self.age_id_list
        self._normalize(self.normalized_age_list)
        
        if self.config.target == "age":
            self.label_list = self.age_list
        else: # target == td/ssd
            self.label_list = self.ts_list
        self.id_list = [i for i in range(len(self.label_list))]
        if self.config.task == "regression":
            self._normalize(self.id_list)
        
        for i, target in enumerate(self.label_list):
            self.label2id[target] = i
        for k, v in self.label2id.items():
            self.id2label[v] = k
    
    def _normalize(self, id_list) -> None:
        min_i = min(id_list)
        max_i = max(id_list)
        
        for i, idx in enumerate(id_list):
            id_list[i] = (idx - min_i) / (max_i - min_i)
    
    def get_label_list(self) -> list:
        return self.label_list
            
    def _get_label(self, id:int) -> Any:
        return self.id2label[id]
    
    def _get_id(self, label:Any) -> int:
        return self.label2id[label]
    
    def _log_mel_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        
        audio = self.spec(audio)
        if audio.shape[-1] % 10 != 0:
            audio = audio[:, :, :audio.shape[-1] - audio.shape[-1]%10]
        
        audio = torch.clamp(audio, min=1e-10).log10()
        audio = (torch.maximum(audio, audio.max() - 8.0) + 4.0) / 4.0
        
        return audio
    
    def __add__(self, other: Dataset) -> ConcatDataset:
        df = pd.concat([self.df, other.df], ignore_index=True)
        
        return CustomDataset(self.config, df, self.data_type, self.processor, self.transform)
        
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, index:int) -> Any:
        # load waveform
        length = self.sampling_rate * self.config.data_length
        audio_path = self.df.iloc[index]["speech_file"]
        audio, sr = torchaudio.load(audio_path)
        audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=self.sampling_rate)
        # audio = librosa.load(audio_path, sr=self.sampling_rate)
        audio = torch.mean(audio, dim=0)[:length].unsqueeze(0) # shape (1, length)
        
        if audio.shape[1] < length:
            audio = torch.nn.functional.pad(audio, (0, length - audio.shape[1]), mode='constant', value=0.0)
        
        # label
        label = self.df.iloc[index][self.config.target]
        if self.config.task == "regression":
            idx = torch.tensor(self._get_id(label), dtype=torch.float64) #
        else:
            idx = torch.tensor(self._get_id(label), dtype=torch.int64) # float...!!!
        # TODO: multi label 가중치를 어떻게 하지??
        
        # if self.config.multi_label:
        #     age = int(self.df.iloc[index]["age"])
        #     # one_hot = nn.functional.one_hot(idx, num_classes=len(self.label_list))
        #     # idx = one_hot * (1 - self.config.age_lambda * (1/age)) + (1 - one_hot) * self.config.age_lambda * (1/age) # 2 dimensional label
        #     idx = torch.tensor(self.smoothed_label[age][int(label)]).float()
            
        # preprocessing (if huggingface model)
        if self.processor:
            try: # wav2vec2
                audio = self.processor(
                    audio.squeeze(0),
                    sampling_rate=self.processor.sampling_rate,
                    padding='max_length',
                    max_length=self.processor.sampling_rate*int(self.config.data_length),
                    truncation=True,
                    return_tensors='pt'
                ).input_values
            except: # whisper
                audio = self.processor(
                    audio.squeeze(0),
                    sampling_rate=self.processor.sampling_rate,
                    padding='max_length',
                    max_length=self.processor.sampling_rate*int(self.config.data_length),
                    truncation=True,
                    return_tensors='pt',
                ).input_features
        
        if self.data_type=='wave':
            return audio.squeeze(0), idx
        
        elif self.data_type=='spectrogram':
            
            if not self.processor:
                if self.config.combine_strategy == "before":
                    audio = self.spec(audio)
                    if audio.shape[-1] % 10 != 0:
                        audio = audio[:, :, :audio.shape[-1] - audio.shape[-1]%10]
                else:
                    audio = self._log_mel_spectrogram(audio)
                
            if self.config.scaling:
                audio = (audio - audio.min()) / (audio.max() - audio.min())
                
            x_value = 224 if self.config.model_name.lower() == 'resnet50' else audio.shape[-1]
            for k, v in self.y_value_map.items():
                if k in self.config.model_name.lower():
                    y_value = v
            # if self.config.model_embedding:
            #     y_value += 1
            
            # if self.config.n_mels != y_value:
            #         # if not self.config.model_embedding:
            #         normalized_age_value = self.normalized_age_list[self.age_list.index(self.df.iloc[index]["age"])] * self.config.age_momentum
            #             # padding_tensor = (torch.clamp(torch.full((1, y_value-self.config.n_mels, x_value), age_value), min=1e-10).log10() + 4.0) / 4.0
            #             # padding_tensor = torch.full((1, y_value-self.config.n_mels, x_value), age_value)
            #         # else:
            #         padding_tensor = torch.full((1, y_value-self.config.n_mels, x_value), normalized_age_value)
            #         audio = torch.cat((audio, padding_tensor), dim=1)

            if not self.processor and self.config.combine_strategy == "before":
                audio = torch.clamp(audio, min=1e-10).log10()
                audio = (torch.maximum(audio, audio.max() - 8.0) + 4.0) / 4.0

            if self.transform:
                audio = self.transform(audio)
            
            #  age embedding 끝에 붙이기 -----------------------------
            if self.config.model_embedding:
            
                age_value = int(self.df.iloc[index]["age"]) # csv에서 age 떼오기
                age_tensor = torch.full((1, 1, x_value), age_value) # spectrogram 세로 길이많금 나이로 채워
                audio = torch.cat((audio, age_tensor), dim=1) # 붙여 맨마지막 한줄 (-1)

            #target text input으로 나이벡터 밑에 붙이기--------------------------------------------------------
            if self.config.model_text_embedding:
                target_text = self.df.iloc[index]["target_text"]
                    
                text_key = text_dict.get(target_text, None)
                if text_key==None:
                    print(target_text)
                    
                #normalizing:
                # text_key_norm = (2 * ((text_key - 0) / (79 - 0))) - 1
                # text_key_norm = round(text_key_norm,3)
                # text_tensor = torch.full((1,1,x_value),  text_key_norm)
                
                text_tensor = torch.full((1,1,x_value),  text_key)
                
                audio = torch.cat((audio,text_tensor), dim=1)

            # Speaker ID input으로 나이벡터 밑에 붙이기--------------------------------------------------------
            if self.config.model_id_embedding:
                id = self.df.iloc[index]["id"]
                #normalizing:
                id = round ((id-1) / (1098), 3)
                # print(f'{id=} from dataset')
                id_tensor = torch.full((1,1,x_value), id)
                
                audio = torch.cat((audio,id_tensor), dim=1)
                                
            if self.config.multi_label:
                if self.config.label_smoothing_by_text:
                    idx = torch.tensor(self.by_text_smoothed_label[round(text_key,3)][int(label)]).float() # dict[text key][label(0 / 1)]
                    # print('hi')
                
                # age_value = int(self.df.iloc[index]["age"]) # csv에서 age 떼오기
                # age_tensor = torch.full((1, 1, x_value), age_value) # spectrogram 세로 길이많금 나이로 채워
                # audio = torch.cat((audio, age_tensor), dim=1) # 붙여 맨마지막 한줄 (-1)
            #------------------------------------------------------------------------------------------------

            if self.config.n_channels != 1:
                audio = audio.repeat(3, 1, 1)
                
            if "whisper" in self.config.model_name.lower():
                audio = audio.squeeze(0)
        # print(audio.shape)  
  
        return audio, idx
    
    def pad_sequence(self, batch):
        # Make all tensor in a batch the same length by padding with zeros
        batch = [item.t() for item in batch]
        batch = nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
        
        return batch.permute(0, 2, 1).squeeze(1)
    
    def collate_fn(self, batch):

        # A data tuple has the form:
        # waveform, sample_rate, label, speaker_id, utterance_number

        tensors, targets = [], []

        # Gather in lists, and encode labels as indices
        for waveform, label in batch:
            tensors += [waveform]
            targets += [label]
        # Group the list of tensors into a batched tensor
        tensors = self.pad_sequence(tensors)
        targets = torch.stack(targets)
        # print(tensors.shape)
        
        return tensors, targets