import os
import sys
import torch
from models import Wav2Vec2, Resnet, Whisper
from .Class_balanced_loss_pytorch.class_balanced_loss import focal_loss, CB_loss
from optimizer import sam
# from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from torch.optim.lr_scheduler import CosineAnnealingLR

from torch.utils.data import DataLoader
from typing import Any, Dict
from argparse import Namespace, ArgumentParser
from tqdm import tqdm
# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from .Contrastive_loss import HybridLoss
from .DifficultyWeightedLoss import difficultyWeightedLoss
from .AnchorLoss import AnchorLoss
from .BalancingLoss import BalancingLoss
from .CERseverityLoss import CERseverityLoss
import numpy as np
# from .target_text_list import text_dict

class Loader:
    @staticmethod
    def load_model(key: str, **kwargs) -> torch.nn.Module:
        key = key.lower()
            
        if key == "wav2vec2_base_classifier":
            model = Wav2Vec2.Base_Classifier(**kwargs)
            
        elif key == "wav2vec2_base_regressor":
            model = Wav2Vec2.Base_Regressor(**kwargs)
            
        elif key == "resnet50":
            model = Resnet.Resnet50(**kwargs)
        
        elif key == "whisper_base_classifier":
            model = Whisper.Base_Classifier(**kwargs)
            
        elif key == "whisper_base_regressor":
            model = Whisper.Base_Regressor(**kwargs)
            
        elif key == "whisper_large_classifier":
            model = Whisper.Large_Classifier(**kwargs)
            
        elif key == "whisper_large_regressor":
            model = Whisper.Large_Regressor(**kwargs)
            
        elif key == "distil_whisper_large_classifier":
            model = Whisper.Distil_Large_Classifier(**kwargs)
            
        elif key == "distil_whisper_large_regressor":
            model = Whisper.Distil_Large_Regressor(**kwargs)
            
        elif key == "distil_whisper_medium_classifier":
            model = Whisper.Distil_Medium_Classifier(**kwargs)
            
        elif key == "distil_whisper_medium_regressor":
            model = Whisper.Distil_Medium_Regressor(**kwargs)
            
        elif key == "age_embedding_whisper":
            model = Whisper.Age_Embedding_Whisper(**kwargs)
        
        elif key == "multihead_distil_medium_whisper_classifier":
            model = Whisper.MultiHead_Distil_Medium_Classifier(**kwargs)
        elif key == "distil_whisper_medium_classifier_with_text_embeddings":
            model  = Whisper.Distil_Medium_Classifier_with_text_embeddings(**kwargs)
        else:
            print(key)
            raise ValueError("wrong model name")
        
        return model
    
    @staticmethod
    def load_criterion(key: str, **kwargs) -> Any:
        key = key.lower()
        
        if key == "mse":
            criterion = torch.nn.MSELoss(**kwargs)
            
        elif key == "ce":
            criterion = torch.nn.CrossEntropyLoss(**kwargs)
            
        elif key == "bce":
            criterion = torch.nn.BCELoss(**kwargs)
            
        elif key == "bce_logit":
            criterion = torch.nn.BCEWithLogitsLoss(**kwargs)
            
        elif key == "focal":
            alpha = kwargs["alpha"] if "alpha" in kwargs else 0.25
            gamma = kwargs["gamma_focal"] if "gamma_focal" in kwargs else 2.0
            
            criterion = lambda outputs, labels: focal_loss(labels, outputs, alpha=alpha, gamma=gamma)
            
        elif key == "cb":
            loss_type = kwargs["loss_type"] if "loss_type" in kwargs else "focal"
            beta = kwargs["beta"] if "beta" in kwargs else 0.999
            gamma = kwargs["gamma_cb"] if "gamma_cb" in kwargs else 1.0
            
            criterion = lambda outputs, labels: CB_loss(labels, outputs,
                                                        samples_per_cls=kwargs["samples_per_cls"],
                                                        no_of_classes=kwargs["no_of_classes"],
                                                        loss_type=loss_type,
                                                        beta=beta,
                                                        gamma=gamma)
        elif key == 'hybrid':
            criterion = HybridLoss(**kwargs)
        
        elif key == 'difficulty':
            criterion = difficultyWeightedLoss(**kwargs)
        
        elif key == 'anchor':
            criterion = AnchorLoss(**kwargs)
        
        elif key == 'balancing':
            criterion = BalancingLoss(**kwargs)
        
        elif key == 'cer':
            criterion = CERseverityLoss(**kwargs)

        return criterion

    @staticmethod
    def load_optimizer(key: str, parameters: Any, **kwargs) -> torch.optim.Optimizer:
        key = key.lower()
        
        if key == "adamw":
            optimizer = torch.optim.AdamW(parameters, **kwargs)

        elif key == "adam":
            optimizer = torch.optim.Adam(parameters, **kwargs)
            
        elif key == "sam":
            optimizer = sam.SAM(parameters, **kwargs)
        
        elif key == "sgd":
            optimizer = torch.optim.SGD(parameters, **kwargs)
            
        return optimizer
    
    @staticmethod
    def load_scheduler(key: str, optimizer: torch.optim.Optimizer, **kwargs) -> torch.optim.lr_scheduler:
        key = key.lower()
        kwargs["max_lr"] = optimizer.param_groups[0]['lr']
        
        if key == "linear":
            kwargs['pct_start'] = kwargs.pop('warmup_ratio')
            kwargs['anneal_strategy'] = "linear"
            kwargs['cycle_momentum'] = False
            
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, **kwargs)
            
        elif key == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
        
        return scheduler

#세임 compute_centroids for contrsastive loss (5/22) ---------------------------------------------------------
    @staticmethod
    def load_precomputed_embeddings(model, train_loader: DataLoader, device: torch.device, save_path: str,**model_kwargs):
        
        # embeddings_file = os.path.join(save_path, "embeddings.pth")
        # centroids_file = os.path.join(save_path, "centroids_dict.pth")

        model.eval()
        embeddings_dict = {}

        with torch.no_grad():
            for inputs, labels in tqdm(train_loader, desc="Precomputing Embeddings"):
                inputs = inputs.to(device)
                text_keys = inputs[:,-1,0]
                logits, embeddings = model(inputs)
                
                for embedding, text_key, label in zip(embeddings, text_keys, labels):
                    text_key = text_key.item()
                    label = label.item()

                    if text_key not in embeddings_dict:
                        embeddings_dict[text_key] = {0: [], 1: []}
                    embeddings_dict[text_key][label].append(embedding.to(device))

        centroids_dict = {}
        for text_key, label_dict in embeddings_dict.items():
            centroids_dict[text_key] = {}
            for label, embeddings in label_dict.items():
                if not embeddings:
                    continue
                centroids_dict[text_key][label] = torch.stack(embeddings).mean(dim=0)
        
        # Convert tensors to numpy arrays
        for text_key, label_dict in centroids_dict.items():
            for label, centroid in label_dict.items():
                centroids_dict[text_key][label] = centroid.cpu().numpy()

        # Save the dictionary as an .npy file
        np.save(save_path, centroids_dict)
        return centroids_dict
    
    @staticmethod

    def load_centroids_dict(device, file_path: str):
        # Load the .npy file
        centroids_dict = np.load(file_path, allow_pickle=True).item()

        # Convert numpy arrays back to tensors
        for text_key, label_dict in centroids_dict.items():
            for label, centroid in label_dict.items():
                centroids_dict[text_key][label] = torch.tensor(centroid).to(device)

        return centroids_dict

    


# --------------------------------------------------------------------------------------

class Parser(object):
    def __call__(self, parser: ArgumentParser, args: Namespace) -> Dict[str, Dict[str, Any]]:
        config = dict()
        for group in parser._action_groups:
            group_dict={a.dest: getattr(args, a.dest, None) for a in group._group_actions}
            for k, v in group_dict.copy().items():
                if v == None:
                    group_dict.pop(k, None)
            if len(group_dict) > 0:
                config[group.title] = group_dict
                
        if "whisper" in args.model_name.lower():
            config['model']['hop_length'] = args.hop_length
            config['model']['data_length'] = args.data_length
            config['dataset']['data_type'] = 'spectrogram'
            args.n_channels = 1
            
            if ("base" in args.model_name.lower() or 
                "distil" in args.model_name.lower() or
                "embedding" in args.model_name.lower()) and args.n_mels > 80:
                args.n_mels = 80
                
            elif "large" in args.model_name.lower() and args.n_mels > 128:
                args.n_mels = 128

        if 'wav2vec2' not in args.model_name.lower():
            args.data_type = 'spectrogram'
            
        return config