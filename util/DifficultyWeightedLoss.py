import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import wandb

class difficultyWeightedLoss(nn.Module):
    
    def __init__(self):
        """
        Hybrid loss function combining cross-entropy loss and contrastive loss.
        :param margin: Margin for contrastive loss.
        :param alpha: Weight for the cross-entropy loss.
        :param beta: Weight for the contrastive loss.
        """
        super(difficultyWeightedLoss, self).__init__()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none') # 평균내지 않기
       
    #  forward method with dynamic computation 
    #  of audio samples to compare with for the contrastive loss -----------------------
    def forward(self, logits, targets, text_keys):
        """
        Forward pass for the hybrid loss function.
        :param logits: Predicted logits from the model.
        :param targets: Ground truth labels for classification.
        :param embeddings: Embeddings from the model.
        :param loader: train dataset to randomly pull 2 embeddings with different classes
        :param target_texts: List of target texts for the batch.
        """
        ce_losses = self.cross_entropy_loss(logits, targets)
        # print(f'{ce_losses}')
        
        for i, (target, text_key) in enumerate(zip(targets,text_keys)):
            text_key = text_key.item()
            # difficulty_factor = self.difficulty_factors.get(text_key, 1.0)  # Default to 1.0 if text_key is not found
            if target == 1: #if disordered, sample run for 포도
                difficulty_factor = 1
            if target == 0:
                difficulty_factor = 1.4
                
            ce_losses[i] *= difficulty_factor

        ce_loss = ce_losses.mean()
        
        return ce_loss