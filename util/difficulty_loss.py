
import torch
import torch.nn as nn

import hangul_jamo
import difficulty_factors
   
    
class DifficultyWeightedCrossEntropyLoss(nn.Module):
    def __init__(self, word_difficulties):
        super().__init__()
        self.word_difficulties = word_difficulties
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, logits, embeddings, targets, model, loader, text_keys, device):
        
        ce_loss = self.cross_entropy_loss(logits, targets)
        
        difficulty_factors = []
        for i, (embedding, text_key, target) in enumerate(zip(embeddings, text_keys, targets)):
            text_key = text_key.item()
            target = target.item()
            word = words[i]
            factor = get_difficulty_factor(word, target, self.word_difficulties)
            difficulty_factors.append(factor)
            
        difficulty_factors = torch.tensor(difficulty_factors, dtype=torch.float32)
        weighted_loss = ce_loss * difficulty_factors
        return torch.mean(weighted_loss)
    
    