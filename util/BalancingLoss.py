import torch
import torch.nn as nn
import torch.nn.functional as F
from .class_weight import class_weight

class BalancingLoss(nn.Module):
    def __init__(self):
        super(BalancingLoss,self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')
        
    def forward(self, logits, targets, text_keys):
        
        ce_losses = self.cross_entropy_loss(logits, targets)
        for i, (target, text_key) in enumerate(zip(targets,text_keys)):
            weight = class_weight[round(text_key.item(),3)][target]
            # ce_losses[i] = ce_losses[i]  * weight
            ce_losses[i] *= weight
            # print('hi')
        ce_loss = ce_losses.mean()
        return ce_loss