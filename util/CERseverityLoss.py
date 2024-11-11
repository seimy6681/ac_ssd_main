
import torch
import torch.nn as nn
import hangul_jamo
import pandas as pd
from .speaker_cer import speaker_cer

# linear_weight()
    
class CERseverityLoss(nn.Module):
    def __init__(self):
        super(CERseverityLoss,self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')
        
    def forward(self, logits, targets, ids):
        
        ce_losses = self.cross_entropy_loss(logits, targets)
        for i, (target, id) in enumerate(zip(targets,ids)):
            
            alpha = 2
            id = round(id.item(),3)
            # print(f'{id=} from forward')
            cer = speaker_cer[id]
            # print(f'{cer=}')

            weight = 1 + alpha * cer
            # ce_losses[i] = ce_losses[i]  * weight
            ce_losses[i] *= weight
            # print('hi')
        ce_loss = ce_losses.mean()
        return ce_loss