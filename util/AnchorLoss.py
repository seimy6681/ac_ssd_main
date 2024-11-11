import torch
import torch.nn as nn
import torch.nn.functional as F

class AnchorLoss(nn.Module):
    def __init__(self, gamma=0.8, delta=0.00):
        super(AnchorLoss, self).__init__()
        self.gamma = gamma
        self.delta = delta
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')  # Keep individual losses

    def forward(self, logits, targets):
        """
        Forward pass for the anchor loss function.
        :param logits: Predicted logits from the model. Shape: [batch_size, num_classes]
        :param targets: Ground truth labels for classification. Shape: [batch_size]
        """
        # Calculate cross-entropy losses for each sample
        ce_losses = self.cross_entropy_loss(logits, targets)
        
        # Calculate probabilities using softmax
        probabilities = F.softmax(logits, dim=1)

        # Gather the anchor probabilities (q*)
        anchor_probs = probabilities.gather(1, targets.view(-1, 1)).squeeze(1)
        #hyper parameter delta
        anchor_probs = torch.clamp(anchor_probs - self.delta, min=0, max=1)

        # Compute the prediction difficulty for each class
        prediction_difficulties = probabilities - anchor_probs.view(-1, 1)

        # Modulator term
        modulator = (1 + prediction_difficulties.abs() ** self.gamma)
        #modulator target
        modulator_target = modulator.gather(1,targets.view(-1,1)).squeeze(1)
        # Compute the anchor loss
        anchor_loss = ce_losses * modulator_target
        
        return anchor_loss.mean()


