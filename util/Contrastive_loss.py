import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import wandb

class ContrastiveLoss(nn.Module):
    
    def __init__(self,margin=4.0):
        super(ContrastiveLoss,self).__init__()
        self.margin = margin
        self.sigmoid = nn.Sigmoid()
        
#     def forward(self,output1, output2, label):
#         euclidean_distance = F.pairwise_distance(output1, output2)
#         loss_contrastive = torch.mean((label) * torch.pow(euclidean_distance,2) + (1-label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0),2))

#         # wandb.log({
#         #             "contrastive": {
#         #                 "pos_cont_loss/step": euclidean_distance.item(),
#         #                 "neg_cont_loss/step": euclidean_dinstance.item(),
        
#         #                 }
#         #         })
#         return loss_contrastive


    def forward(self, output1, output2, label):
        
        # print('hi')
        output1 = F.normalize(output1, p=2, dim=0) #normalzing with L2 norm over 1st dimension(0th)
        output2 = F.normalize(output2, p=2, dim=0)
        
        euclidean_distance = F.pairwise_distance(output1, output2)
        # normalizing  euclidean distance
        # option 1: sigmoid
        # euclidean_distance = self.sigmoid(euclidean_distance)
        #option 2: min max scaling
        # min_val = torch.min(euclidean_distance)
        # max_val = torch.max(euclidean_distance)
        # euclidean_distance = (euclidean_distance - min_val) / (max_val - min_val)

        euclidean_distance = torch.log(1 + euclidean_distance)
        
        # Separate positive and negative samples
        pos_mask = (label == 1)
        neg_mask = (label == 0)
        
        pos_distances = euclidean_distance[pos_mask]
        neg_distances = euclidean_distance[neg_mask]
        
        # Compute contrastive loss
        pos_loss = torch.mean(pos_mask.float() * torch.pow(euclidean_distance, 2))
        neg_loss = torch.mean(neg_mask.float() * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        loss_contrastive = pos_loss + neg_loss
        # loss_contrastive = pos_loss

        # # Log the distances to Wandb
        if pos_distances.numel() > 0:  # Check if there are positive samples
            wandb.log({"contrastive/euclidian_distance_for_positive_pair_step": pos_distances.mean().item()})
        if neg_distances.numel() > 0:  # Check if there are negative samples
            wandb.log({"contrastive/euclidian_distance_for_negative_pair_step": neg_distances.mean().item()})

        return loss_contrastive
    
class HybridLoss(nn.Module):
    
    def __init__(self, margin=4.0, alpha=0.8, beta=0.2):
        """
        Hybrid loss function combining cross-entropy loss and contrastive loss.
        :param margin: Margin for contrastive loss.
        :param alpha: Weight for the cross-entropy loss.
        :param beta: Weight for the contrastive loss.
        """
        super(HybridLoss, self).__init__()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.contrastive_loss = ContrastiveLoss(margin)
        self.alpha = alpha
        self.beta = beta
     
     
    #  forward method with dynamic computation 
    #  of audio samples to compare with for the contrastive loss -----------------------
    def forward(self, logits, embeddings, targets, model, loader, text_keys, device):
        """
        Forward pass for the hybrid loss function.
        :param logits: Predicted logits from the model.
        :param targets: Ground truth labels for classification.
        :param embeddings: Embeddings from the model.
        :param loader: train dataset to randomly pull 2 embeddings with different classes
        :param target_texts: List of target texts for the batch.
        """
        ce_loss = self.cross_entropy_loss(logits, targets)
        
        # randomly sampling audio with the same or different target ---------------------
        def random_audio_matching_text(target, text_key,same=True):
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                mask = (labels == target) if same else (labels != target)
                indices = mask.nonzero(as_tuple=True)[0]
                indices = [i for i in indices if inputs[i, -1, 0].item() == text_key]
                
                if len(indices) > 0:
                    idx = random.choice(indices).item() 
                    with torch.no_grad():
                        _, embedding = model(inputs[idx].unsqueeze(0))
                        return embedding.squeeze(0)
            return None
        
        cont_loss = 0
        pos_cont_losses = []
        neg_cont_losses = []
        
        for i, (embedding, text_key, target) in enumerate(zip(embeddings, text_keys, targets)):
            text_key = text_key.item()
            target = target.item()
            
            rand_audio_pos = random_audio_matching_text(target, text_key, same=True)
            rand_audio_neg = random_audio_matching_text(target, text_key,same=False)
            if rand_audio_pos is not None and rand_audio_neg is not None:
                pos_cont_loss = self.contrastive_loss(embedding, rand_audio_pos.unsqueeze(0), torch.tensor([1.0]).to(embedding.device))
                neg_cont_loss = self.contrastive_loss(embedding, rand_audio_neg.unsqueeze(0), torch.tensor([0.0]).to(embedding.device))

                # print(f'{target=},{pos_cont_loss=},{neg_cont_loss=}')
            
                cont_loss += pos_cont_loss
                cont_loss += neg_cont_loss
                
                pos_cont_losses.append(pos_cont_loss.item())
                neg_cont_losses.append(neg_cont_loss.item())
                
                wandb.log({
                    "contrastive": {
                        "pos_cont_loss/step": pos_cont_loss.item(),
                        "neg_cont_loss/step": neg_cont_loss.item(),
        
                        }
                })
                   
        cont_loss /= len(text_keys) # normalizing by the number of examples in batch
        wandb.log({ "cont_loss/batch": cont_loss.item()})       
        
        total_loss = self.alpha * ce_loss + self.beta * cont_loss
        return total_loss
        
    # @forward method: pre-computed-centroid approach ----------------------------------------------------
    
    
    # def forward(self, logits, embeddings, targets, centroids_dict, text_keys):
    #     """
    #     Forward pass for the hybrid loss function.
    #     :param logits: Predicted logits from the model.
    #     :param targets: Ground truth labels for classification.
    #     :param embeddings: Embeddings from the model.
    #     :param centroids_dict: Dictionary of precomputed centroids.
    #     :param target_texts: List of target texts for the batch.
    #     """
    #     ce_loss = self.cross_entropy_loss(logits, targets)
        
    #     cont_loss = 0
    #     for i, (embedding, text_key, target) in enumerate(zip(embeddings, text_keys, targets)):
    #         text_key = text_key.item()
    #         target = target.item()

    #         # pre-computed-centroid approach ----------------------------------------------------
    #         if text_key not in centroids_dict:
    #             # raise KeyError(f"text_key {text_key} not found in centroids_dict")
    #             print('skipped text_key', text_key)
    #             continue
    #         if target not in centroids_dict[text_key]:
    #             # raise KeyError(f"target {target} not found in centroids_dict[{text_key}]")
    #             print('skipped', centroids_dict[text_key],target)
    #             continue
    #         if (1 - target) not in centroids_dict[text_key]:
    #             # raise KeyError(f"1 - target {1 - target} not found in centroids_dict[{text_key}]")
    #             print('skipped',centroids_dict[text_key],1-target)
    #             continue
            
    #         centroid_pos = centroids_dict[text_key][target]
    #         centroid_neg = centroids_dict[text_key][1 - target]

    #         pos_cont_loss = self.contrastive_loss(embedding, centroid_pos.unsqueeze(0), torch.tensor([1.0]).to(embedding.device))
    #         neg_cont_loss = self.contrastive_loss(embedding, centroid_neg.unsqueeze(0), torch.tensor([0.0]).to(embedding.device))

    #         print(f'{target=},{pos_cont_loss=},{neg_cont_loss=}')
            
    #         cont_loss += pos_cont_loss
    #         cont_loss += neg_cont_loss

    #     cont_loss /= len(text_keys) # normalizing by the number of examples in batch
        
    #     total_loss = self.alpha * ce_loss + self.beta * cont_loss
    #     return total_loss
