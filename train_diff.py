# train file for difficultyWeightedLoss and BalancingLoss,
# assume text_embeddings at -1

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, classification_report
import pandas as pd
from tqdm import tqdm
import wandb
from typing import Any, Tuple, Union, List, Callable, Optional

from util.util import Loader
from target_text_list_norm import text_dict

def calc_uar(confusion_matrix: np.ndarray) -> float:
    """calculate UAR(Unweighted Avarage Recall)

    Args:
        confusion_matrix (np.ndarray): 2 * 2 ndarray
        form of [[tn, fp
                  fn, tp]]
        
    Returns:
        float: computed uar
    """
    cm = confusion_matrix
    
    tp, tn = cm[1, 1], cm[0, 0]
    fn, fp = cm[1, 0], cm[0, 1]
    
    sensitivity = tp / (tp + fn)
    specificity = tn / (fp + tn)
    
    uar = (specificity + sensitivity)/2.0
    
    return uar


def train_diff(model: nn.Module, 
          criterion: Any, 
          optimizer: optim.Optimizer, 
          scheduler: optim.lr_scheduler,
          train_loader: DataLoader,
          valid_loader: DataLoader, 
          callbacks: Union[Callable, List[Callable]],
          device: torch.device, 
          config: Any,
          ) -> None:

    wandb.watch(model, log_freq=config.logging_steps, log=config.watch)
    num_epochs = config.epochs
    best_acc = 0
    best_loss = 0


    epoch_pbar = tqdm(range(1,num_epochs+1), total=num_epochs, position=0, leave=True)
    for epoch in epoch_pbar:
        epoch_pbar.set_description(f"Training with Text tokens: Epoch {epoch}/{num_epochs}")
        # train loop
        model.train()
        train_loss = 0.0
        
        preds = []
        y_true = []
        
        step_pbar = tqdm(train_loader, total=len(train_loader), position=1, leave=False)
        for step, (inputs, labels) in enumerate(step_pbar):
            step_pbar.set_description(f"Step {step}/{len(train_loader)}")
            inputs, labels = inputs.to(device), labels.to(device)

            outputs, _ = model(inputs)
            text_keys = inputs[:,-1,0]
                # print(f'{text_keys=} text in evaluation')
           
            cur_lr = scheduler.get_last_lr()[0]

            loss = criterion(outputs, labels, text_keys)
            loss /= config.gradient_accumulation_steps
            loss.backward()
            train_loss += loss.item()
            
            if ((step+1) % config.gradient_accumulation_steps == 0) or (step + 1 == len(train_loader)):
                
                if "multihead" in config.model_name.lower():
                    #age 일단 가져오고  classifier와 인덱스 맞추기 위해 2 뺀다.
                    age = inputs[:, -1, 0].int() - 2
                    #배치에 있는 나이 돌면서
                    for batch in age.tolist():
                        idx = batch -2
                        for i, classifier in enumerate(model.classifiers):
                            if i != idx: # 해당 나이 아닌 나이의 classifier 는 parameter update 하지 않기
                                for param in classifier.parameters():
                                    param.grad = None
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            
            if config.task == 'classification':
                
                if config.multi_label:
                    
                    pred = (torch.sigmoid(outputs.data) >= config.acc_thrs).float()
                    pred = outputs.argmax(dim=1, keepdim=True) 
                    
                    y_true.extend(labels.argmax(dim=1).tolist())
                    preds.extend(pred.squeeze(1).tolist())
                    
                else:
                    
                    pred = outputs.argmax(dim=1, keepdim=True)  # 가장 높은 확률을 가진 클래스를 예측값으로
                    # correct_ += pred.eq(labels.view_as(pred)).sum().item()  # 정답 수를 누적
                    preds.extend(pred.squeeze(1).tolist())
                    y_true.extend(labels.tolist())
                            
            else: # regression
                ss_res = torch.sum((labels - outputs.data) ** 2)
                ss_tot = torch.sum((labels - torch.mean(labels)) ** 2)
                acc = 1 - ss_res / ss_tot
            
            # train/step 정확도 계산 -------------------------------------
            train_matrix = confusion_matrix(y_true, preds, labels=[0,1])
            acc = train_matrix.trace() / train_matrix.sum()
            acc0 = train_matrix[0][0] / (train_matrix[0][0] + train_matrix[0][1])
            acc1 = train_matrix[1][1] / (train_matrix[1][0] + train_matrix[1][1])
            train_uar = calc_uar(train_matrix)   
            # train/step 정확도 계산 -------------------------------------
       
            if (step % config.logging_steps == 0) or (step == len(train_loader)-1):
                wandb.log({
                    "epoch": epoch,
                    "global_step": epoch*(len(train_loader) + len(valid_loader)) + step,
                    "train": {
                        "loss/step": loss.item(),
                        "accuracy/step" if config.task == "classification" else "R^2/step": acc,
                        "uar/step" if config.task == "classification" else "R^2/step": train_uar,
                        "learning_rate": cur_lr,
                        "acc0/step": acc0 if config.multi_label else None,
                        "acc1/step": acc1 if config.multi_label else None,
                    }
                })
                if config.verbose:
                    print(f'\nEpoch [{epoch+1}/{num_epochs}], Step [{step+1}/{len(train_loader)}], ' +
                  f'Loss: {loss.item()[0]:.4f}, ' + 
                  'Accuracy: ' if config.task == "classification" else 'R^2: ' + f'{acc:.2f}, ' +
                  f'Learning Rate: {cur_lr:.1e}')
                    
       # train/epoch 정확도 계산--------------------------------------------------------             
        correct_epoch = train_matrix.trace()
        correct_0_epoch = train_matrix[0][0]  # 정상->정상 정답 수
        correct_1_epoch = train_matrix[1][1]  # 장애->장애 정답 수
        acc0_epoch = train_matrix[0][0] / (train_matrix[0][0] + train_matrix[0][1])
        acc1_epoch = train_matrix[1][1] / (train_matrix[1][0] + train_matrix[1][1])
        acc_epoch = correct_epoch / train_matrix.sum()
       # train/epoch 정확도 계산--------------------------------------------------------             
        
        train_loss /= len(train_loader)
        if config.task == 'classification':
            wandb.log({
                "epoch": epoch,
                "global_step": epoch*(len(train_loader) + len(valid_loader)) + step,
                "train": {
                    "loss/epoch": train_loss,
                    "accuracy/epoch": acc_epoch,
                    "acc0/epoch": acc0_epoch if config.multi_label else None,
                    "acc1/epoch": acc1_epoch if config.multi_label else None,
                }
            })
        else:
            wandb.log({
                "epoch": epoch,
                "global_step": epoch*(len(train_loader) + len(valid_loader)) + step,
                "train": {
                    "loss/epoch": train_loss,
                }
            })
            
            
        # validation loop
        model.eval()
        valid_loss = 0.0
        total = 0
        correct = 0
        correct_0 = 0
        correct_1 = 0
        # for UAR -----
        preds = []
        y_true = []

        # -------------
        
        step_pbar = tqdm(valid_loader, total=len(valid_loader), position=1, leave=False)
        with torch.no_grad():
            for step, (inputs, labels) in enumerate(step_pbar):
                inputs, labels = inputs.to(device), labels.to(device)
                
                if("text" in config.model_name):
                    outputs, embeddings = model(inputs)
                    text_keys = inputs[:,-1,0]


                loss = criterion(outputs, labels, text_keys)
                
                valid_loss += loss.item()
                
                if config.task == "classification":
                    d = labels.size(0)
                    total += d
                    
                    if config.multi_label:
                        
                        pred = (torch.sigmoid(outputs.data) >= config.acc_thrs).float()
                        pred = outputs.argmax(dim=1, keepdim=True)
    
                        y_true.extend(labels.argmax(dim=1).tolist())
                        preds.extend(pred.squeeze(1).tolist())
                    
                    else:
            
                        pred = outputs.argmax(dim=1, keepdim=True)  # 가장 높은 확률을 가진 클래스를 예측값으로
                        # correct_ += pred.eq(labels.view_as(pred)).sum().item()  # 정답 수를 누적
                        preds.extend(pred.squeeze(1).tolist())
                        y_true.extend(labels.tolist())
                else:
                    ss_res = torch.sum((labels - outputs.data) ** 2)
                    ss_tot = torch.sum((labels - torch.mean(labels)) ** 2)
                    acc = 1 - ss_res / ss_tot
        # Valid 정확도 계산 ------------------------------------------
        matrix = confusion_matrix(y_true, preds, labels=[0,1])
        valid_uar = calc_uar(matrix) 
           
        correct = matrix.trace()
        correct_0 = matrix[0][0]
        correct_1 = matrix[1][1]
        total = matrix.sum()
        
        valid_acc = correct / total # adding tp and tn and dividing by total
        valid_acc0 = correct_0/total
        valid_acc1 = correct_1/total
        # Valid 정확도 계산 ------------------------------------------
                  
        valid_loss /= len(valid_loader)
        
        wandb.log({
            "epoch": epoch,
            "global_step": (epoch+1)*len(train_loader) + epoch*len(valid_loader) + step,
            "eval": {
                "valid_loss": valid_loss,
                "valid_acc" if config.task == 'classification' else 'R^2': valid_acc,
                'valid_UAR' : valid_uar,
                "valid_acc0": valid_acc0 if config.multi_label else None,
                "valid_acc1": valid_acc1 if config.multi_label else None,
            }
        })
        if config.verbose:
            print(f'\n[Validation] loss: {valid_loss:.4f}, ' + 
                  'accuracy: ' if config.task == 'classification' else 'R^2: ' +
                  f'{valid_acc:.2f}')
        
        if acc > best_acc:
            best_acc = valid_acc
            best_loss = valid_loss
        
        if config.save_model:
            torch.save(model.state_dict(), f'epoch{epoch}_checkpoint.pt')
            print(f'model checkpoint saved, epoch {epoch}')
                      
        if config.no_eval:
            continue
        
        if not isinstance(callbacks, list):
            callbacks = list(callbacks)
        flag = False
        for callback in callbacks:
            flag = flag or callback(valid_loss if callback.monitor=="val_loss" else valid_acc)
        if flag: break # early stopping
            
        # torch.save(model.state_dict(), f'epoch{epoch}_checkpoint.pt')
        # print('model saved')
        
    return best_loss, best_acc
        
        
def test_diff(model: nn.Module, 
         data_loader: DataLoader, 
         criterion: Any,
        #  loss_loader: DataLoader,
         device: torch.device, 
         config: Any) -> Tuple[float, float]:
    
    model.eval()  # 모델을 평가 모드로 설정
    test_loss = 0
    correct = 0
    preds = []
    y_true = []
    
    correct_0 = 0
    correct_1 = 0
    preds_0 = []
    preds_1 = []
    y_true_0 = []
    y_true_1 = []
    text_keys = [] # for csv
    results = []
    with torch.no_grad():  # 그래디언트 계산을 비활성화
        for data, target in tqdm(data_loader, desc="Test"):
            data, target = data.to(device), target.to(device)
            
            
            if('text' in config.model_name):
                outputs, embeddings = model(data)
                text_keys = data[:,-1,0]
    
                
            test_loss += criterion(outputs, target, text_keys).item()
            
            if config.multi_label:
                pred = (torch.sigmoid(outputs.data) >= config.acc_thrs).float()
                labels = (target >= config.acc_thrs).float()
                
                preds_0.extend(pred[:, 0].tolist())
                preds_1.extend(pred[:, 1].tolist())
                y_true_0.extend(labels[:, 0].tolist())
                y_true_1.extend(labels[:, 1].tolist())
                
                pred = outputs.argmax(dim=1, keepdim=True)
                y_true.extend(target.argmax(dim=1).tolist())
                preds.extend(pred.squeeze(1).tolist())
                

            else:
                pred = outputs.argmax(dim=1, keepdim=True)  # 가장 높은 확률을 가진 클래스를 예측값으로
                correct += pred.eq(target.view_as(pred)).sum().item()  # 정답 수를 누적
                
                # print(pred)
                preds.extend(pred.squeeze(1).tolist())
                # print(target, type(target), len(target))
                # print(target.shape)
                # print(target.squeeze().tolist())
                y_true.extend(target.tolist())
                text_keys.tolist().extend(data[:, -1, 0].tolist())
                
                if config.result_csv:
                    reverse_text_dict = {v: k for k, v in text_dict.items()}
                    #collecting result for dataframe
                    for label, prediction, text_key in zip(target.tolist(), pred.squeeze(1).tolist(),text_keys.tolist()):
                        text_key = round(text_key,3)
                        text = reverse_text_dict.get(text_key, None)
                        results.append([label,prediction,text])

    test_loss /= len(data_loader.dataset)
    
    labels_name = ["2", "3", "4", "5", "6", "7", "8", "9", "10"] if config.target == 'age' else ["td", "ssd"]
    
    # test 정확도 계산 --------------------------------------------------------------
    matrix = confusion_matrix(y_true, preds, labels =[0,1])
    uar = calc_uar(matrix)
    
    correct = matrix.trace()
    total = len(data_loader.dataset)
    
    test_accuracy = 100.* correct / total
    test_accuracy0 = matrix[0][0] / (matrix[0][0] + matrix[0][1])
    test_accuracy1 = matrix[1][1] / (matrix[1][1] + matrix[1][0])
    # test 정확도 계산 --------------------------------------------------------------
    
    
    print('test set cfm matrix: ',matrix)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{total} ({test_accuracy:.3f}%), UAR: {uar:.3f}')
    wandb.log({
        "test": {
            "test_loss": test_loss,
            "test_acc": test_accuracy,
            "UAR": uar
        }
    })
    cm = wandb.plot.confusion_matrix(
        y_true=y_true,
        preds=preds,
        class_names=labels_name)
    heatmap = wandb.plots.HeatMap(labels_name, labels_name, matrix, show_text=True)
    
    wandb.log({'confusion matrix': cm,
                'heatmap': heatmap
                })
    
    if config.multi_label:
        wandb.log({
        "test": {
            "test_acc0": test_accuracy0,
            "test_acc1": test_accuracy1,
        }
        })
        matrix0= confusion_matrix(y_true_0, preds_0)
        matrix1= confusion_matrix(y_true_1, preds_1)
        cm0 = wandb.plot.confusion_matrix(
            y_true=y_true_0,
            preds=preds_0,
            class_names=labels_name)
        heatmap0 = wandb.plots.HeatMap(labels_name, labels_name, matrix0, show_text=True)
        cm1 = wandb.plot.confusion_matrix(
            y_true=y_true_1,
            preds=preds_1,
            class_names=labels_name)
        heatmap1 = wandb.plots.HeatMap(labels_name, labels_name, matrix1, show_text=True)
        
        wandb.log({
            'confusion matrix_0': cm0,
            'confusion matrix_1': cm1,
            'heatmap_0': heatmap0,
            'heatmap_1': heatmap1
        })
        
    print("classification report")
    print(classification_report(y_true, preds, target_names=labels_name))
    wandb.log({
        "classification_report": classification_report(y_true, preds, target_names=labels_name, output_dict=True)
    })
    
    torch.save(model.state_dict(), f"checkpoint/{config.run_name}.pt")
    
    df = pd.DataFrame(results, columns=['Target', 'Prediction', 'Text Key'])
    df.to_csv(f"{config.run_name}_results.csv", index=False)
    
    return test_loss, test_accuracy