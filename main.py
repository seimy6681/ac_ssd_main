import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torchaudio.transforms as T
from transformers import AutoFeatureExtractor
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
import os
import math
import random

from train import train, test, train_with_loss_loader, test_with_loss_loader
from train_diff import train_diff, test_diff
from util.callback import EarlyStopping, SaveBestModel
from util.util import Loader, Parser
import argparse, textwrap
from argparse import Namespace
import wandb
from typing import Tuple, Union
import debugpy
from util.dataset import CustomDataset
torch.cuda.empty_cache()

#============================================
# PATH
#============================================
# DATA_PATH="/home/selinawisco/whisper/data/kochild"
DATA_PATH="/home/selinawisco/hdd/korean_asr"
WORKSPACE_PATH="/home/selinawisco/whisper"
WORKSPACE_PATH="/home/selinawisco/hdd/0.Research"


#============================================
# Debugging
#============================================
# debugpy.listen(5678)
# print("waiting for debugger")
# debugpy.wait_for_client()
# debugpy.breakpoint()
# print('break on this line')

#============================================
# arguments parsing
#============================================
parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
default_group = parser.add_argument_group('default')
default_group.add_argument('--train', default=True, help='train mode')
default_group.add_argument('--test', default=False, action="store_true", help='model test mode')
default_group.add_argument('--checkpoint', type=str, help='model checkpoint to test')
default_group.add_argument('--verbose', action="store_true")
default_group.add_argument('--data_path', type=str, default=DATA_PATH, help="directory path of data files")
default_group.add_argument('--task', type=str, default='classification', help="regression or classification")
default_group.add_argument('--seed', type=int, default=42, help="random seed for data split")
default_group.add_argument('--test_best_model', action="store_true")
default_group.add_argument('--save_name', type=str)
default_group.add_argument('--result_csv', action="store_true")
default_group.add_argument('--debug', action="store_true")

# file 
file_group = parser.add_argument_group('file')
file_group.add_argument('--data_filename', type=str, default='hospital_target_all.csv')
file_group.add_argument('--splitted_data_files', default=False, action='store_true')
file_group.add_argument('--filter_dataset', default=False, action='store_true')
file_group.add_argument('--train_filename', type=str, help="file name of training data")
file_group.add_argument('--valid_filename', type=str, help="file name of validation data")
file_group.add_argument('--test_filename', type=str, help="file name of test data")

# wandb
wandb_group = parser.add_argument_group('wandb')
wandb_group.add_argument('--run_name', type=str, default="test", help="wandb run name")
wandb_group.add_argument('--logging_steps', type=int, default=50, help='wandb log & watch steps')
wandb_group.add_argument('--watch', type=str, default='all', help="wandb.watch parameter log")

# train args
train_group = parser.add_argument_group('train')
train_group.add_argument('--k_fold', type=int, help="k for k fold cross validation")
# train_group.add_argument('--callback', action='extend', nargs='*', type=str, 
#                          help=textwrap.dedent("""\
#                              callback list to use during training
#                              - es: early stopping
#                              - best: saving best model (watching vlidation loss)"""))
train_group.add_argument('--batch_size', type=int, default=16, help='batch size of training')
train_group.add_argument('--epochs', type=int, default=10, help='epochs of training')
train_group.add_argument('--model_embedding', action='store_true')
train_group.add_argument('--model_text_embedding', action='store_true')
train_group.add_argument('--model_id_embedding', action='store_true')
train_group.add_argument('--dynamic_threshold', action='store_true')
train_group.add_argument('--compute_centroids', action='store_true')
train_group.add_argument('--no_eval', default=False, action='store_true')
train_group.add_argument('--save_model',default=False, action='store_true')
train_group.add_argument('--optimizer', type=str, default='adamW',
                    help=textwrap.dedent("""\
                        - adamW: torch.optim.AdamW (default)
                        - adam: torch.optim.Adam
                        - sam: SAM(Sharpness-Aware Minimization) (https://github.com/davda54/sam)"""))
optim_group = parser.add_argument_group('optim')
optim_group.add_argument('--lr', type=float, default=3e-5, help='initial learning rate')
optim_group.add_argument('--weight_decay', type=float)
optim_group.add_argument('--momentum', type=float)

train_group.add_argument('--scheduler', type=str, default='linear', 
                    help=textwrap.dedent("""\
                        - linear: linear scheduler with warmup (default)
                        - cosine_annealing: cosine annealing with warmup (https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup)"""))
schdl_group = parser.add_argument_group('scheduler')
schdl_group.add_argument('--warmup_ratio', type=float, default=0.1, help="warm up ratio of training steps")

# TODO: add cosine annealing args
cosine_group = parser.add_argument_group('cosine')
cosine_group.add_argument('--cycle_mult', type=float, help="Cycle steps magnification")
cosine_group.add_argument('--min_lr', type=float, default=12e-7)
cosine_group.add_argument('--gamma', type=float, help="Decrease rate of max learning rate by cycle")
cosine_group.add_argument('--first_cycle_steps', type=int, help="First cycle step size")
cosine_group.add_argument('--num_cycles', type=int, default=1, help="number of cycles")

train_group.add_argument('--loss', type=str, default="ce",
                    help=textwrap.dedent("""\
                        - mse: mean square error
                        - ce: cross entropy
                        - bce: binary cross entropy
                        - bce_logit: bce for independent multi label (soft label)
                        - focal: focal
                        - cb: class balanced loss using focal loss
                        - hybrid: Hybrid contrastive loss
                        - balancing: In-class Feaure Balancing (Word 0:1 Ratio)
                        - """))

focal_group = parser.add_argument_group('focal')
focal_group.add_argument('--alpha', type=float, help="alpha of focal loss")
focal_group.add_argument('--focal_gamma', type=float, help="gamma of focal loss and class balanced loss (focal)")

cb_group = parser.add_argument_group('cb')
cb_group.add_argument('--beta', type=float, help="beta of focal loss and class balaced loss (focal)")
cb_group.add_argument('--cb_gamma', type=float, help="gamma of focal loss and class balanced loss (focal)")
cb_group.add_argument('--loss_type', type=str, help="loss type used in class balanced loss (default focal)")

# model config
train_group.add_argument('--model_name', type=str, default='Wav2Vec2_Base_Classifier', 
                    help=textwrap.dedent("""\
                    ! 대소문자 구별 없음
                    - Wav2Vec2_Base_Classifier/Regressor
                    - Resnet50
                    - Whisper_Base_Classifier/Regressor
                    - Whisper_Large_Classifier/Regressor
                    - Distil_Whisper_Large_Classifier/Regressor"""))
train_group.add_argument('--multi_label', default=False, action='store_true')
train_group.add_argument('--age_lambda', type=float, default=0.5, help="soft labeling lambda*(1/age)")
train_group.add_argument('--acc_thrs', type=float, default=0.5, help="threshold of logit to predict 1")
train_group.add_argument('--gradient_accumulation_steps', type=int, default=1)

model_group = parser.add_argument_group('model')
model_group.add_argument('--pretrained', default=True)
model_group.add_argument('--num_freeze', type=int, default=0, help="freeze n encoder layers (0 to n-1)")

# dataset config
dataset_group = parser.add_argument_group('dataset')
dataset_group.add_argument('--data_length', type=int, default=3, help='max length of waveform data')
dataset_group.add_argument('--word', type=str, required=False, help='a single string input')
dataset_group.add_argument('--test_size', type=float, default=0.3, help="test size rate over entire dataset")
dataset_group.add_argument('--valid_size', type=float, default=0.2, help="validation size rate over entire dataset")
dataset_group.add_argument('--sampling_rate', type=int, default=16000, help="sampling rate of the audio")
dataset_group.add_argument('--data_type', type=str, default='wave', help="wave or spectrogram")
dataset_group.add_argument('--target', type=str, default='new_label', help='target column in dataset csv file')
dataset_group.add_argument('--no_shuffle', default=True, action='store_false')
dataset_group.add_argument('--augmented', default=False, action='store_true')
dataset_group.add_argument('--processor', type=str, help="huggingface feature extractor path")
dataset_group.add_argument('--combine_strategy', type=str, default="before", 
                           help=textwrap.dedent('''\
                               before: combine mel spectrogram and age tensor -> log10
                               after: combine log mel spectrogram and age tensor'''))
dataset_group.add_argument('--label_smoothing_by_text', default=False, action='store_true')

# TODO: transform args (specaugment)

# spectrogram config
spec_group = parser.add_argument_group('spectrogram')
spec_group.add_argument('--n_fft', type=int, default=400, 
                    help=textwrap.dedent('''\
                    config for torchaudio.transforms.Spectrogram below
                    (https://pytorch.org/audio/main/generated/torchaudio.transforms.Spectrogram.html)'''))
spec_group.add_argument('--n_mels', type=int, default=224) # default 128
spec_group.add_argument('--win_length', type=int)
spec_group.add_argument('--hop_length', type=int, default=160) # 16000*3 / 224 = 214...
spec_group.add_argument('--pad', type=int, default=0)
spec_group.add_argument('--power', type=float, default=2.0)
spec_group.add_argument('--normalized', default=True)
spec_group.add_argument('--center', default=True)
spec_group.add_argument('--onesided', default=True)
spec_group.add_argument('--n_channels', type=int, default=3, help="number of channels of spectrogram images")

trans_group = parser.add_argument_group('transforms')
trans_group.add_argument('--transform', default=False, action="store_true")
trans_group.add_argument('--time_mask', default=True)
trans_group.add_argument('--freq_mask', default=True)
trans_group.add_argument('--age_momentum', type=float, default=1.0, help="padding value of spectrogram (normalized age * momentum)")
trans_group.add_argument('--scaling', default=False, action='store_true', help="min max scaling to final spectrogram (without age embedding data)")


def stratified_group_split(df: pd.DataFrame, 
                           n_splits: Union[int, float], 
                           config: Namespace) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(n_splits, int):
        splitter = StratifiedGroupKFold(n_splits=n_splits, random_state=config.seed if config.no_shuffle else None, shuffle=config.no_shuffle)
        split = splitter.split(df, df[config.target], groups=df['id'])
        train_idx, test_idx = next(split)
    else:
        test_size = n_splits
        new_df = pd.concat([df, pd.DataFrame({
            'age': [2, 2, 2, 2, 2, 2, 10, 10, 10, 10, 10, 10],
            'new_label': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        })], ignore_index=True, join='outer')
        # train_df, test_df = train_test_split(new_df, test_size=test_size, stratify=pd.concat([new_df['new_label'], new_df['age']], axis=1), random_state=config.seed)    
        train_df, test_df = train_test_split(new_df, test_size=test_size, stratify=pd.concat([new_df['new_label']], axis=1), random_state=config.seed)    
        train_df.dropna(axis=0, inplace=True); test_df.dropna(axis=0, inplace=True)
        
        train_idx = np.array(train_df.index)
        test_idx = np.array(test_df.index)
    
    return train_idx, test_idx

if __name__=='__main__':
    config = parser.parse_args()
    arg_parser = Parser()
    args = arg_parser(parser, config)
    
    seed = config.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
        
    if config.processor:
        processor = AutoFeatureExtractor.from_pretrained(config.processor, 
                                                         feature_size=config.n_mels,
                                                         hop_length=config.hop_length)
    else:
        processor = None
        
    if config.no_eval:
        test_split, valid_split = config.test_size, config.valid_size
    else:
        test_split, valid_split = int(1/config.test_size), int((1-config.test_size)/config.valid_size)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)
    
    if config.debug:
        # Set the address and port for the debugger to listen on
        debugpy.listen(("0.0.0.0", 5678))

        print("Waiting for debugger to attach...")
        debugpy.wait_for_client()

        # Optionally, you can add a breakpoint here to stop the execution
        debugpy.breakpoint()
        print('Debugger attached')
        
    if config.transform:
        transforms = nn.Sequential(
            T.TimeMasking(time_mask_param=10),
            T.FrequencyMasking(freq_mask_param=10)
        )
    
    # TODO: create dataset, dataloader
    if config.splitted_data_files:
        test_df = pd.read_csv(os.path.join(DATA_PATH, config.test_filename))
        if(config.filter_dataset):
            test_df = test_df[test_df["target_text"]==config.word] #filtering dataset to contain a single target word
            
        if config.train:
            if config.valid_filename:
                train_df = pd.read_csv(os.path.join(DATA_PATH, config.train_filename))
                if(config.filter_dataset):
                    train_df = train_df[train_df["target_text"]==config.word] #filtering dataset to contain a single target word
                
                valid_df = pd.read_csv(os.path.join(DATA_PATH, config.valid_filename))
                if(config.filter_dataset):
                    valid_df = valid_df[valid_df["target_text"]==config.word] #filtering dataset to contain a single target word
            else:
                train_df = pd.read_csv(os.path.join(DATA_PATH, config.train_filename))
                if(config.filter_dataset):
                    train_df = train_df[train_df["target_text"]==config.word] #filtering dataset to contain a single target word
                    # Sample train_df to have exactly 140 rows where 'new_label' == 0
                    # sampled_zero_label_df = train_df[train_df['new_label'] == 0].sample(n=140, replace=True, random_state=42)

                    # # Combine the sampled zero label rows with the rest of the dataframe
                    # train_df = pd.concat([sampled_zero_label_df, train_df[train_df['new_label'] != 0]])

                    # # Reset index if necessary
                    # train_df = train_df.reset_index(drop=True)
                    
                    print(f'target word: {config.word} (train: {len(train_df)}, test: {len(test_df)})')
    else:
        data_df = pd.read_csv(os.path.join(DATA_PATH, config.data_filename))
        train_idx, test_idx = stratified_group_split(data_df, test_split, config)
        train_df = data_df.iloc[train_idx, :]
        test_df = data_df.iloc[test_idx, :]
        
    if not config.k_fold and not config.valid_filename:
        train_idx, valid_idx = stratified_group_split(train_df, valid_split, config)
        valid_df = train_df.iloc[valid_idx, :]
        train_df = train_df.iloc[train_idx, :]
    
    k = config.k_fold if config.k_fold else 1
    if config.k_fold: 
        kf = StratifiedGroupKFold(n_splits=config.k_fold, random_state=config.seed if config.no_shuffle else None, shuffle=config.no_shuffle)
        fold = kf.split(train_df, train_df[config.target], groups=train_df['id'])
    else: 
        fold = None
        if config.train:
            train_dataset = CustomDataset(config, file=train_df, data_type=config.data_type, transform=transforms if config.transform else None, processor=processor)
            valid_dataset = CustomDataset(config, file=valid_df, data_type=config.data_type, transform=transforms if config.transform else None, processor=processor)
            if "text" in config.model_name:
                precompute_dataset = CustomDataset(config, file=train_df, data_type=config.data_type, transform=transforms if config.transform else None, processor=processor)
        test_dataset = CustomDataset(config, file=test_df, data_type=config.data_type, transform=transforms if config.transform else None, processor=processor)
        
    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
        
    for i in range(k):
        if fold:
            train_idx, valid_idx = next(fold)
            valid_df_tmp = train_df.iloc[valid_idx, :]
            train_df_tmp = train_df.iloc[train_idx, :]
        
            if config.train:
                train_dataset = CustomDataset(config, file=train_df_tmp, data_type=config.data_type, transform=transforms if config.transform else None, processor=processor)
                valid_dataset = CustomDataset(config, file=valid_df_tmp, data_type=config.data_type, transform=transforms if config.transform else None, processor=processor)
                
            test_dataset = CustomDataset(config, file=test_df, data_type=config.data_type, transform=transforms if config.transform else None, processor=processor)
            
        if config.no_eval:
            train_dataset = train_dataset + valid_dataset
            valid_dataset = test_dataset
        
        if config.train:
            # if config.curriculum:
        
            train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=config.no_shuffle,) # collate_fn=train_dataset.collate_fn)
            valid_loader = DataLoader(dataset=valid_dataset, batch_size=config.batch_size, shuffle=config.no_shuffle,) # collate_fn=valid_dataset.collate_fn)
            if "text" in config.model_name:
                precompute_loader = DataLoader(dataset=precompute_dataset, batch_size=config.batch_size, shuffle=config.no_shuffle,)
                loss_loader = DataLoader(dataset=precompute_dataset, batch_size=64, shuffle=config.no_shuffle,)

        test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size, shuffle=config.no_shuffle,) # collate_fn=test_dataset.collate_fn)
        
        if config.task == "classification":
            args['model']['num_classes'] = len(train_dataset.get_label_list())
        if config.loss == "cb":
            # TODO: samples_per_cls, no_of_classes 넣기
            pass
        if config.loss == "cb" or config.loss == "focal":
            args['loss'] = dict(args['loss'], **args[config.loss])
        if config.scheduler == "cosine_annealing":
            if config.first_cycle_steps == None:
                args['cosine']['first_cycle_steps'] = config.epochs * (len(train_loader) // config.gradient_accumulation_steps) // config.num_cycles
            args['cosine']['warmup_steps'] = config.epochs * (len(train_loader) // config.gradient_accumulation_steps) * config.warmup_ratio
            args['scheduler'] = dict(args['scheduler'], **args['cosine'])
        else:
            args['scheduler']['total_steps'] = config.epochs * math.ceil(len(train_loader) / config.gradient_accumulation_steps)
        if "embedding" in config.model_name:
            args['model']['device'] = device
            
            
        model = Loader.load_model(config.model_name, **args['model']).to(device)
        if torch.cuda.device_count() > 1:
            print(f"Let's use {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model)

        #loading model for test
        if config.test:
            model.load_state_dict(torch.load(config.checkpoint))
            
            
        criterion = Loader.load_criterion(config.loss, **args[config.loss] if config.loss in args else {})
        optimizer = Loader.load_optimizer(config.optimizer, model.parameters(), **args["optim"])
        scheduler = Loader.load_scheduler(config.scheduler, optimizer, **args['scheduler'])
        # TODO: callback
        es = EarlyStopping(model, monitor="val_loss", patience=3, delta=1e-6, verbose=True)
        sb = SaveBestModel(model, monitor="val_loss", path=f"{config.run_name if not config.checkpoint else config.checkpoint}_best"+(f"_{i}" if config.k_fold else "")+".pt" , verbose=True)
        callbacks = [es, sb]
        
        # TODO: wandn init, training code
        wandb.init(project="children_age_classification",
                config={
                    "task": config.task,
                    "data": config.data_filename,
                    "data_length": config.data_length,
                    "data_type": config.data_type,
                    "batch_size": config.batch_size,
                    "target": config.target,
                    "model": config.model_name,
                    "epoch": config.epochs,
                    "loss": config.loss,
                    "optimizer": config.optimizer,
                    "scheduler": config.scheduler,
                    "warmup_ratio": config.warmup_ratio,
                    "learning_rate": config.lr,
                    "transform": config.transform
                })
        wandb.run.log_code(".", include_fn=lambda path: path.endswith(".sh") or path.endswith("main.py"))
        if config.run_name:
            wandb.run.name = config.run_name + (f"_{i}" if config.k_fold else "")
        
        if not config.test:
            if config.loss == 'hybrid':
                train_loss, train_acc = train_with_loss_loader(model, criterion, optimizer, scheduler, 
                                        train_loader, loss_loader, valid_loader, callbacks, device, config)
            elif config.loss == 'difficulty':
                train_loss, train_acc = train_diff(model, criterion, optimizer, scheduler, 
                                        train_loader, valid_loader, callbacks, device, config)
            elif config.loss == 'balancing':
                train_loss, train_acc = train_diff(model, criterion, optimizer, scheduler, 
                                        train_loader, valid_loader, callbacks, device, config)
            else: #ce
                train_loss, train_acc = train(model, criterion, optimizer, scheduler, 
                                        train_loader, valid_loader, callbacks, device, config)
        if config.test_best_model:
            model.load_state_dict(torch.load(sb.path))
    
        if config.loss == 'hybrid':
            test_loss, test_acc = test_with_loss_loader(model, test_loader, criterion, loss_loader,device, config)
        # elif config.loss == 'difficulty' or 'balancing':
        #     test_loss, test_acc = test_diff(model, test_loader, criterion, device, config)
        else:
            test_loss, test_acc = test(model, test_loader, criterion, device, config)
        
        if not config.test:    
            history["train_acc"].append(train_acc)
            history["train_loss"].append(train_loss)
        history["test_acc"].append(test_acc)
        history["test_loss"].append(test_loss)
        
        wandb.finish()
        
    if config.k_fold:
        #TODO: 전체 결과 저장하는 wandb run
        wandb.init(project="children_age_classification",
                config={
                    "task": config.task,
                    "data": config.data_filename,
                    "data_length": config.data_length,
                    "data_type": config.data_type,
                    "batch_size": config.batch_size,
                    "target": config.target,
                    "model": config.model_name,
                    "epoch": config.epochs,
                    "loss": config.loss,
                    "optimizer": config.optimizer,
                    "scheduler": config.scheduler,
                    "warmup_ratio": config.warmup_ratio,
                    "learning_rate": config.lr,
                    "transform": config.transform
                })
        if config.run_name:
            wandb.run.name = config.run_name + "_total"
        
        for i in range(k):
            wandb.log({
                "train_acc": history['train_acc'][i],
                "train_loss": history['train_loss'][i],
                "test_acc": history['test_acc'][i],
                "test_loss": history['test_loss'][i]
            })
            
    