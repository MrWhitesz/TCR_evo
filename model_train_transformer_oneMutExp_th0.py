import numpy as np
import pandas as pd
import torch
import os
import random
from tqdm import tqdm
from model.models import BiLSTM, BiLSTM_atten, Transformer_atten
from torch.utils.data import Dataset, DataLoader
import argparse
from sklearn.metrics import roc_auc_score,accuracy_score,f1_score,precision_score,recall_score,precision_recall_curve,auc,average_precision_score
import logging
import joblib
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

AA_dict = {'X':0,  'Y':1,  'S':2,  'M':3,  'R':4,  'E':5,  'I':6,  'N':7,  'V':8,  'G':9, 'L':10,
           'D':11, 'T':12, 'W':13, 'H':14, 'K':15, 'A':16, 'F':17, 'Q':18, 'C':19, 'P':20,'*':21}


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def seed_torch(seed = 1998):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministics = True

class CustomData(Dataset):
    def __init__(self, 
            #  datapath='/share/home/biopharm/bsz/mengmeng/train_data/G9L_train.csv',
             exp_path = '/share/home/biopharm/bsz/mengmeng/train_data/exp_train_fullseq_th0.csv',
             flag='train',
             fold_idx=0,
             exp_weight=1,):
        super(CustomData, self).__init__()
        # Load data
        # fulldata = pd.read_csv(datapath)
        expdata = pd.read_csv(exp_path) 
        expdata = expdata[expdata['label']!='undefined']

        selected_data = expdata
        print(selected_data.shape)
        self.peptides = selected_data['peptide'].values
        self.labels = selected_data['label'].values

        skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=1234)  
        folds = list(skf.split(self.peptides, self.labels)) 

        if flag == 'train':
            self.peptides = self.peptides
            self.labels = self.labels
            # 设置权重，阳性数据权重更高
            self.weights = np.ones(len(self.labels), dtype=np.float32)  
            self.weights[self.labels == 1] = exp_weight 

            print(len(self.labels[self.labels==1]))
            # print(self.weights)
        else:
            _, val_indices = folds[fold_idx] 
            self.peptides = self.peptides[val_indices]
            self.labels = self.labels[val_indices]

            self.weights = np.ones(len(self.labels), dtype=np.float32)

            print(len(self.labels[self.labels==1]))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        peptide = self.peptides[i]
        peptide = np.asarray([AA_dict[char] for char in peptide])
        peptide = torch.FloatTensor(peptide)

        label = torch.FloatTensor([float(self.labels[i])])
        
        weight = self.weights[i]
        
        return peptide, label, weight

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss) 
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction =='none':
            return F_loss
        else:
            return torch.mean(F_loss)

def train(model, train_loader, device, optimizer):
    model.train()

    criterion = FocalLoss(alpha=0.9, gamma=2, reduction='none')

    train_loss = 0
    for idx, data in enumerate(train_loader):
        pep = data[0].to(device)
        lbl = data[1].to(device).squeeze()

        weights = data[2].to(device)

        # print(f"Device of model: {next(model.parameters()).device}")
        # print(f"Device of data: {pep.device}")

        optimizer.zero_grad()

        # focal loss
        score = model(pep).squeeze()
        loss_f = criterion(score, lbl)
        loss = (loss_f*weights).mean()
        train_loss += loss.item()

        loss.backward()
        optimizer.step()

    train_loss = train_loss / len(train_loader)
    return train_loss

def predict(model, device, loader):
    model.eval()
    preds = torch.Tensor()
    labels = torch.Tensor()

    with torch.no_grad():
        for data in loader:
            pep = data[0].to(device)
            lbl = data[1]

            score = model(pep)
            preds = torch.cat((preds, score.cpu()), 0)
            labels = torch.cat((labels, lbl), 0)
    
    return preds.numpy().flatten(), labels.numpy().flatten()

def evalute(score, lbl):
    Labels = list()
    Pre = list()
    for n,item in enumerate(lbl):
        if (not np.isnan(item)):
            Labels.append(int(item))
            Pre.append(score[n])

    auc = roc_auc_score(Labels,Pre)
    prauc = average_precision_score(Labels,Pre)
    return auc, prauc

def get_args():
    parser = argparse.ArgumentParser(description='Train the model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    return parser.parse_args()

def LoadDataset(fold):
    #Load Train and Val Data
    trainDataset = CustomData(fold_idx=fold, flag='train')
    valDataset = CustomData(fold_idx=fold, flag='valid')

    return trainDataset, valDataset


if __name__ == '__main__':
    seed_torch(1998)

    #Train setting
    BATCH_SIZE = 128 
    # embedding_d = 48
    transformer_hidden_dim = 256
    hidden_size1 = 64
    hidden_size2 = 32
    num_layers=2
    LR = 1e-3 # 0.00005 
    NUM_EPOCHS = 6

    args = get_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    for fold in range(5):
        best_AUC = 0
        best_PRAUC = 0
        early_stop_count = 0

        model = Transformer_atten(transformer_hidden_dim=transformer_hidden_dim,
                                num_layers=num_layers,
                                hidden_size1=hidden_size1,
                                hidden_size2=hidden_size2).to(device)

        model_dir = './output/models/Transformer_atten_noexp/'
        model_name = f'Transformer_atten_{fold}.model'
        model.load_state_dict(torch.load(model_dir+model_name))
        frozen_layers = [
                            'blosum50',
                            'positionalEncodings_a','positionalEncodings_b',
                            'transformer_encoder_a','transformer_encoder_b',
                            'attention',
                          ]
        for name, param in model.named_parameters():
            if any(layer in name for layer in frozen_layers):
                param.requires_grad = False

        # for name, param in model.named_parameters():
        #     print(f"{name} requires_grad: {param.requires_grad}")

        # optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01) 
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=0.01)


        print('-'*15+'fold: '+str(fold)+'-'*15)
        trainDataset, valDataset = LoadDataset(fold=fold)
        train_loader = DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=True)
        valid_loader = DataLoader(valDataset, batch_size=BATCH_SIZE, shuffle=False)

        train_loss_list = []
        val_auc_list = []
        for epoch in tqdm(range(NUM_EPOCHS)):
        # for epoch in (range(NUM_EPOCHS)):
            train_loss = train(model, train_loader, device, optimizer)
            train_loss_list.append(train_loss)

            score, lbl = predict(model, device, valid_loader)

            val_auc, val_prauc = evalute(score, lbl)
            val_auc_list.append(val_auc)

        save_path = './output/models/Transformer_atten_onemut_th0/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # model_name = 'BiLSTM_'+str(fold)+'.model'
        model_name = 'Transformer_atten_'+str(fold)+'.model'

        torch.save(model.state_dict(),save_path+model_name)
