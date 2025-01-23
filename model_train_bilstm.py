import numpy as np
import pandas as pd
import torch
import os
import random
from tqdm import tqdm
from model.models import BiLSTM, BiLSTM_atten
from torch.utils.data import Dataset, DataLoader
import argparse
from sklearn.metrics import roc_auc_score,accuracy_score,f1_score,precision_score,recall_score,precision_recall_curve,auc
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
             datapath='./train_data/G9L_train.csv',
             exp_path = './train_data/exp_train_fullseq.csv',
             flag='train',
             fold_idx=0):
        super(CustomData, self).__init__()
        # Load data
        fulldata = pd.read_csv(datapath)
        expdata = pd.read_csv(exp_path)
        expdata = expdata[expdata['label']!='undefined']

        fulldata['len_a'] = fulldata['peptide'].apply(lambda x:len(x[:30].replace('X','')))
        fulldata['len_b'] = fulldata['peptide'].apply(lambda x:len(x[30:].replace('X','')))
        # fulldata = fulldata[(fulldata['len_a']>=7) & (fulldata['len_a']<=24) & (fulldata['len_b']>=7) & (fulldata['len_b']<=24)]
        fulldata = fulldata[(fulldata['len_a']>=8) & (fulldata['len_a']<=18) & (fulldata['len_b']>=8) & (fulldata['len_b']<=18)]
        fulldata['peptide'] = fulldata['peptide'].apply(lambda x:x[:25]+x[30:55])

        positive_samples = fulldata[fulldata['label'] == 1] 
        negative_samples = fulldata[fulldata['label'] == 0]

        negative_samples = negative_samples.sample(n=round(5*len(positive_samples))) 

        selected_data = pd.concat([positive_samples, negative_samples])
        # selected_data.drop_duplicates(inplace=True)

        train_df = selected_data.copy()
        train_df = train_df[(train_df['len_a']>=8) & (train_df['len_a']<=18) & (train_df['len_b']>=8) & (train_df['len_b']<=18)]
        train_df['CDR3a'] = train_df['peptide'].apply(lambda x:x[:25].replace('X','')[1:-1])
        train_df['CDR3b'] = train_df['peptide'].apply(lambda x:x[25:].replace('X','')[1:-1])
        train_df['peptide'] = 'GILGFVFTL'
        train_df.rename(columns={'label':'binder'},inplace=True)
        train_df = train_df[['CDR3a','CDR3b','peptide','binder']]

        self.peptides = selected_data['peptide'].values
        self.labels = selected_data['label'].values

        skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=1234)  
        folds = list(skf.split(self.peptides, self.labels))  

        if flag == 'train':
            train_indices, _ = folds[fold_idx] 

            self.peptides = self.peptides[train_indices]
            self.labels = self.labels[train_indices]

        else:
            _, val_indices = folds[fold_idx]  
            self.peptides = self.peptides[val_indices]
            self.labels = self.labels[val_indices]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        peptide = self.peptides[i]
        peptide = np.asarray([AA_dict[char] for char in peptide])
        peptide = torch.FloatTensor(peptide)

        label = torch.FloatTensor([float(self.labels[i])])
        
        return peptide, label

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

        if self.reduction is None:
            return F_loss
        else:
            return torch.mean(F_loss)



def train(model, train_loader, device, optimizer):
    model.train()

    criterion = FocalLoss(alpha=0.9, gamma=2)

    train_loss = 0
    for idx, data in enumerate(train_loader):
        pep = data[0].to(device)
        lbl = data[1].to(device).squeeze()

        # print(f"Device of model: {next(model.parameters()).device}")
        # print(f"Device of data: {pep.device}")

        optimizer.zero_grad()

        # focal loss
        score = model(pep).squeeze()
        loss = criterion(score, lbl)
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
    return auc

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
    lstm_hidden_dim = 32
    hidden_size1 = 64
    hidden_size2 = 32
    num_layers=2
    LR = 1e-3 # 0.00005 
    # LOG_INTERVAL = 3000 
    NUM_EPOCHS = 200

    args = get_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    for fold in range(5):
        best_AUC = 0
        early_stop_count = 0

        model = BiLSTM_atten(lstm_hidden_dim=lstm_hidden_dim,
                            num_layers=num_layers,
                            hidden_size1=hidden_size1,
                            hidden_size2=hidden_size2).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01) 

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

            val_auc = evalute(score, lbl)
            val_auc_list.append(val_auc)


            if val_auc > best_AUC:
                best_AUC = val_auc
                best_epoch = epoch
                early_stop_count = 0

                # save_path = './output/models/BiLSTM/'
                save_path = './output/models/BiLSTM_atten_noexp/'
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                # model_name = 'BiLSTM_'+str(fold)+'.model'
                model_name = 'BiLSTM_atten_'+str(fold)+'.model'

                torch.save(model.state_dict(),save_path+model_name)

                print('BestEpoch={}; Best AUC={:.4f}.'.format(
                    best_epoch, best_AUC
                ))


            else:
                early_stop_count += 1

            if early_stop_count >= 25:
                break