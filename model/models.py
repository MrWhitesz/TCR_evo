import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

blosum50 = {
        16 : np.array((5,-2,-1,-2,-1,-1,-1,0,-2,-1,-2,-1,-1,-3,-1,1,0,-3,-2,0)),
         4 : np.array((-2,7,-1,-2,-4,1,0,-3,0,-4,-3,3,-2,-3,-3,-1,-1,-3,-1,-3)),
         7 : np.array((-1,-1,7,2,-2,0,0,0,1,-3,-4,0,-2,-4,-2,1,0,-4,-2,-3)),
        11 : np.array((-2,-2,2,8,-4,0,2,-1,-1,-4,-4,-1,-4,-5,-1,0,-1,-5,-3,-4)),
        19 : np.array((-1,-4,-2,-4,13,-3,-3,-3,-3,-2,-2,-3,-2,-2,-4,-1,-1,-5,-3,-1)),
        18 : np.array((-1,1,0,0,-3,7,2,-2,1,-3,-2,2,0,-4,-1,0,-1,-1,-1,-3)),
         5 : np.array((-1,0,0,2,-3,2,6,-3,0,-4,-3,1,-2,-3,-1,-1,-1,-3,-2,-3)),
         9 : np.array((0,-3,0,-1,-3,-2,-3,8,-2,-4,-4,-2,-3,-4,-2,0,-2,-3,-3,-4)),
        14 : np.array((-2,0,1,-1,-3,1,0,-2,10,-4,-3,0,-1,-1,-2,-1,-2,-3,2,-4)),
         6 : np.array((-1,-4,-3,-4,-2,-3,-4,-4,-4,5,2,-3,2,0,-3,-3,-1,-3,-1,4)),
        10 : np.array((-2,-3,-4,-4,-2,-2,-3,-4,-3,2,5,-3,3,1,-4,-3,-1,-2,-1,1)),
        15 : np.array((-1,3,0,-1,-3,2,1,-2,0,-3,-3,6,-2,-4,-1,0,-1,-3,-2,-3)),
         3 : np.array((-1,-2,-2,-4,-2,0,-2,-3,-1,2,3,-2,7,0,-3,-2,-1,-1,0,1)),
        17 : np.array((-3,-3,-4,-5,-2,-4,-3,-4,-1,0,1,-4,0,8,-4,-3,-2,1,4,-1)),
        20 : np.array((-1,-3,-2,-1,-4,-1,-1,-2,-2,-3,-4,-1,-3,-4,10,-1,-1,-4,-3,-3)),
         2 : np.array((1,-1,1,0,-1,0,-1,0,-1,-3,-3,0,-2,-3,-1,5,2,-4,-2,-2)),
        12 : np.array((0,-1,0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1,2,5,-3,-2,0)),
        13 : np.array((-3,-3,-4,-5,-5,-1,-3,-3,-3,-3,-2,-3,-1,1,-4,-4,-3,15,2,-3)),
         1 : np.array((-2,-1,-2,-3,-3,-1,-2,-3,2,-1,-1,-2,0,4,-3,-2,-2,2,8,-1)),
         8 : np.array((0,-3,-3,-4,-1,-3,-3,-4,-4,4,1,-3,1,-1,-3,-2,0,-3,-1,5)),
         0 : np.array((-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5)),
    }
    

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attn_linear = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, lstm_output):
        # lstm_output: [batch_size, seq_len, hidden_dim * 2]
        attn_weights = F.softmax(self.attn_linear(lstm_output), dim=1)
        attn_applied = torch.bmm(attn_weights.transpose(1, 2), lstm_output)
        return attn_applied, attn_weights

class BiLSTM_atten(nn.Module):
    def __init__(self,
                 dropout=0.2,
                 embedding_dim=20, 
                 lstm_hidden_dim=256,
                 num_layers=1,
                 hidden_size1=1024,
                 hidden_size2=256
                 ):
        super(BiLSTM_atten, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.blosum50 = torch.Tensor(np.array([blosum50[aa] for aa in range(21)]))
        # self.embedding = nn.Embedding(21,embedding_dim)
        
        self.lstm = nn.LSTM(embedding_dim, lstm_hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.lstm_a = nn.LSTM(embedding_dim, lstm_hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.lstm_b = nn.LSTM(embedding_dim, lstm_hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.attention = Attention(lstm_hidden_dim*2)  

        self.classifier = nn.Sequential(nn.Dropout(dropout),
                                        nn.Linear(lstm_hidden_dim*4, hidden_size1), 
                                        nn.ReLU(),
                                        nn.Dropout(dropout),
                                        nn.Linear(hidden_size1, hidden_size2),
                                        nn.ReLU(),
                                        nn.Dropout(dropout),
                                        nn.Linear(hidden_size2, 1),
                                        # nn.Sigmoid()
                                        )
       
    def forward(self, inputs):
        cdr3a = self.blosum50[inputs[:,:25].long()].to(inputs.device)
        cdr3b = self.blosum50[inputs[:,25:].long()].to(inputs.device)
        output_a ,(h_n_a, _) = self.lstm_a(cdr3a)
        output_b ,(h_n_b, _) = self.lstm_b(cdr3b)

        attn_output_a, attn_weights_a = self.attention(output_a)
        attn_output_b, attn_weights_b = self.attention(output_b)
        attn_output_a = attn_output_a.mean(dim=1)
        attn_output_b = attn_output_b.mean(dim=1)
        
        output = torch.cat([attn_output_a, attn_output_b], dim=1)
        output = self.classifier(output)

        return output



class Transformer_atten(nn.Module):
    def __init__(self,
                 num_layers=1,
                 embedding_dim=20,
                 num_heads=4,
                 transformer_hidden_dim=256,
                 hidden_size1=64,
                 hidden_size2=32,
                 dropout=0.2,
                 max_len=50):
        super(Transformer_atten, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.blosum50 = nn.Embedding(21, embedding_dim)
        # self.embedding = nn.Embedding(21,embedding_dim)
        
        self.blosum50.weight.data = torch.Tensor(np.array([blosum50[aa] for aa in range(21)]))
        
        self.positionalEncodings_a = nn.Parameter(torch.rand(25, embedding_dim), requires_grad=True)
        self.positionalEncodings_b = nn.Parameter(torch.rand(25, embedding_dim), requires_grad=True)
        
        encoder_layer_a = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dim_feedforward=transformer_hidden_dim, dropout=dropout)
        self.transformer_encoder_a = nn.TransformerEncoder(encoder_layer_a, num_layers=num_layers)
        
        encoder_layer_b = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dim_feedforward=transformer_hidden_dim, dropout=dropout)
        self.transformer_encoder_b = nn.TransformerEncoder(encoder_layer_b, num_layers=num_layers)
        
        self.attention = Attention(embedding_dim)  # Attention module
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, hidden_size1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size2, 1)
        )
       
    def forward(self, inputs):
        cdr3a = self.blosum50(inputs[:, :25].long())
        cdr3b = self.blosum50(inputs[:, 25:].long())
        # cdr3a = self.embedding(inputs[:, :25].long())
        # cdr3b = self.embedding(inputs[:, 25:].long())
        
        cdr3a = cdr3a + self.positionalEncodings_a.unsqueeze(0)
        cdr3b = cdr3b + self.positionalEncodings_b.unsqueeze(0)
        
        transformer_output_a = self.transformer_encoder_a(cdr3a)
        transformer_output_b = self.transformer_encoder_b(cdr3b)
        
        attn_output_a, attn_weights_a = self.attention(transformer_output_a)
        attn_output_b, attn_weights_b = self.attention(transformer_output_b)
        
        attn_output_a = attn_output_a.mean(dim=1)
        attn_output_b = attn_output_b.mean(dim=1)
        
        output = torch.cat([attn_output_a, attn_output_b], dim=1)
        output = self.classifier(output)

        return output


import esm
esm_model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()

for param in esm_model.parameters():
    param.requires_grad = False

class ESMClassifier(nn.Module):
    def __init__(self, esm_model=esm_model, dropout=0.5):
        super(ESMClassifier, self).__init__()
        self.esm_model = esm_model
        self.dropout = nn.Dropout(dropout)  
        self.classifier = nn.Linear(33, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, tokens):
        results = self.esm_model(tokens)
        token_representations = results['logits'][:, 1:50+1, :]
        
        sequence_representation = token_representations.mean(1)
        sequence_representation = self.dropout(sequence_representation)  
        logits = self.classifier(sequence_representation)
        return self.sigmoid(logits)
    

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),   
            nn.ReLU(True),
            nn.Linear(64, 1)
        )

    def forward(self, encoder_outputs):
        energy = self.projection(encoder_outputs)  # (batch_size, sequence_length, 1)
        weights = F.softmax(energy.squeeze(-1), dim=1)  # (batch_size, sequence_length)

        outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)  # (batch_size, hidden_dim)
        return outputs, weights


class BiLSTM_ESM(nn.Module):
    def __init__(self,
                 esm_model=esm_model,
                 dropout=0.5,
                 lstm_hidden_dim=256,
                 num_layers=1,
                 hidden_size1=1024,
                 hidden_size2=256):
        super(BiLSTM_ESM, self).__init__()
        self.esm_model = esm_model
        self.dropout = nn.Dropout(dropout)
        embedding_dim = 640  
        self.lstm = nn.LSTM(embedding_dim, lstm_hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.self_atten = SelfAttention(lstm_hidden_dim)
        self.classifier = nn.Sequential(nn.Dropout(dropout),
                                        nn.Linear(lstm_hidden_dim*2, hidden_size1),
                                        nn.ReLU(),
                                        nn.Dropout(dropout),
                                        nn.Linear(hidden_size1, hidden_size2),
                                        nn.ReLU(),
                                        nn.Dropout(dropout),
                                        nn.Linear(hidden_size2, 1),
                                        nn.Sigmoid()
                                        )

    def forward(self, tokens):
        with torch.no_grad():
            results = self.esm_model(tokens, repr_layers=[30])

        token_representations = results['representations'][30][:, 1:50+1, :]
        
        lstm_output, (h_n, c_n) = self.lstm(token_representations)
        self_att_out, self_weights = self.self_atten(lstm_output)
        output = self.classifier(self_att_out)
        return output

