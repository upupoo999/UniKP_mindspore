from build_vocab import WordVocab
from pretrain_trfm import TrfmSeq2seq as TrfmMs
import math
from mindspore import Parameter, Tensor, save_checkpoint, load_param_into_net, load_checkpoint
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import mindspore


PAD = 0
UNK = 1
EOS = 2
SOS = 3
MASK = 4

class PositionalEncoding(nn.Module):
    "Implement the PE function. No batch support?"
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model) # (T,H)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)

class TrfmSeq2seq(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, n_layers, dropout=0.1):
        super(TrfmSeq2seq, self).__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(in_size, hidden_size)
        self.pe = PositionalEncoding(hidden_size, dropout)
        self.trfm = nn.Transformer(d_model=hidden_size, nhead=4, 
        num_encoder_layers=n_layers, num_decoder_layers=n_layers, dim_feedforward=hidden_size)
        self.out = nn.Linear(hidden_size, out_size)

    def forward(self, src):
        # src: (T,B)
        embedded = self.embed(src)  # (T,B,H)
        embedded = self.pe(embedded) # (T,B,H)
        hidden = self.trfm(embedded, embedded) # (T,B,H)
        out = self.out(hidden) # (T,B,V)
        out = F.log_softmax(out, dim=2) # (T,B,V)
        return out # (T,B,V)

    def _encode(self, src):
        # src: (T,B)
        embedded = self.embed(src)  # (T,B,H)
        embedded = self.pe(embedded) # (T,B,H)
        output = embedded
        for i in range(self.trfm.encoder.num_layers - 1):
            output = self.trfm.encoder.layers[i](output, None)  # (T,B,H)
        penul = output.detach().numpy()
        output = self.trfm.encoder.layers[-1](output, None)  # (T,B,H)
        if self.trfm.encoder.norm:
            output = self.trfm.encoder.norm(output) # (T,B,H)
        output = output.detach().numpy()
        # mean, max, first*2
        return np.hstack([np.mean(output, axis=0), np.max(output, axis=0), output[0,:,:], penul[0,:,:] ]) # (B,4H)
    
    def encode(self, src):
        # src: (T,B)
        batch_size = src.shape[1]
        if batch_size<=100:
            return self._encode(src)
        else: # Batch is too large to load
            print('There are {:d} molecules. It will take a little time.'.format(batch_size))
            st,ed = 0,100
            out = self._encode(src[:,st:ed]) # (B,4H)
            while ed<batch_size:
                st += 100
                ed += 100
                out = np.concatenate([out, self._encode(src[:,st:ed])], axis=0)
            return out
        
def get_param_pt_csv():
    vocab = WordVocab.load_vocab('./models/trfm/vocab.pkl')
    trfm = TrfmSeq2seq(len(vocab), 256, len(vocab), 4)
    trfm.cuda()
    pytorch_weights_dict = trfm.state_dict()
    param_torch = pytorch_weights_dict.keys()
    param_torch_lst = pd.DataFrame(param_torch)
    param_torch_lst.to_csv('./model_map/param_pt.csv')


def get_param_ms_csv():
    vocab = WordVocab.load_vocab('vocab.pkl')
    trfm = TrfmMs(len(vocab), 256, len(vocab), 4)
    prams_ms = trfm.parameters_dict().keys()
    prams_ms_lst = pd.DataFrame(prams_ms)
    prams_ms_lst.to_csv('./model_map/param_ms.csv')

def get_map_csv():
    df_ms = pd.read_csv('./model_map/param_ms.csv')
    df_ms.columns = ['id', 'value']
    df_pt = pd.read_csv('./model_map/param_pt.csv')
    df_pt.columns = ['id', 'value']
    df_pt = df_pt.drop(1)
    df_pt = df_pt.reset_index(drop=True)
    ms_param = df_ms['value'].tolist()
    pt_param = df_pt['value'].tolist()
    df = pd.DataFrame({'ms_param': ms_param, 'pt_param': pt_param})
    df.to_csv('./model_map/map.csv', index=False)

def save_param_ms():
    vocab = WordVocab.load_vocab('./models/trfm/vocab.pkl')
    trfm_pt = TrfmSeq2seq(len(vocab), 256, len(vocab), 4)
    trfm_pt.load_state_dict(torch.load('./model_map/trfm_12_23000.pkl', map_location=torch.device('cpu')))
    trfm_ms = TrfmMs(len(vocab), 256, len(vocab), 4)
    df_map = pd.read_csv('./model_map/map.csv')
    ms_param = df_map['ms_param'].tolist()
    pt_param = df_map['pt_param'].tolist()
    param_mapping = dict(zip(pt_param, ms_param))
    ms_values_dict = {}
    for pt_key, param in trfm_pt.named_parameters():
        ms_key = param_mapping.get(pt_key, None)
        if ms_key is not None:
            ms_param = param.cpu().detach().numpy()
            ms_val = Parameter(ms_param, ms_key)
            ms_values_dict[ms_key] = ms_val
    save_checkpoint(ms_values_dict, './models/trfm/ms_trfm_12_23000.ckpt')

def test_param():
    
    vocab = WordVocab.load_vocab('./models/trfm/vocab.pkl')
    trfm_pt = TrfmSeq2seq(len(vocab), 256, len(vocab), 4)
    trfm_pt.load_state_dict(torch.load('./model_map/trfm_12_23000.pkl', map_location=torch.device('cpu')))
    mindspore.set_context(device_target="GPU")
    trfm_ms = TrfmMs(len(vocab), 256, len(vocab), 4)
    load_param_into_net(trfm_ms, load_checkpoint("./models/trfm/ms_trfm_12_23000.ckpt"))

    ms_x = Tensor([[9,8],[8, 18]])
    pt_x = torch.tensor([[9,8],[8, 18]])
    ms_out = trfm_ms(ms_x)
    pt_out = trfm_pt(pt_x)
    print(ms_out)
    print(pt_out)

if __name__ == "__main__":
    try:
        # get_param_pt_csv()
        # get_param_ms_csv()
        # get_map_csv()
        # save_param_ms()
        test_param()
    except KeyboardInterrupt as e:
        print("[STOP]", e)
