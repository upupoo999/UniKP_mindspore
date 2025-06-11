import mindspore
from mindspore import Tensor, load_param_into_net, load_checkpoint, ops
from mindspore import dtype as mstype
from mindnlp.transformers import T5Tokenizer, T5EncoderModel
import re
import gc
from sklearn.ensemble import ExtraTreesRegressor
import numpy as np
import pandas as pd
import random
import pickle
import math
import json
from utils import split
from build_vocab import WordVocab
from pretrain_trfm import TrfmSeq2seq
from tqdm import tqdm

from collections import Counter
from scipy.ndimage import convolve1d
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
import matplotlib.pyplot as plt

from build_vocab import WordVocab
from sklearn import metrics
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

def smiles_to_vec(Smiles):
    pad_index = 0
    unk_index = 1
    eos_index = 2
    sos_index = 3
    mask_index = 4
    vocab = WordVocab.load_vocab('./models/trfm/vocab.pkl')
    def get_inputs(sm):
        seq_len = 220
        sm = sm.split()
        if len(sm)>218:
            # print('SMILES is too long ({:d})'.format(len(sm)))
            sm = sm[:109]+sm[-109:]
        ids = [vocab.stoi.get(token, unk_index) for token in sm]
        ids = [sos_index] + ids + [eos_index]
        seg = [1]*len(ids)
        padding = [pad_index]*(seq_len - len(ids))
        ids.extend(padding), seg.extend(padding)
        return ids, seg
    def get_array(smiles):
        x_id, x_seg = [], []
        for sm in smiles:
            a,b = get_inputs(sm)
            x_id.append(a)
            x_seg.append(b)
        return Tensor(x_id), Tensor(x_seg)
    
    trfm = TrfmSeq2seq(len(vocab), 256, len(vocab), 4)
    load_param_into_net(trfm, load_checkpoint("./models/trfm/ms_trfm_12_23000.ckpt"))
    trfm.set_train(False)
    x_split = [split(sm) for sm in Smiles]
    xid, xseg = get_array(x_split)
    X = trfm.encode(xid.T)
    return X


def Seq_to_vec(Sequence):
    for i in range(len(Sequence)):
        if len(Sequence[i]) > 1000:
            Sequence[i] = Sequence[i][:500] + Sequence[i][-500:]
    sequences_Example = []
    for i in range(len(Sequence)):
        zj = ''
        for j in range(len(Sequence[i]) - 1):
            zj += Sequence[i][j] + ' '
        zj += Sequence[i][-1]
        sequences_Example.append(zj)
    tokenizer = T5Tokenizer.from_pretrained("./models/prot_t5_xl_uniref50", do_lower_case=False, local_files_only=True)
    model = T5EncoderModel.from_pretrained("./models/prot_t5_xl_uniref50", ms_dtype = mstype.float32, local_files_only=True, device_map=0)
    gc.collect()
    model = model.eval()
    features = []
    for i in range(len(sequences_Example)):
        print('For sequence ', str(i+1))
        sequences_Example_i = sequences_Example[i]
        sequences_Example_i = [re.sub(r"[UZOB]", "X", sequences_Example_i)]
        ids = tokenizer.batch_encode_plus(sequences_Example_i, add_special_tokens=True, padding=True)
        input_ids = Tensor(ids['input_ids'])
        attention_mask = Tensor(ids['attention_mask'])
        embedding = model(input_ids=input_ids, attention_mask=attention_mask)
        embedding = embedding.last_hidden_state
        seq_len = (attention_mask[0] == 1).sum()
        seq_emd = embedding[0][:seq_len - 1]
        features.append(seq_emd.mean(axis=0))
    
    features_normalize = ops.stack(features, axis=0).asnumpy()
    return features_normalize



def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(map(laplace, np.arange(-half_ks, half_ks + 1)))
    return kernel_window


def Smooth_Label(Label_new):
    labels = Label_new
    for i in range(len(labels)):
        labels[i] = labels[i] - min(labels)
    bin_index_per_label = [int(label*10) for label in labels]
    # print(bin_index_per_label)
    Nb = max(bin_index_per_label) + 1
    print(Nb)
    num_samples_of_bins = dict(Counter(bin_index_per_label))
    print(num_samples_of_bins)
    emp_label_dist = [num_samples_of_bins.get(i, 0) for i in range(Nb)]
    print(emp_label_dist, len(emp_label_dist))
    eff_label_dist = []
    beta = 0.9
    for i in range(len(emp_label_dist)):
        eff_label_dist.append((1-math.pow(beta, emp_label_dist[i])) / (1-beta))
    print(eff_label_dist)
    eff_num_per_label = [eff_label_dist[bin_idx] for bin_idx in bin_index_per_label]
    weights = [np.float32(1 / x) for x in eff_num_per_label]
    weights = np.array(weights)
    print(weights)
    print(len(weights))
    return weights


def Kcat_predict(Ifeature, Label, weights):
    for i in range(3, 10):
        print('tag loop' , i)
        kf = KFold(n_splits=5, shuffle=True)
        All_pre_label = []
        All_real_label = []
        for train_index, test_index in kf.split(Ifeature, Label):
            print('tag loop2' , train_index)
            Train_data, Train_label = Ifeature[train_index], Label[train_index]
            Test_data, Test_label = Ifeature[test_index], Label[test_index]
            print('tag xxx')
            model = ExtraTreesRegressor()
            print('tag zzz')
            model.fit(Train_data, Train_label, sample_weight=weights[train_index])
            Pre_label = model.predict(Test_data)
            print('tag yyy')
            All_pre_label.extend(Pre_label)
            All_real_label.extend(Test_label)
        res = pd.DataFrame({'Value': All_real_label, 'Predict_Label': All_pre_label})
        res.to_excel('PreKcat_new/'+str(i)+'_0.9_Re_weighting_Kcat_5_cv'+'.xlsx')


if __name__ == '__main__':
    # Dataset Load
    mindspore.set_context(device_target='GPU', device_id=0)

    with open('./datasets/Kcat_combination_0918_wildtype_mutant.json', 'r') as file:
        datasets = json.load(file)
    print(len(datasets))
    # datasets = datasets[:50]
    sequence = [data['Sequence'] for data in datasets]
    Smiles = [data['Smiles'] for data in datasets]
    Label = [float(data['Value']) for data in datasets]
    ECNumber = [data['ECNumber'] for data in datasets]
    Organism = [data['Organism'] for data in datasets]
    Substrate = [data['Substrate'] for data in datasets]
    Type = [data['Type'] for data in datasets]
    for i in range(len(Label)):
        if Label[i] == 0:
            Label[i] = -10000000000
        else:
            Label[i] = math.log(Label[i], 10)
    Label = np.array(Label)
    print(max(Label), min(Label))
    # Feature Extractor
    # smiles_input = smiles_to_vec(Smiles)
    # sequence_input = Seq_to_vec(sequence)
    # feature = np.concatenate((smiles_input, sequence_input), axis=1)
    # with open("PreKcat_new/features_17010_PreKcat.pkl", "wb") as f:
    #     pickle.dump(feature, f)
    with open("PreKcat_new/features_17010_PreKcat.pkl", "rb") as f:
        feature = pickle.load(f)
    # feature = feature[:50]
    # Input dataset
    feature_new = []
    Label_new = []
    sequence_new = []
    Smiles_new = []
    ECNumber_new = []
    Organism_new = []
    Substrate_new = []
    Type_new = []
    print('tag 1')
    for i in range(len(Label)):
        if -10000000000 < Label[i] and '.' not in Smiles[i]:
            feature_new.append(feature[i])
            Label_new.append(Label[i])
            sequence_new.append(sequence[i])
            Smiles_new.append(Smiles[i])
            ECNumber_new.append(ECNumber[i])
            Organism_new.append(Organism[i])
            Substrate_new.append(Substrate[i])
            Type_new.append(Type[i])
    print(len(Label_new), min(Label_new), max(Label_new))
    feature_new = np.array(feature_new)
    Label_new = np.array(Label_new)
    sl_label = [Label_new[i] for i in range(len(Label_new))]
    weights = Smooth_Label(sl_label)
    # weights = np.ones([len(Label_new)], dtype=float)
    # for i in range(len(weights)):
    #     if Label_new[i] > 5:
    #         weights[i] = 2
    # sum_weights = np.sum(weights)
    # for i in range(len(weights)):
    #     weights[i] /= sum_weights
    # Modelling
    print('tag 2')
    Kcat_predict(feature_new, Label_new, weights)
