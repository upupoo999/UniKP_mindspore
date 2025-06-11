import mindspore
from mindspore import Tensor, load_param_into_net, load_checkpoint, ops
from mindspore import dtype as mstype
from mindnlp.transformers import T5Tokenizer, T5EncoderModel
import re
import gc
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold
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
        # print('device:',mindspore.context.get_context('device_target'))
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

def Kcat_predict(Ifeature, Label):
    kf = KFold(n_splits=5, shuffle=True)
    All_pre_label = []
    All_real_label = []
    for train_index, test_index in kf.split(Ifeature, Label):
        Train_data, Train_label = Ifeature[train_index], Label[train_index]
        Test_data, Test_label = Ifeature[test_index], Label[test_index]
        model = ExtraTreesRegressor()
        model.fit(Train_data, Train_label)
        Pre_label = model.predict(Test_data)
        All_pre_label.extend(Pre_label)
        All_real_label.extend(Test_label)
    res = pd.DataFrame({'Value': All_real_label, 'Predict_Label': All_pre_label})
    res.to_excel('PreKcat_new/ph_Kcat_5_cv.xlsx')


if __name__ == '__main__':
    # Dataset Load
    mindspore.set_context(device_target='GPU', device_id=0)
    database = np.array(pd.read_excel('./datasets/Generated_pH_unified_smiles_636.xlsx')).T
    sequence = database[1]
    smiles = database[3]
    pH = database[5].reshape([len(smiles), 1])
    Label = database[4]
    for i in range(len(Label)):
        Label[i] = math.log(Label[i], 10)
    print(max(Label), min(Label))
    # # Feature Extractor

    # print('device:',mindspore.context.get_context('device_target'))
    # sequence_input = Seq_to_vec(sequence)
    # print('device:',mindspore.context.get_context('device_target'))
    # smiles_input = smiles_to_vec(smiles)

    # print(sequence_input.shape, sequence_input.shape, pH.shape)
    # feature = np.concatenate((smiles_input, sequence_input), axis=1)
    # with open("PreKcat_new/features_636_pH_PreKcat.pkl", "wb") as f:
    #     pickle.dump(feature, f)
    with open("PreKcat_new/features_636_pH_PreKcat.pkl", "rb") as f:
        feature = pickle.load(f)
    # Modelling
    Kcat_predict(feature, Label)
