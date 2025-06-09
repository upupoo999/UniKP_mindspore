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


def Kcat_predict(Ifeature, Label, sequence_new, Smiles_new, ECNumber_new, Organism_new, Substrate_new, Type_new):
    for i in tqdm(range(5), desc="epoch"):  
        # Generate training or test set index
        ALL_index = [j for j in range(len(Ifeature))]
        train_index = np.array(random.sample(ALL_index, int(len(ALL_index)*0.9)))
        Training_or_test = []
        for j in range(len(ALL_index)):
            if ALL_index[j] in train_index:
                Training_or_test.append(0)
            else:
                Training_or_test.append(1)
        Training_or_test = np.array(Training_or_test)
        Train_data, Train_label = Ifeature[train_index], Label[train_index]
        
        model = ExtraTreesRegressor()
        model.fit(Train_data, Train_label)
        
        Pre_all_label = model.predict(Ifeature)
        
        res = pd.DataFrame({
            'sequence': sequence_new, 
            'smiles': Smiles_new, 
            'ECNumber': ECNumber_new,
            'Organism': Organism_new, 
            'Substrate': Substrate_new, 
            'Type': Type_new,
            'Label': Label, 
            'Predict_Label': Pre_all_label, 
            'Training or test': Training_or_test
        })
        
        res.to_excel(f'PreKcat_new/{i+1}_all_samples_metrics.xlsx')
        
        with open(f'PreKcat_new/{i+1}_model.pkl', "wb") as f:
            pickle.dump(model, f)


def get_features():
    mindspore.set_context(device_target='GPU', device_id=0)
    # Dataset Load
    with open('./datasets/Kcat_combination_0918_wildtype_mutant.json', 'r') as file:
        datasets = json.load(file)
    # print(len(datasets))
    sequence = [data['Sequence'] for data in datasets]
    Smiles = [data['Smiles'] for data in datasets]
    # Feature Extractor
    sequence_input = Seq_to_vec(sequence)
    smiles_input = smiles_to_vec(Smiles)
    feature = np.concatenate((smiles_input, sequence_input), axis=1)
    with open("PreKcat_new/features_16838_PreKcat.pkl", "wb") as f:
        pickle.dump(feature, f)


def train_and_predict():
    # Dataset Load
    with open('./datasets/Kcat_combination_0918_wildtype_mutant.json', 'r') as file:
        datasets = json.load(file)
    # print(len(datasets))
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
    print(max(Label), min(Label))
    # Feature Extractor
    with open("PreKcat_new/features_16838_PreKcat.pkl", "rb") as f:
        feature = pickle.load(f)
    Label = np.array(Label)
    # Input dataset
    feature_new = []
    Label_new = []
    sequence_new = []
    Smiles_new = []
    ECNumber_new = []
    Organism_new = []
    Substrate_new = []
    Type_new = []
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
    Label_new = np.array(Label_new)
    feature_new = np.array(feature_new)
    # Modelling
    Kcat_predict(feature_new, Label_new, sequence_new, Smiles_new, ECNumber_new,
                Organism_new, Substrate_new, Type_new)
if __name__ == '__main__':
    get_features()
    train_and_predict()