import mindspore
from mindspore import Tensor, load_param_into_net, load_checkpoint, ops
from mindspore import dtype as mstype
from mindnlp.transformers import T5Tokenizer, T5EncoderModel
import re
import gc
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
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


def Kcat_predict(Ifeature, Label):
    for i in range(1):
        model = ExtraTreesRegressor()
        model.fit(Ifeature, Label)
        with open('PreKcat_new/'+str(i)+"_model.pkl", "wb") as f:
            pickle.dump(model, f)


if __name__ == '__main__':
    mindspore.set_context(device_target='GPU', device_id=0)
    with open('./datasets/Kcat_combination_0918_wildtype_mutant.json', 'r') as file:
        datasets = json.load(file)
    # print(len(datasets))
    Label = [float(data['Value']) for data in datasets]
    Smiles = [data['Smiles']for data in datasets]
    for i in range(len(Label)):
        if Label[i] == 0:
            Label[i] = -10000000000
        else:
            Label[i] = math.log(Label[i], 10)
    with open("PreKcat_new/features_16838_PreKcat.pkl", "rb") as f:
        feature = pickle.load(f)
    Label = np.array(Label)
    Label_new = []
    feature_new = []
    for i in range(len(Label)):
        if -10000000000 < Label[i] and '.' not in Smiles[i]:
            Label_new.append(Label[i])
            feature_new.append(feature[i])
    print(len(Label_new))
    Label_new = np.array(Label_new)
    feature_new = np.array(feature_new)
    Kcat_predict(feature_new, Label_new)
