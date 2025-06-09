import argparse
import math
import os

import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
import mindspore
from mindspore import dtype as mstype


PAD = 0
UNK = 1
EOS = 2
SOS = 3
MASK = 4

class PositionalEncoding(nn.Cell):
    "Implement the PE function. No batch support?"
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute positional encodings once in log space.
        pe = ops.zeros((max_len, d_model), dtype=mstype.float32)
        position = ops.arange(0, max_len).unsqueeze(1)
        div_term = ops.exp(ops.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = ops.sin(position * div_term)
        pe[:, 1::2] = ops.cos(position * div_term)
        self.pe = Tensor(pe.unsqueeze(0))

    def construct(self, x):
        # x: (B, T, H)
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout(x)
    

class TrfmSeq2seq(nn.Cell):
    def __init__(self, in_size, hidden_size, out_size, n_layers, dropout=0.1):
        super().__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(in_size, hidden_size)
        self.pe = PositionalEncoding(hidden_size, dropout)
        self.trfm = nn.Transformer(
            d_model=hidden_size,
            nhead=4,
            num_encoder_layers=n_layers,
            num_decoder_layers=n_layers,
            dim_feedforward=hidden_size
        )
        self.out = nn.Dense(hidden_size, out_size)

    def construct(self, src):
        # src: (T, B)
        embedded = self.embed(src)  # (T, B, H)
        embedded = self.pe(embedded)  # (T, B, H)
        hidden = self.trfm(embedded, embedded)  # (T, B, H)
        out = self.out(hidden)  # (T, B, V)
        out = ops.log_softmax(out, axis=2)  # (T, B, V)
        return out  # (T, B, V)

    def _encode(self, src):
        # src: (T, B)
        embedded = self.embed(src)  # (T, B, H)
        embedded = self.pe(embedded)  # (T, B, H)
        output = embedded
        encoder_layers = self.trfm.encoder.layers
        for i in range(len(encoder_layers) - 1):
            output = encoder_layers[i](output, None)  # (T, B, H)
        penul = output.asnumpy()
        output = encoder_layers[-1](output, None)  # (T, B, H)
        if self.trfm.encoder.norm is not None:
            output = self.trfm.encoder.norm(output)  # (T, B, H)
        output = output.asnumpy()
        # mean, max, first*2
        return np.hstack([
            np.mean(output, axis=0),
            np.max(output, axis=0),
            output[0, :, :],  # (B, H)
            penul[0, :, :]   # (B, H)
        ])  # (B, 4H)

    def encode(self, src):
        # src: (T, B)
        batch_size = src.shape[1]
        if batch_size <= 100:
            return self._encode(src)
        else:
            print('There are {:d} molecules. It will take a little time.'.format(batch_size))
            st, ed = 0, 100
            out = self._encode(src[:, st:ed])  # (B, 4H)
            while ed < batch_size:
                st += 100
                ed += 100
                out = np.concatenate([out, self._encode(src[:, st:ed])], axis=0)
            return out

