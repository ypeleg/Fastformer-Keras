

import argparse
from tqdm import tqdm
from tensorflow.keras import Model
from fastformer import Fastformer, Time2Vector, TransformerEncoder
from tensorflow.keras.layers import Input, Concatenate, GlobalAveragePooling1D, Dropout, Dense

def create_model(seq_len, d_k, d_v, n_heads, ff_dim, filter_dim = 5):
    in_seq = Input(shape=(seq_len, filter_dim))
    x = Time2Vector(seq_len)(in_seq)
    x = Concatenate(axis=-1)([in_seq, x])
    x = TransformerEncoder(d_k, d_v, n_heads, ff_dim)((x, x, x))
    x = TransformerEncoder(d_k, d_v, n_heads, ff_dim)((x, x, x))
    x = TransformerEncoder(d_k, d_v, n_heads, ff_dim)((x, x, x))
    x = GlobalAveragePooling1D(data_format = 'channels_first')(x)
    x = Dropout(0.1)(x)
    x = Dense(64, activation = 'relu')(x)
    x = Dropout(0.1)(x)
    out = Dense(1, activation = 'linear')(x)
    model = Model(inputs = in_seq, outputs = out)
    model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae', 'mape'])
    return model

def create_model_fastformer(seq_len, dim, filter_dim = 5):
    in_seq = Input(shape=(seq_len, filter_dim))
    x = Fastformer(dim)(in_seq)
    x = GlobalAveragePooling1D(data_format='channels_first')(x)
    x = Dropout(0.1)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.1)(x)
    out = Dense(1, activation='linear')(x)
    model = Model(inputs=in_seq, outputs=out)
    model.compile(loss='mse', optimizer='adam', metrics=['mae', 'mape'])
    return model

if __name__ == "__main__":

    d_k = 64
    seq_len = 128
    # model = create_model(args.seq_len, args.d_k, args.d_v, args.n_heads, args.ff_dim)
    model = create_model_fastformer(seq_len, d_k)
    model.summary()
