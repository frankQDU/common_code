import pandas as pd
from tqdm import tqdm
tqdm.pandas(desc='pandas bar')
import gensim
from gensim.models import Word2Vec
import gc
import numpy as np



def reduce_mem(df):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('{:.2f} Mb, {:.2f} Mb ({:.2f} %)'.format(start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))
    gc.collect()
    return df


for fea in ['tagid']:
    texts = df[fea].values.tolist()
    
    w2v_dim=10
    model = Word2Vec(texts, 
                     vector_size=w2v_dim,
                     epochs=30, 
                     window=15, 
                     workers=-1, 
                     seed=1017, 
                     min_count=1)
    vacab = list(model.wv.key_to_index.keys())
    w2v_feature = np.zeros((len(texts), w2v_dim))
    w2v_feature_avg = np.zeros((len(texts), w2v_dim))
    print(len(texts))

    for i, line in tqdm(enumerate(texts)):
        w2v_feature_avg[i, :] = np.mean(np.array([model.wv.get_vector(word) for word in line] ),axis=0)
    for i in range(w2v_dim):
        df[f'{fea}_w2v_{i}'] = w2v_feature_avg[::,i ]


