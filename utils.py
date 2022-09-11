import pandas as pd
from tqdm import tqdm

tqdm.pandas(desc='pandas bar')
import gc
import time

import gensim
import numpy as np
import pandas as pd
import xgboost as xgb
from gensim.models import Word2Vec


def fe_ima2vec(df, img_path, img_col, filename_extension, model,layer_output_size=128, sub_path = None):
    
    import pandas as pd
    import torch
    from img2vec_pytorch import Img2Vec
    from PIL import Image
    from tqdm import tqdm

    img2vec = Img2Vec(cuda=(torch.cuda.is_available())   , model=model,layer_output_size=layer_output_size)

    fe_img = []
    for idx in tqdm(range(len(df))):
        path = f'{img_path}/{df.loc[idx, img_col]}.{filename_extension}'
        if sub_path:
            path = f'{img_path}/{df.loc[idx, sub_path]}/{df.loc[idx, img_col]}.{filename_extension}'
        try:
            img = Image.open(path)
            vec = img2vec.get_vec(img)
        except:
            vec = [None] * 512
        fe_img.append(vec)

    fe_img = pd.DataFrame(fe_img)
    fe_img.columns = ['img_vec_' + str(i) for i in range(fe_img.shape[1])]
    return fe_img





import datetime


def printlog():
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n"+"=========="*8 + "%s"%nowtime)

    

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

def get_file_info(path):
    '''
    目录下文件信息
    '''
    import os
    from os.path import getsize, join
    for root, _, files in os.walk(path):
        for file in files:
            path_ = join(root, file).replace('\\','/')
            print(f'the size of {path_} is {round(getsize(path_)/(1024 ** 2) , 5)} M')


def timmer(func):
    def wrapper(*args,**kwargs):
        start_time = time.time()
        result=func(*args,**kwargs)
        end_time = time.time()
        m, s = divmod(end_time - start_time, 60)
        h, m = divmod(m, 60)
        print(f'{int(h)}:{int(m)}:{s}')
        return result
    return wrapper


def check_consistence(X_train,X_test,feature_col):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    from sklearn.utils import shuffle
    import xgboost as xgb
    xgb_para = {"nthread":-1,
                "learning_rate":0.01,
                'objective': 'binary:logistic',
                'eval_metric':'auc',
                "max_depth":2,
                "subsample":0.6,
                "colsample_bytree":0.6,
                "lambda":10,
                "alpha":0.05}
    
    X_train['label'] = 0
    X_test['label'] = 1
    df = X_train.append(X_test).reset_index(drop = True)
    df = shuffle(df, random_state=2020)
    
    X_train, X_test, y_train, y_test = train_test_split(
             df[feature_col], df['label'], test_size=0.33, random_state=42)
    
    train_set = xgb.DMatrix(X_train,y_train)
    test_set = xgb.DMatrix(X_test,y_test)
    xgb_model = xgb.train(xgb_para,
                          train_set,
                          evals=[(train_set,'train'),
                                 (test_set, 'test')],
#                           early_stopping_rounds=0, 
                          num_boost_round=100,
                          verbose_eval=50)
    return roc_auc_score(y_test, xgb_model.predict( xgb.DMatrix(X_test[feature_col])))
