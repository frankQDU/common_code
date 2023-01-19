from annoy import AnnoyIndex
import pandas as pd  
import numpy as np
from tqdm import tqdm  
from collections import defaultdict  
import os, math, warnings, math, pickle
from tqdm import tqdm
import faiss
import collections
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from deepctr.feature_column import SparseFeat, VarLenSparseFeat
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from deepmatch.models import *
from deepmatch.utils import sampledsoftmaxloss
warnings.filterwarnings('ignore')

def embdding_sim(click_df, item_emb_df, save_path, topk):
    """
        基于内容的文章embedding相似性矩阵计算
        :param click_df: 数据表
        :param item_emb_df: 文章的embedding
        :param save_path: 保存路径
        :patam topk: 找最相似的topk篇
        return 文章相似性矩阵
        
        思路: 对于每一篇文章， 基于embedding的相似性返回topk个与其最相似的文章， 只不过由于文章数量太多，这里用了faiss进行加速
    """
    
    item_idx_2_rawid_dict = dict(zip(item_emb_df.index, item_emb_df['aid']))
    
    item_emb_cols = [x for x in item_emb_df.columns if 'emb' in x]
    item_emb_np = np.ascontiguousarray(item_emb_df[item_emb_cols].values, dtype=np.float32)
    # 向量进行单位化
    item_emb_np = item_emb_np / np.linalg.norm(item_emb_np, axis=1, keepdims=True)
    item_index = AnnoyIndex(30, 'angular')
    item_index.set_seed(2022)

    index_to_item_dict, item_to_index_dict = {}, {}

    for i in tqdm(range(len(item_emb_np))):
        emb = item_emb_np[i,:]

        item = item_emb_df.loc[i,'aid']
        index_to_item_dict[i] = item
        item_to_index_dict[item] = i
        item_index.add_item(i, emb)

    item_index.build(30)
    result = {}
    for i in tqdm(range(len(item_emb_np))):
        user_emb = item_emb_np[i,:]
        item = item_emb_df.loc[i,'aid']

        ids, distances = item_index.get_nns_by_vector(user_emb,
                                                          topk,
                                                          include_distances=True)
        item_ids = [index_to_item_dict[id] for id in ids]
        # 返回的是距离，距离越小越相似
        item_sim_scores = [2 - distance for distance in distances]
        result[item] = dict(zip(item_ids[1:], item_sim_scores[1:]) )
        
    return result