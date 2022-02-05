from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool




def eda_float_base(fea):
    
    df_fea = df[fea]
    tmp = {}
    tmp['fea'] = fea
    tmp['count'] = len(df_fea)
    tmp['nunique'] = df_fea.nunique()
    tmp['nunique_rate'] = tmp['nunique']/tmp['count']
        
    tmp['std'] = df_fea.std()
    
    tmp['missing'] = df_fea.isnull().sum()
    tmp['missing_rate'] = tmp['missing']/tmp['count']
    tmp['mean'] = df_fea.mean()
    
    tmp['min'] = df_fea.min()
        
    tmp['5%'] = df_fea.quantile(.05)
    tmp['25%'] = df_fea.quantile(.25)
    tmp['50%'] = df_fea.quantile(.5)
    tmp['75%'] = df_fea.quantile(.75)
    tmp['95%'] = df_fea.quantile(.95)
        
    tmp['max'] = df_fea.max()
    return tmp
    


feature_cols = df.select_dtypes('number').columns.tolist()
    
pool = ThreadPool(6)
results = list(tqdm(pool.imap(eda_float_base,feature_cols)))
pool.close() 
pool.join()
    