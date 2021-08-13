import warnings
warnings.filterwarnings("ignore")

import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from scipy.linalg import svd
from tqdm import tqdm, trange
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow import contrib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR,'../../'))

user_action_preliminary_path = os.path.join(ROOT_DIR, 'data/wedata/wechat_algo_data1/user_action.csv')
user_action_path = os.path.join(ROOT_DIR, 'data/wedata/wechat_algo_data2/user_action.csv')
feed_embeddings_path = os.path.join(ROOT_DIR, 'data/features/feed_embeddings_origin.pkl')

save_path = os.path.join(ROOT_DIR,'data/features/history_feed_embeddings_origin.pkl')



def func(df):
    return list(np.mean(df))

def gen_history_feed_embedding_features(df):
    result = []
    window = 5
    for d in range(1, 16):
        history = df[(df['date_'] < d) & (df['date_'] >= d - window)]
        
        tqdm.pandas(desc=str('processing date:' + str(d)))
        h = history.groupby('userid')['feed_embedding'].progress_apply(func).reset_index()
            
        udata = pd.DataFrame({'userid': h['userid'], 'date_': d, 'avg_emb_rolling': h['feed_embedding']})
    
        result.append(udata)
    return pd.concat(result)


if __name__ == '__main__':
    user_action_preliminary = pd.read_csv(user_action_preliminary_path)
    user_action = pd.read_csv(user_action_path)
    user_action = user_action.append(user_action_preliminary)
    feed_embeddings = pd.read_pickle(feed_embeddings_path)
    user_action = user_action.merge(feed_embeddings, on='feedid', how='left')
    user_action = user_action[['userid', 'date_', 'feed_embedding']]
    
    history_feed_embeddings = gen_history_feed_embedding_features(user_action)
    
    history_feed_embeddings['avg_emb_rolling'] = history_feed_embeddings['avg_emb_rolling'].progress_apply(
        lambda x:list([0.0] * 512) if type(x) != list else list(x))
    history_feed_embeddings.to_pickle(save_path)