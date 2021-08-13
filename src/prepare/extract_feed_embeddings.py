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

feed_embeddings_path = os.path.join(ROOT_DIR,'data/wedata/wechat_algo_data2/feed_embeddings.csv')

save_path = os.path.join(ROOT_DIR,'data/features/feed_embeddings_origin.pkl')


def PCA_reduceDim(X, n):
    pca = PCA(n_components=n) #实例化
    return pca.fit_transform(X) #获取新矩阵

if __name__ == '__main__':
    feed_embeddings = pd.read_csv(feed_embeddings_path)
    
    feed_embedding = feed_embeddings['feed_embedding'].apply(lambda x: np.fromstring(x, dtype=np.float, sep=' '))
    feed_embeddings['feed_embedding'] = list(PCA_reduceDim(np.array(list(feed_embedding)), 512))
    
    feed_embeddings.to_pickle(save_path)