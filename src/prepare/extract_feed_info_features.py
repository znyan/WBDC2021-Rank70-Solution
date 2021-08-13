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

feed_info_path = os.path.join(ROOT_DIR, 'data/wedata/wechat_algo_data2/feed_info.csv')
save_path = os.path.join(ROOT_DIR,'data/features/feed_info.pkl')

feed_info = pd.read_csv(feed_info_path)

def gen_Multivalued_Sparse_features(df):
    tqdm.pandas(desc="keyword_list") 
    df['manual_keyword_list'] = df['manual_keyword_list'].progress_apply(lambda x: str(x).split(';') if str(x) != 'nan' else ['-1'])
    df['machine_keyword_list'] = df['machine_keyword_list'].progress_apply(lambda x: str(x).split(';') if str(x) != 'nan' else ['-1'])
    #df['keyword_list'] = df.progress_apply(lambda x: x['manual_keyword_list'] + x['machine_keyword_list'], axis=1)
    
    tqdm.pandas(desc="tag_list") 
    df['manual_tag_list'] = df['manual_tag_list'].progress_apply(lambda x: str(x).split(';') if str(x) != 'nan' else ['-1'])
    df['machine_tag_list'] = df['machine_tag_list'].str.split(';').progress_apply(lambda x: [i.split(' ')[0] for i in x] if type(x) == list else ['-1'])
    #df['tag_list'] = df.progress_apply(lambda x: x['manual_tag_list'] + x['machine_tag_list'], axis=1)

    return df



def gen_embedding_list(df, feature, emb_dim):
    tags_str = df[feature].apply(lambda x: '|'.join(x)).tolist()

    TAG_SET = list(set('|'.join(tags_str).split('|')))
    
    table = contrib.lookup.index_table_from_tensor(mapping=TAG_SET, default_value=-1)
    split_tags = tf.string_split(tags_str,"|")

    tags=tf.SparseTensor(indices=split_tags.indices,
                         values=table.lookup(split_tags.values),
                         dense_shape=split_tags.dense_shape)
    
    embedding_params=tf.Variable(tf.truncated_normal([len(TAG_SET), emb_dim]))
    
    embedded_tags=tf.nn.embedding_lookup_sparse(embedding_params,sp_ids=tags,sp_weights=None)
    
    with tf.Session() as s:
        s.run([tf.global_variables_initializer(), tf.tables_initializer()])
        result = s.run([embedded_tags])[0]
    
    return list(result)

if __name__ == '__main__':
    feed_info = pd.read_csv(feed_info_path)
    
    # preprocesse
    feed_info[["bgm_song_id", "bgm_singer_id"]] += 1 
    feed_info[["bgm_song_id", "bgm_singer_id", "videoplayseconds"]] = feed_info[["bgm_song_id", "bgm_singer_id", "videoplayseconds"]].fillna(0)
    feed_info['videoplayseconds'] = np.log(feed_info['videoplayseconds'])
    feed_info['bgm_song_id'] = feed_info['bgm_song_id'].astype('int64')
    feed_info['bgm_singer_id'] = feed_info['bgm_singer_id'].astype('int64')
    
    #Multivalued Sparse features
    feed_info = gen_Multivalued_Sparse_features(feed_info)
    feed_info['manual_keyword_embedding'] = gen_embedding_list(feed_info, 'manual_keyword_list', 3)
    feed_info['manual_tag_embedding'] = gen_embedding_list(feed_info, 'manual_tag_list', 3)
    
    #tag/keyword first stat
    feed_info['manual_tag_first'] = feed_info['manual_tag_list'].apply(lambda x: x[0]).astype(int)
    feed_info['machine_tag_first'] = feed_info['machine_tag_list'].apply(lambda x: x[0]).astype(int)
    
    #tag/keyword str2int
    feed_info['manual_keyword_list_int'] = feed_info['manual_keyword_list'].apply(lambda x:[int(i) + 1 for i in x])
    feed_info['machine_keyword_list_int'] = feed_info['machine_keyword_list'].apply(lambda x:[int(i) + 1 for i in x])
    feed_info['manual_tag_list_int'] = feed_info['manual_tag_list'].apply(lambda x:[int(i) + 1 for i in x])
    feed_info['machine_tag_list_int'] = feed_info['machine_tag_list'].apply(lambda x:[int(i) + 1 for i in x])
    
    #tag/keyword_int padding for VarLenSparseFeat
    pad = 4
    feed_info['manual_tag_list_int'] = feed_info['manual_tag_list_int'].apply(lambda x: x[:4] + [0] * (pad - len(x)))
    feed_info['machine_tag_list_int'] = feed_info['machine_tag_list_int'].apply(lambda x: x[:4] + [0] * (pad - len(x)))

    feed_info[['feedid','authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id',
               'manual_keyword_list_int', 'machine_keyword_list_int', 'manual_tag_list_int', 'machine_tag_list_int',
               'manual_keyword_embedding', 'manual_tag_embedding', 
               'manual_tag_first', 'machine_tag_first']].to_pickle(save_path)