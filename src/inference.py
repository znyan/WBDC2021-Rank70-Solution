import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR,'../'))

import pandas as pd
import numpy as np
from tqdm import tqdm, trange
import tensorflow as tf
from tensorflow import feature_column as fc
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import time
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names, VarLenSparseFeat
from tensorflow.python.keras.utils import multi_gpu_model
import random
from tensorflow.python.keras.models import save_model,load_model
from deepctr.layers import custom_objects
import gc

from model.mmoe import MMOE,MMOELayer
from evaluation import evaluate_deepctr



def del_all_flags(FLAGS): 
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list: 
        FLAGS.__delattr__(keys) 
        
flags = tf.app.flags
FLAGS = flags.FLAGS

del_all_flags(FLAGS)

tf.app.flags.DEFINE_string('f', '', 'kernel') 
flags.DEFINE_float('embed_l2', None, 'embedding l2 reg')
flags.DEFINE_integer('embed_dim', 16, 'embed_dim')



def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    # GPU相关设置
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # 设置GPU按需增长
    config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)


def read_data(test_path):
    user_action_path = test_path
    feed_info_path = os.path.join(ROOT_DIR,'data/features/feed_info.pkl')
    feed_embeddings_path = os.path.join(ROOT_DIR,'data/features/feed_embeddings_origin.pkl')
    history_feed_embeddings_path = os.path.join(ROOT_DIR, 'data/features/history_feed_embeddings.pkl')
    
    user_action = pd.read_csv(user_action_path)
    feed_info = pd.read_pickle(feed_info_path)
    feed_embeddings = pd.read_pickle(feed_embeddings_path)
    history_feed_embeddings = pd.read_pickle(history_feed_embeddings_path)
    history_feed_embeddings = history_feed_embeddings[history_feed_embeddings['date_'] == 15]
    
    feed_info = feed_info.merge(feed_embeddings, on='feedid', how='left')
    
    data = user_action.merge(feed_info, on='feedid', how='left')
    data = data.merge(history_feed_embeddings, on='userid',how='left')
    data['avg_emb_rolling'] = data['avg_emb_rolling'].apply(lambda x:list([0.0] * 48) if type(x) != list else list(x))
    
    return data


target = ['read_comment', 'like', 'click_avatar', 'forward', 'favorite', 'comment', 'follow']

drop_features = ['date_', 'device', 'play', 'stay', 'pause', 'play_ratio'] + ['read_comment','comment','like','click_avatar','forward','follow','favorite']
dense_features = ['videoplayseconds',]     
sparse_features = ['userid', 'feedid', 'device', 'bgm_song_id', 'bgm_singer_id', 'authorid',]





loss_weights = [4, 3, 2, 1, 1, 1, 1]
# In[6]:


def predict(SEED, test):    
    batch_size = 10240
    # embedding_dim = 128
    expert_dim = 128
    EMBEDDING_DIM = 192
    # 2.count unique features for each sparse field,and record dense feature field name
    fixlen_feature_columns = [SparseFeat('userid', 250249, EMBEDDING_DIM),
                              SparseFeat('feedid', 112872, EMBEDDING_DIM),
                              SparseFeat('device', 3, 3),
                              SparseFeat('bgm_song_id', 25160, EMBEDDING_DIM),
                              SparseFeat('bgm_singer_id', 17501, EMBEDDING_DIM),
                              SparseFeat('authorid', 18789, EMBEDDING_DIM),
                              DenseFeat('videoplayseconds', 1),
                              DenseFeat('manual_keyword_embedding', 3,), 
                              DenseFeat('manual_tag_embedding', 3,), 
                              DenseFeat('feed_embedding', 512,),
                              DenseFeat('avg_emb_rolling', 48),
                              VarLenSparseFeat(SparseFeat('manual_tag_list_int', vocabulary_size=353+1,embedding_dim=8,
                                                          embedding_name='manual_tag_list_int'), maxlen=4),
                              VarLenSparseFeat(SparseFeat('machine_tag_list_int',vocabulary_size=346+1,embedding_dim=8,
                                                          embedding_name='machine_tag_list_int'), maxlen=4)
                              ] 
    
    dnn_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(dnn_feature_columns)
    
    # 3.generate input training_set for model
    test_model_input = {name: test[name] for name in feature_names}

        
    test_model_input['manual_keyword_embedding'] = np.array(list(test['manual_keyword_embedding']))
    test_model_input['manual_tag_embedding'] = np.array(list(test['manual_tag_embedding']))
    test_model_input['feed_embedding'] = np.array(list(test['feed_embedding']))
    test_model_input['avg_emb_rolling'] = np.array(list(test['avg_emb_rolling']))
    test_model_input['manual_tag_list_int'] = np.array(list(test['manual_tag_list_int']))
    test_model_input['machine_tag_list_int'] = np.array(list(test['machine_tag_list_int']))


    train_model = MMOE(dnn_feature_columns, num_tasks=7, expert_dim=expert_dim, dnn_hidden_units=(512, 512),
                       tasks=['binary', 'binary', 'binary', 'binary', 'binary', 'binary', 'binary'], seed=SEED)
    # train_model = multi_gpu_model(train_model, gpus=2)
    seed_everything(seed=SEED)
    
    train_model.compile("adagrad", loss='binary_crossentropy')
    train_model.load_weights(os.path.join(ROOT_DIR,'data/model/MMOE_w_' + str(SEED) + '.h5'))
    
    pred_ans = train_model.predict(test_model_input, batch_size=batch_size, verbose=1)

    # 5.生成提交文件
    for i, action in enumerate(target):
        test[action] = pred_ans[i]
        
    return test[['userid', 'feedid'] + target]

def main(argv):
    t = time.time() 
    stage = argv[1]
    print('Stage: %s'%stage)
    test_path = ''
    if len(argv)==3:
        test_path = argv[2]
        t1 = time.time()
        test_input = read_data(test_path)
        print('Get test input cost: %.4f s'%(time.time()-t1))
    
    # vocabulary_size_df = pd.read_csv(os.path.join(ROOT_DIR,'data/features/vocabulary_size_df.csv'))
    
    sub_1 = predict(42, test_input)
    sub_2 = predict(2021, test_input)
    
    sub = pd.DataFrame()
    sub['userid'] = sub_1['userid']
    sub['feedid'] = sub_1['feedid']
    for t in target:
        sub[t] = (sub_1[t] + sub_2[t]) / 2
    
    sub.to_csv(os.path.join(ROOT_DIR,'data/submission/result.csv'), index=False)

if __name__ == "__main__":
    tf.app.run(main)