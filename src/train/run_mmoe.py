import warnings
warnings.filterwarnings("ignore")
import ipykernel
import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm, trange
import tensorflow as tf
from tensorflow import feature_column as fc
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.optimizers import Adam,Adagrad
from time import time
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names, VarLenSparseFeat
import random
from tensorflow.python.keras.models import save_model,load_model
from deepctr.layers import custom_objects
from tensorflow.python.keras.utils import multi_gpu_model
import gc

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR,'../../'))
sys.path.append(os.path.join(BASE_DIR,".."))

from model.mmoe import MMOE,MMOELayer,xDeepFM_mmoe,DeepFM_mmoe
from evaluation import evaluate_deepctr, fast_uAUC


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
    # os.environ['TF_DETERMINISTIC_OPS'] = '1'
    # GPU相关设置
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # 设置GPU按需增长
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)




def read_data():
    user_action_preliminary_path = os.path.join(ROOT_DIR, 'data/wedata/wechat_algo_data1/user_action.csv')
    user_action_path = os.path.join(ROOT_DIR,'data/wedata/wechat_algo_data2/user_action.csv')
    feed_info_path = os.path.join(ROOT_DIR,'data/features/feed_info.pkl')
    feed_embeddings_path = os.path.join(ROOT_DIR,'data/features/feed_embeddings_origin.pkl')
    history_feed_embeddings_path = os.path.join(ROOT_DIR, 'data/features/history_feed_embeddings.pkl')
    
    user_action = pd.read_csv(user_action_preliminary_path).append(pd.read_csv(user_action_path),ignore_index=True)
    user_acrtion = user_action.sample(frac=0.01)
    feed_info = pd.read_pickle(feed_info_path)
    feed_embeddings = pd.read_pickle(feed_embeddings_path)
    history_feed_embeddings = pd.read_pickle(history_feed_embeddings_path)
    
    feed_info = feed_info.merge(feed_embeddings, on='feedid', how='left')
    
    # data = user_action.merge(feed_info, on='feedid', how='left')
    # data = data.merge(history_feed_embeddings, on=['userid', 'date_'], how='left')
    
    # tqdm.pandas(desc="avg_emb_rolling fillna")
    # data['avg_emb_rolling'] = data['avg_emb_rolling'].progress_apply(lambda x:list([0.0] * 48) if type(x) != list else list(x))
    
    return user_action, feed_info, history_feed_embeddings

def read_data_reduce_mem():
    user_action_preliminary_path = os.path.join(ROOT_DIR, 'data/wedata/wechat_algo_data1/user_action.csv')
    user_action_path = os.path.join(ROOT_DIR,'data/wedata/wechat_algo_data2/user_action.csv')
    feed_info_path = os.path.join(ROOT_DIR,'data/features/feed_info.pkl')
    feed_embeddings_path = os.path.join(ROOT_DIR,'data/features/feed_embeddings_origin.pkl')
    history_feed_embeddings_path = os.path.join(ROOT_DIR, 'data/features/history_feed_embeddings.pkl')
    
    feed_info = pd.read_pickle(feed_info_path)
    feed_embeddings = pd.read_pickle(feed_embeddings_path)
    history_feed_embeddings = pd.read_pickle(history_feed_embeddings_path)
    feed_info = feed_info.merge(feed_embeddings, on='feedid', how='left')
    
    
    data_list = []
        
    for df in tqdm(pd.read_csv(user_action_preliminary_path, chunksize=1000000)):
        df = df.merge(feed_info, on='feedid', how='left')
        df = df.merge(history_feed_embeddings, on=['userid', 'date_'], how='left')

        data_list.append(df)
    
    for df in tqdm(pd.read_csv(user_action_path, chunksize=1000000)):
        df = df.merge(feed_info, on='feedid', how='left')
        df = df.merge(history_feed_embeddings, on=['userid', 'date_'], how='left')

        data_list.append(df)

    data = pd.concat(data_list, ignore_index=True)
    
    
    tqdm.pandas(desc="avg_emb_rolling fillna")
    data['avg_emb_rolling'] = data['avg_emb_rolling'].progress_apply(lambda x:list([0.0] * 48) if type(x) != list else list(x))
    
    return data


user_action, feed_info, history_feed_embeddings = read_data()
print('dataset size',len(user_action))



target = ['read_comment', 'like', 'click_avatar', 'forward', 'favorite', 'comment', 'follow']

drop_features = ['date_', 'device', 'play', 'stay', 'pause', 'play_ratio',
                 'read_comment','comment','like','click_avatar','forward','follow','favorite']
dense_features = ['videoplayseconds',]     
sparse_features = ['userid', 'feedid', 'device', 'bgm_song_id', 'bgm_singer_id', 'authorid',]
varlen_features = ['manual_keyword_list','manual_tag_list','machine_keyword_list','machine_tag_list']



def train_data_generator(data_user_action, batch_size, f, t):
    count = 1
    while 1:
        data_batch = data_user_action.loc[(count - 1) * batch_size: count * batch_size]
        data_batch = data_batch.merge(feed_info, on='feedid', how='left')
        #data_batch = data_batch.merge(history_feed_embeddings, on=['userid', 'date_'], how='left')
        #data_batch['avg_emb_rolling'] = data_batch['avg_emb_rolling'].apply(lambda x:list([0.0] * 48) if type(x) != list else list(x))
        
        train_model_input = {name: data_batch[name] for name in f}
        
        train_model_input['manual_keyword_embedding'] = np.array(list(data_batch['manual_keyword_embedding']))
        train_model_input['manual_tag_embedding'] = np.array(list(data_batch['manual_tag_embedding']))
        train_model_input['feed_embedding'] = np.array(list(data_batch['feed_embedding']))
        train_model_input['avg_emb_rolling'] = np.array(list(data_batch['avg_emb_rolling']))
        train_model_input['manual_tag_list_int'] = np.array(list(data_batch['manual_tag_list_int']))
        train_model_input['machine_tag_list_int'] = np.array(list(data_batch['machine_tag_list_int']))

        train_labels = [data_batch[y].values for y in t]
        
        count = count + 1
        yield (train_model_input, train_labels)
        
def valid_data_generator(data_user_action, batch_size, f, t):
    count = 1
    while 1:
        data_batch = data_user_action.loc[(count - 1) * batch_size: count * batch_size]
        data_batch = data_batch.merge(feed_info, on='feedid', how='left')

        valid_model_input = {name: data_batch[name] for name in f}
        
        # train_model_input['manual_keyword_embedding'] = np.array(list(data_batch['manual_keyword_embedding']))
        # train_model_input['manual_tag_embedding'] = np.array(list(data_batch['manual_tag_embedding']))
        valid_model_input['feed_embedding'] = np.array(list(data_batch['feed_embedding']))
        # train_model_input['avg_emb_rolling'] = np.array(list(data_batch['avg_emb_rolling']))
        # train_model_input['manual_tag_list_int'] = np.array(list(data_batch['manual_tag_list_int']))
        # train_model_input['machine_tag_list_int'] = np.array(list(data_batch['machine_tag_list_int']))

        valid_labels = [data_batch[y].values for y in t]
        
        count = count + 1
        yield valid_model_input

# train_valid_split
train_user_action = user_action[user_action['date_'] <= 14] # 全量/线下测试
train_user_action = train_user_action.merge(history_feed_embeddings, on=['userid', 'date_'], how='left')
tqdm.pandas(desc="train avg_emb_rolling fillna")
train_user_action['avg_emb_rolling'] = train_user_action['avg_emb_rolling'].progress_apply(lambda x:list([0.0] * 48) if type(x) != list else list(x))
train_user_action.reset_index(inplace=True)

valid_user_action = user_action[user_action['date_'] == 14].sample(frac=0.25) # 第14天样本作为验证集
valid_data = valid_user_action.merge(feed_info, on='feedid', how='left')
valid_data = valid_data.merge(history_feed_embeddings, on=['userid', 'date_'], how='left')
tqdm.pandas(desc="valid avg_emb_rolling fillna")
valid_data['avg_emb_rolling'] = valid_data['avg_emb_rolling'].progress_apply(lambda x:list([0.0] * 48) if type(x) != list else list(x))
valid_data.reset_index(inplace=True)

def run_training(SEED): 
    EPOCHS = 7
    BATCH_SIZE = 8192
    EXPERT_DIM = 128
    EMBEDDING_DIM = 192
    
    # 2.count unique features for each sparse field,and record dense feature field name
    #pretrained_feed_emb_weights = np.array(list(feed_embedding))
    #pretrained_weights_initializer = tf.initializers.identity(pretrained_feed_emb_weights)
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
    # save sparse features vocabulary size to csv
    # pd.DataFrame({feat.name: feat.vocabulary_size for feat in fixlen_feature_columns if feat.name in sparse_features}, index=[0]).to_csv(
    #     os.path.join(ROOT_DIR,'data/features/vocabulary_size_df.csv'), index=False)

    # dnn_feature_columns = fixlen_feature_columns
    # feature_names = get_feature_names(dnn_feature_columns)
    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # 3.generate input training_set for model
    valid_model_input = {name: valid_data[name] for name in feature_names}
    valid_model_input['manual_keyword_embedding'] = np.array(list(valid_data['manual_keyword_embedding']))
    valid_model_input['manual_tag_embedding'] = np.array(list(valid_data['manual_tag_embedding']))
    valid_model_input['feed_embedding'] = np.array(list(valid_data['feed_embedding']))
    valid_model_input['avg_emb_rolling'] = np.array(list(valid_data['avg_emb_rolling']))
    valid_model_input['manual_tag_list_int'] = np.array(list(valid_data['manual_tag_list_int']))
    valid_model_input['machine_tag_list_int'] = np.array(list(valid_data['machine_tag_list_int']))
    
    
    # 4.Define Model,train,predict and evaliduate
    seed_everything(seed=SEED)
    train_model = MMOE(dnn_feature_columns, 
                               num_tasks=len(target), 
                               expert_dim=EXPERT_DIM,
                               tasks=['binary', 'binary', 'binary', 'binary', 'binary', 'binary', 'binary'],
                               dnn_hidden_units=(512, 512), 
                               seed=SEED)
    train_model.compile("adagrad", 
                        loss='binary_crossentropy',  
                        loss_weights=[4, 3, 2, 1, 1, 1, 1],)
    
    weight_auc_list = []
    for epoch in range(EPOCHS):
        # train
        history = train_model.fit_generator(train_data_generator(train_user_action, BATCH_SIZE, feature_names, target),
                                            steps_per_epoch=int(len(train_user_action) / BATCH_SIZE) + 1, 
                                            epochs=1, 
                                            verbose=1, )
        # train_model.save_weights(os.path.join(ROOT_DIR,'data/model/MMOE_w_seed' + str(SEED) + '_epoch' + str(epoch) +'.h5'))
        
        # evaluate
        weight_auc = evaluate_deepctr([valid_data[y].values for y in target],
                                      train_model.predict(valid_model_input, batch_size=BATCH_SIZE * 4, verbose=1),
                                      valid_user_action['userid'].astype(str).tolist(),
                                      target)
                
        # weight_auc_list.append(weight_auc)
        # pd.DataFrame({'weight_auc':weight_auc_list}).to_csv(os.path.join(ROOT_DIR,'data/features/weight_aucs_' + str(SEED) + '.csv'))
    
    train_model.save_weights(os.path.join(ROOT_DIR,'data/model/MMOE_w_' + str(SEED) + '.h5'))


run_training(42)
run_training(2021)