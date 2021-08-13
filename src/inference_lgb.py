import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR,'../'))


user_action_path = os.path.join(ROOT_DIR,'data/wedata/wechat_algo_data2/user_action.csv')
feed_info_path = os.path.join(ROOT_DIR, 'data/wedata/wechat_algo_data2/feed_info.csv')
feed_embeddings_path = os.path.join(ROOT_DIR,'data/wedata/wechat_algo_data2/feed_embeddings.csv')
test_data_path = os.path.join(ROOT_DIR,'data/wedata/wechat_algo_data2/test_a.csv')

feed_info_frt_path = os.path.join(ROOT_DIR,'data/features/feed_info_frt.csv')
feed_embeddings_PCA_path = os.path.join(ROOT_DIR,'data/features/feed_embeddings_PCA.csv')
feed_info_manual_key_path = os.path.join(ROOT_DIR,'data/features/feed_info_manual_key.csv')
feed_info_manual_tag_path = os.path.join(ROOT_DIR,'data/features/feed_info_manual_key.csv')
feed_info_manual_tag_sta_path = os.path.join(ROOT_DIR,'data/features/feed_info_manual_tag_sta.csv')
# model_root_path = 'train'
submission_path = os.path.join(ROOT_DIR,'data/submission/result_lgb.csv')

import pandas as pd
import numpy as np
from copy import deepcopy
from gensim.models import word2vec
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from lightgbm.sklearn import LGBMClassifier
from catboost import CatBoostClassifier,Pool
from collections import defaultdict
import gc
import time
def trans_tag_key_emb(data,emb_size):
    def get_w2v(file,key):
        a=""
        for line in file[key]:
            line=str(line)
            if line!="nan":
                for one in line.split(";"):
                    a+=one+" "
        with open(str(key)+".txt","w") as f:
            f.write(a)
        sentences =word2vec.Text8Corpus(str(key)+".txt")  # 加载语料
        model =word2vec.Word2Vec(sentences, vector_size=emb_size, hs=1, min_count=1, window=3)  #训练skip-gram模型，默认window=5
        return model
    def split_col(data, column):
        data = deepcopy(data)
        max_len = max(list(map(len, data[column].values)))  # 最大长度
        new_col = data[column].apply(lambda x: x + [None]*(max_len - len(x)))  # 补空值，None可换成np.nan
        new_col = np.array(new_col.tolist()).T  # 转置
        for i, j in enumerate(new_col):
            data[column + str(i)] = j
        return data
    def trans1(s):
        s=str(s)
        res=np.array([0. for i in range(emb_size)])
        if s!="nan":
            for one in s.split(";"):
                res+=np.array(model1.wv[one])
        return list(res)
    def trans2(s):
        s=str(s)
        res=np.array([0. for i in range(emb_size)])
        if s!="nan":
            for one in s.split(";"):
                res+=np.array(model2.wv[one])
        return list(res)
    def trans3(s):
        s=str(s)
        res=np.array([0. for i in range(emb_size)])
        if s!="nan":
            for one in s.split(";"):
                res+=np.array(model3.wv[one])
        return list(res)
    def trans4(s):
        s=str(s)
        res=np.array([0. for i in range(emb_size)])
        if s!="nan":
            for one in s.split():
                res+=np.array(model4.wv[one])
        return list(res)
    model1=get_w2v(data,"manual_keyword_list")
    model2=get_w2v(data,"manual_tag_list")
    model3=get_w2v(data,"machine_keyword_list")
    model4=get_w2v(data,"description")
    data['manual_keyword_list_emb'] = data['manual_keyword_list'].apply(trans1)
    data['manual_tag_list_emb'] = data['manual_tag_list'].apply(trans2)
    data['machine_keyword_list_emb'] = data['machine_keyword_list'].apply(trans3)
    data['description_emb'] = data['description'].apply(trans4)
    data=split_col(data, column='manual_keyword_list_emb')
    data=split_col(data, column='manual_tag_list_emb')
    data=split_col(data, column='machine_keyword_list_emb')
    data=split_col(data, column='description_emb')
    return data

pd.set_option('display.max_columns', None)
def reduce_mem(df, cols):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in tqdm(cols):
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
## 从官方baseline里面抽出来的评测函数
def uAUC(labels, preds, user_id_list):
    """Calculate user AUC"""
    user_pred = defaultdict(lambda: [])
    user_truth = defaultdict(lambda: [])
    for idx, truth in enumerate(labels):
        user_id = user_id_list[idx]
        pred = preds[idx]
        truth = labels[idx]
        user_pred[user_id].append(pred)
        user_truth[user_id].append(truth)

    user_flag = defaultdict(lambda: False)
    for user_id in set(user_id_list):
        truths = user_truth[user_id]
        flag = False
        # 若全是正样本或全是负样本，则flag为False
        for i in range(len(truths) - 1):
            if truths[i] != truths[i + 1]:
                flag = True
                break
        user_flag[user_id] = flag
    total_auc = 0.0
    size = 0.0
    for user_id in user_flag:
        if user_flag[user_id]:
            auc = roc_auc_score(np.asarray(user_truth[user_id]), np.asarray(user_pred[user_id]))
            total_auc += auc
            size += 1.0
    user_auc = float(total_auc)/size
    return user_auc
def K_flod(df,n):
    a = df.shape[0]
    b = a//5
    print(b)
    df_1 = df[:b].reset_index(drop=True)
    df_2 = df[b:2*b].reset_index(drop=True)
    df_3 = df[2*b:3*b].reset_index(drop=True)
    df_4 = df[3*b:4*b].reset_index(drop=True)
    df_5 = df[4*b:].reset_index(drop=True)
    if(n == 1):
        return df_1
    elif(n == 2):
        return df_2
    elif(n == 3):
        return df_3
    elif(n == 4):
        return df_4
    else:
        return df_5
y_list = ['read_comment', 'like', 'click_avatar', 'forward', 'favorite', 'comment', 'follow']
max_day = 15
play_cols = [
    'is_finish', 'play_times', 'play', 'stay'
]
UM_EPOCH_DICT = {"read_comment": 0.2, "like": 0.2, "click_avatar": 0.2,"forward": 0.2,
                                "comment":0.2, "follow":0.2, "favorite": 0.2, }

def prepare_data(yy):

    ## 读取训练集

    train = pd.read_csv(user_action_path)#[:10000]
    train_neg = train[train[yy] == 0]
    train_neg = train_neg.sample(frac=UM_EPOCH_DICT[yy],random_state=13,replace=False)
    ## 增加五折验证
    # train_neg = K_flod(train_neg,n)
    train     = pd.concat([train_neg,train[train[yy] == 1]])
    print(train.shape)
    for y in y_list:
        print(y, train[y].mean())



    ## 读取测试集
    test = pd.read_csv(test_data_path)
    test['date_'] = max_day
    print(test.shape)
    ## 合并处理

    df = pd.concat([train, test], axis=0, ignore_index=True)
    print(df.head(3))

    ## 读取视频信息表

    feed_info = pd.read_csv(feed_info_path)

    ## 此份baseline只保留这三列

    emg_size = 16
    feed_info = trans_tag_key_emb(feed_info, emg_size)

    feed_info = feed_info[
        ['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id'] +
        ['manual_keyword_list_emb' + str(i) for i in range(emg_size)] +
        ['manual_tag_list_emb' + str(i) for i in range(emg_size)] +
        ['machine_keyword_list_emb' + str(i) for i in range(emg_size)] +
        ['description_emb' + str(i) for i in range(emg_size)]
        ]
    df = df.merge(feed_info, on='feedid', how='left')

    ## 从这里增加特征以及交叉特征
    #####################################################
    FEA_FEED_LIST = ['feedid', 'bgm_song_id', 'bgm_singer_id']
    FEED_INFO_FRT_LIST=['authorid_count', 'vPlaysecBucket_count', 'authorid_feedid_nunique',
            'authorid_vPlaysecBucket_nunique', 'vPlaysecBucket_feedid_nunique', 'vPlaysecBucket_authorid_nunique']
    FEA_FEED_LIST=FEA_FEED_LIST+FEED_INFO_FRT_LIST
    ## 32 64 128 256 对四个属性有不同影响
    feddemb = pd.read_csv(feed_embeddings_PCA_path)
    df = df.merge(feddemb,on='feedid',how='left')
    feed_info_frt = pd.read_csv(feed_info_frt_path)[FEA_FEED_LIST]
    df = df.merge(feed_info_frt,on='feedid',how='left')
    feed_info_manual_key = pd.read_csv(feed_info_manual_key_path)
    df = df.merge(feed_info_manual_key,on='feedid',how='left')
    feed_info_manual_tag = pd.read_csv(feed_info_manual_tag_sta_path)
    df = df.merge(feed_info_manual_tag,on='feedid',how='left')
    #### 加入下面的会波动
    # feed_info_machine_key = pd.read_csv('./data/feed_info_machine_key.csv')
    # df = df.merge(feed_info_machine_key,on='feedid',how='left')
    #####################################################
    ## 视频时长是秒，转换成毫秒，才能与play、stay做运算

    df['videoplayseconds'] *= 1000

    ## 是否观看完视频（其实不用严格按大于关系，也可以按比例，比如观看比例超过0.9就算看完）

    df['is_finish'] = (df['play'] >= df['videoplayseconds']).astype('int8')

    df['play_times'] = df['play'] / df['videoplayseconds']



    ## 统计历史5天的曝光、转化、视频观看等情况（此处的转化率统计其实就是target encoding）
    n_day = 14
    for stat_cols in tqdm([
        ['userid'],

        ['feedid'],

        ['authorid'],

        ['userid', 'authorid']

    ]):

        f = '_'.join(stat_cols)

        stat_df = pd.DataFrame()

        for target_day in range(2, max_day + 1):

            left, right = max(target_day - n_day, 1), target_day - 1
            tmp = df[((df['date_'] >= left) & (df['date_'] <= right))].reset_index(drop=True)
            tmp['date_'] = target_day
            tmp['{}_{}day_count'.format(f, n_day)] = tmp.groupby(stat_cols)['date_'].transform('count')

            g = tmp.groupby(stat_cols)

            tmp['{}_{}day_finish_rate'.format(f, n_day)] = g[play_cols[0]].transform('mean')
            feats = ['{}_{}day_count'.format(f, n_day), '{}_{}day_finish_rate'.format(f, n_day)]
            for x in play_cols[1:]:
                for stat in ['max', 'mean']:
                    tmp['{}_{}day_{}_{}'.format(f, n_day, x, stat)] = g[x].transform(stat)
                    feats.append('{}_{}day_{}_{}'.format(f, n_day, x, stat))



            for y in y_list:

                tmp['{}_{}day_{}_sum'.format(f, n_day, y)] = g[y].transform('sum')

                tmp['{}_{}day_{}_mean'.format(f, n_day, y)] = g[y].transform('mean')

                feats.extend(['{}_{}day_{}_sum'.format(f, n_day, y), '{}_{}day_{}_mean'.format(f, n_day, y)])

            tmp = tmp[stat_cols + feats + ['date_']].drop_duplicates(stat_cols + ['date_']).reset_index(drop=True)
            stat_df = pd.concat([stat_df, tmp], axis=0, ignore_index=True)
            del g, tmp

        df = df.merge(stat_df, on=stat_cols + ['date_'], how='left')
        del stat_df
        gc.collect()


    ## 全局信息统计，包括曝光、偏好等，略有穿越，但问题不大，可以上分，只要注意不要对userid-feedid做组合统计就行

    for f in tqdm(['userid', 'feedid', 'authorid']):

        df[f + '_count'] = df[f].map(df[f].value_counts())

    for f1, f2 in tqdm([

        ['userid', 'feedid'],

        ['userid', 'authorid']

    ]):

        df['{}_in_{}_nunique'.format(f1, f2)] = df.groupby(f2)[f1].transform('nunique')

        df['{}_in_{}_nunique'.format(f2, f1)] = df.groupby(f1)[f2].transform('nunique')

    for f1, f2 in tqdm([

        ['userid', 'authorid']

    ]):

        df['{}_{}_count'.format(f1, f2)] = df.groupby([f1, f2])['date_'].transform('count')

        df['{}_in_{}_count_prop'.format(f1, f2)] = df['{}_{}_count'.format(f1, f2)] / (df[f2 + '_count'] + 1)

        df['{}_in_{}_count_prop'.format(f2, f1)] = df['{}_{}_count'.format(f1, f2)] / (df[f1 + '_count'] + 1)

    df['videoplayseconds_in_userid_mean'] = df.groupby('userid')['videoplayseconds'].transform('mean')

    df['videoplayseconds_in_authorid_mean'] = df.groupby('authorid')['videoplayseconds'].transform('mean')

    df['feedid_in_authorid_nunique'] = df.groupby('authorid')['feedid'].transform('nunique')



    ## 内存够用的不需要做这一步

    df = reduce_mem(df, [f for f in df.columns if f not in ['date_'] + play_cols + y_list])
    # df.to_csv('data_df.csv',index=False)
    return df
def predict(df,y):
    # df = pd.read_csv('data_df.csv')
    train = df[~df['read_comment'].isna()].reset_index(drop=True)

    test = df[df['read_comment'].isna()].reset_index(drop=True)

    cols = [f for f in df.columns if f not in ['date_'] + play_cols + y_list]

    print(train[cols].shape)
    
    ########################### 预测 ########################

#     r_dict = dict(zip(y_list, r_list))

    # for y in y_list[:4]:
    for yy in range(1):
        print('=========', y, '=========')

        t = time.time()

        clf = CatBoostClassifier(loss_function="Logloss",
                               eval_metric="AUC",
                               task_type="CPU",
                               learning_rate=0.01,
                               iterations=5000,
                               random_seed=42,
                               od_type="Iter",
                               depth=10
                               )
        clf.load_model(os.path.join(ROOT_DIR,'data/model/' + 'model_' + str(y) + '.txt'))
        test[y] = clf.predict_proba(test[cols])[:, 1]

        print('runtime: {}\n'.format(time.time() - t))

    return test[y]


if __name__ == '__main__':

    blank = pd.read_csv(test_data_path)
    for y in y_list:

        df = prepare_data(y)
        blank[y] =predict(df,y)
    blank.to_csv(submission_path, index=False)