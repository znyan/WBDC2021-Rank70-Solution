import numpy as np
import pandas as pd
from numba import njit
from scipy.stats import rankdata
from joblib import Parallel, delayed


@njit
def _auc(actual, pred_ranks):
    actual = np.asarray(actual)
    pred_ranks = np.asarray(pred_ranks)
    n_pos = np.sum(actual)
    n_neg = len(actual) - n_pos
    return (np.sum(pred_ranks[actual == 1]) - n_pos*(n_pos+1)/2) / (n_pos*n_neg)


def auc(actual, predicted):
    pred_ranks = rankdata(predicted)
    return _auc(actual, pred_ranks)


def fast_uAUC(y_true, y_pred, userids):
    num_labels = y_pred.shape[1]

    def uAUC_infunc(i):
        uauc_df = pd.DataFrame()
        uauc_df['userid'] = userids
        uauc_df['y_true'] = y_true[:, i]
        uauc_df['y_pred'] = y_pred[:, i]


        label_nunique = uauc_df.groupby(by='userid')['y_true'].transform('nunique')
        uauc_df = uauc_df[label_nunique == 2]
        
        aucs = uauc_df.groupby(by='userid').apply(
            lambda x: auc(x['y_true'].values, x['y_pred'].values))
        return np.mean(aucs)

    uauc = Parallel(n_jobs=4)(delayed(uAUC_infunc)(i) for i in range(num_labels))
    return np.average(uauc, weights=[4, 3, 2, 1, 1, 1, 1]), uauc


# coding: utf-8
import time
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score

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


def compute_weighted_score(score_dict, weight_dict):
    # 基于多个行为的uAUC值，计算加权uAUC
    # Input:
    #     scores_dict: 多个行为的uAUC值映射字典, dict
    #     weights_dict: 多个行为的权重映射字典, dict
    # Output:
    #     score: 加权uAUC值, float
    score = 0.0
    weight_sum = 0.0
    for action in score_dict:
        weight = float(weight_dict[action])
        score += weight*score_dict[action]
        weight_sum += weight
    score /= float(weight_sum)
    score = round(score, 6)
    return score


def evaluate_deepctr(val_labels,val_pred_ans,userid_list,target):
    eval_dict = {}
    for i, action in enumerate(target):
        eval_dict[action] = uAUC(val_labels[i], val_pred_ans[i], userid_list)
    print(eval_dict)
    weight_dict = {"read_comment": 4, "like": 3, "click_avatar": 2, "favorite": 1, "forward": 1,
                   "comment": 1, "follow": 1}
    weight_auc = compute_weighted_score(eval_dict, weight_dict)
    print("Weighted uAUC: ", weight_auc)
    return weight_auc
'''
# coding: utf-8
import time
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score

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


def compute_weighted_score(score_dict, weight_dict):
    # 基于多个行为的uAUC值，计算加权uAUC
    # Input:
    #     scores_dict: 多个行为的uAUC值映射字典, dict
    #     weights_dict: 多个行为的权重映射字典, dict
    # Output:
    #     score: 加权uAUC值, float
    score = 0.0
    weight_sum = 0.0
    for action in score_dict:
        weight = float(weight_dict[action])
        score += weight*score_dict[action]
        weight_sum += weight
    score /= float(weight_sum)
    score = round(score, 6)
    return score


def evaluate_deepctr(val_labels,val_pred_ans,userid_list,target):
    eval_dict = {}
    for i, action in enumerate(target):
        eval_dict[action] = uAUC(val_labels[i], val_pred_ans[i], userid_list)
    print(eval_dict)
    weight_dict = {"read_comment": 4, "like": 3, "click_avatar": 2, "favorite": 1, "forward": 1,
                   "comment": 1, "follow": 1}
    weight_auc = compute_weighted_score(eval_dict, weight_dict)
    print("Weighted uAUC: ", weight_auc)
    return weight_auc
'''