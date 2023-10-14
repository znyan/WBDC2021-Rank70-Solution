# **2021中国高校计算机大赛-微信大数据挑战赛**
## 队伍名:Pogo
Rank：68（初赛）/70（复赛）/6768

## **1. 环境依赖**
- Python 3.6.5
- tensorflow-gpu==1.13.1
- pandas>=1.0.5
- numpy>=1.16.4
- numba>=0.45.1
- scipy>=1.3.1
- deepctr[gpu]==0.8.6
- catboost==0.26
- lightgbm==3.2.1
- gensim
- tqdm

## **2. 目录结构**

```
./
├── README.md
├── requirements.txt, python package requirements 
├── init.sh, script for installing package requirements
├── train.sh, script for preparing train/inference data and training models, including pretrained models
├── inference.sh, script for inference 
├── src
│   ├── prepare, codes for preparing train/inference dataset
|       ├──extract_history_feed_embeddings.py
|       ├──extract_feed_embeddings.py
|       ├──extract_feed_info_features.py
│   ├── model, codes for model architecture
|       ├──wide_and_deep.py  
|   ├── train, codes for training
|       ├──mmoe.py  
|   ├── inference.py, main function for inference on test dataset with mmoe model
|   ├── inference_lgb.py, main function for inference on test dataset with lightgbm model
|   ├── evaluation.py, main function for evaluation 
├── data
│   ├── wedata, dataset of the competition
│       ├── wechat_algo_data1, preliminary dataset
│       ├── wechat_algo_data2, semi dataset
│   ├── submission, prediction result after running inference.sh
│   ├── model, model files (e.g. tensorflow checkpoints)
```

## **3. 运行流程**
- 安装环境：bash init.sh
- 进入目录：cd /home/tione/notebook/wbdc2021-semi
- 数据准备和模型训练：bash train.sh
- 预测并生成结果文件：bash inference.sh ../wbdc2021/data/wedata/wechat_algo_data2/test_b.csv

## **4. 模型及特征**
- 模型：MMOE
- 参数：
    - batch_size: 8192
    - emded_dim: 192
    - expert_dim: 128
    - num_epochs: 7
    - learning_rate: 0.1
- 特征：
    - dnn 特征: userid, feedid, authorid, bgm_singer_id, bgm_song_id
    - linear 特征：videoplayseconds, feed_embedding, manual_tag_embedding, history_feed_embedding
    
## **5. 算法性能**
- 资源配置：2*P40_48G显存_14核CPU_112G内存
- 预测耗时  
    - 总预测时长: 478 s
    - 单个目标行为2000条样本的平均预测时长: 32.1186 ms


## **6. 代码说明**
模型预测部分代码位置如下：

| 路径 | 行数 | 内容 |
| :--- | :--- | :--- |
| src/inference.py | 139 | `pred_ans = train_model.predict(test_model_input, batch_size=batch_size, verbose=1)`|
