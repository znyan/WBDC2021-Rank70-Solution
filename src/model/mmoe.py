 # -*- coding:utf-8 -*-

import tensorflow as tf

from deepctr.feature_column import build_input_features, input_from_feature_columns,get_linear_logit,DEFAULT_GROUP_NAME
from deepctr.layers.utils import combined_dnn_input, concat_func, add_func
from deepctr.layers.core import PredictionLayer, DNN
from deepctr.layers.interaction import FM,CIN
from tensorflow.python.keras.initializers import glorot_normal
from tensorflow.python.keras.layers import Layer

from itertools import chain



class MMOELayer(Layer):
    def __init__(self, num_tasks, num_experts, output_dim, seed=1024, **kwargs):
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.output_dim = output_dim
        self.seed = seed
        super(MMOELayer, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        self.expert_kernel = self.add_weight( #shape=(128, 32) dtype=float32>
            name='expert_kernel',
            shape=(input_dim, self.num_experts * self.output_dim),
            dtype=tf.float32,
            initializer=glorot_normal(seed=self.seed))
        self.gate_kernels = []
        for i in range(self.num_tasks):
            self.gate_kernels.append(self.add_weight(
                name='gate_weight_'.format(i),
                shape=(input_dim, self.num_experts),   #128,8
                dtype=tf.float32,
                initializer=glorot_normal(seed=self.seed)))
        super(MMOELayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        outputs = []
        expert_out = tf.tensordot(inputs, self.expert_kernel, axes=(-1, 0)) #?,32
        expert_out = tf.reshape(expert_out, [-1, self.output_dim, self.num_experts])  #?, 8, 4
        for i in range(self.num_tasks):
            gate_out = tf.tensordot(inputs, self.gate_kernels[i], axes=(-1, 0)) #gate_weight_:0' shape=(128, 4)       
            gate_out = tf.nn.softmax(gate_out)
            gate_out = tf.tile(tf.expand_dims(gate_out, axis=1), [1, self.output_dim, 1])
            output = tf.reduce_sum(tf.multiply(expert_out, gate_out), axis=2)  #  8
            outputs.append(output)
        return outputs

    def get_config(self):

        config = {'num_tasks': self.num_tasks,
                  'num_experts': self.num_experts,
                  'output_dim': self.output_dim}
        base_config = super(MMOELayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return [input_shape[0], self.output_dim] * self.num_tasks


def MMOE(dnn_feature_columns, num_tasks, tasks, num_experts=4, expert_dim=8, dnn_hidden_units=(128, 128),
         l2_reg_embedding=1e-5, l2_reg_dnn=0, task_dnn_units=None, seed=1024, dnn_dropout=0, dnn_activation='relu'):
    if num_tasks <= 1:
        raise ValueError("num_tasks must be greater than 1")   
    if len(tasks) != num_tasks:
        raise ValueError("num_tasks must be equal to the length of tasks")
    for task in tasks:
        if task not in ['binary', 'regression']:
            raise ValueError("task must be binary or regression, {} is illegal".format(task))
 
    features = build_input_features(dnn_feature_columns)

    inputs_list = list(features.values())

    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
                                                                         l2_reg_embedding, seed)
    dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
    dnn_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                  False, seed=seed)(dnn_input)
    mmoe_outs = MMOELayer(num_tasks, num_experts, expert_dim)(dnn_out)
    if task_dnn_units != None:
        mmoe_outs = [DNN(task_dnn_units, dnn_activation, l2_reg_dnn, dnn_dropout, False, seed=seed)(mmoe_out) for mmoe_out in
                     mmoe_outs]

    task_outputs = []
    for mmoe_out, task in zip(mmoe_outs, tasks):
        logit = tf.keras.layers.Dense(
            1, use_bias=False, activation=None)(mmoe_out)
        output = PredictionLayer(task)(logit)
        task_outputs.append(output)

    model = tf.keras.models.Model(inputs=inputs_list,
                                  outputs=task_outputs)
    return model


def DeepFM_mmoe(linear_feature_columns,dnn_feature_columns,num_tasks, tasks, num_experts=4, expert_dim=8, task_dnn_units=None, fm_group=[DEFAULT_GROUP_NAME], dnn_hidden_units=(128, 128),l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, seed=1024, dnn_dropout=0,
           dnn_activation='relu', dnn_use_bn=False, task='binary'):
    """Instantiates the DeepFM Network architecture.
    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param fm_group: list, group_name of features that will be used to do feature interactions.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.
    """

    if num_tasks <= 1:
        raise ValueError("num_tasks must be greater than 1")
    if len(tasks) != num_tasks:
        raise ValueError("num_tasks must be equal to the length of tasks")
    for task in tasks:
        if task not in ['binary', 'regression']:
            raise ValueError("task must be binary or regression, {} is illegal".format(task))

    features = build_input_features(linear_feature_columns+dnn_feature_columns)

    inputs_list = list(features.values())

    linear_logit = get_linear_logit(features, linear_feature_columns, seed=seed, prefix='linear',
                                    l2_reg=l2_reg_linear)


    #sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,l2_reg_embedding, seed)
    group_embedding_dict, dense_value_list = input_from_feature_columns(features, dnn_feature_columns, l2_reg_embedding,
                                                                        seed, support_group=True)
    fm_logit = add_func([FM()(concat_func(v, axis=1))
                         for k, v in group_embedding_dict.items() if k in fm_group])


    dnn_input = combined_dnn_input(list(chain.from_iterable(
        group_embedding_dict.values())), dense_value_list)
    dnn_output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed)(dnn_input)



    mmoe_outs = MMOELayer(num_tasks, num_experts, expert_dim)(dnn_output)
    if task_dnn_units != None:
        mmoe_outs = [DNN(task_dnn_units, dnn_activation, l2_reg_dnn, dnn_dropout, False, seed=seed)(mmoe_out) for mmoe_out in
                     mmoe_outs]

    task_outputs = []
    for mmoe_out, task in zip(mmoe_outs, tasks):
        dnn_logit = tf.keras.layers.Dense(
            1, use_bias=False, activation=None)(mmoe_out)
        final_logit = add_func([linear_logit, fm_logit, dnn_logit])
        output = PredictionLayer(task)(final_logit)
        task_outputs.append(output)

    model = tf.keras.models.Model(inputs=inputs_list,
                                  outputs=task_outputs)
    return model


def xDeepFM_mmoe(linear_feature_columns,dnn_feature_columns,num_tasks, tasks, num_experts=4, expert_dim=8, task_dnn_units=None, fm_group=[DEFAULT_GROUP_NAME], dnn_hidden_units=(128, 128),l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, seed=1024, dnn_dropout=0,
           dnn_activation='relu', l2_reg_cin=0,dnn_use_bn=False, task='binary',cin_layer_size=(128, 128,), cin_split_half=True, cin_activation='relu'):


    if num_tasks <= 1:
        raise ValueError("num_tasks must be greater than 1")
    if len(tasks) != num_tasks:
        raise ValueError("num_tasks must be equal to the length of tasks")
    for task in tasks:
        if task not in ['binary', 'regression']:
            raise ValueError("task must be binary or regression, {} is illegal".format(task))

    features = build_input_features(
        linear_feature_columns + dnn_feature_columns)

    inputs_list = list(features.values())

    linear_logit = get_linear_logit(features, linear_feature_columns, seed=seed, prefix='linear',
                                    l2_reg=l2_reg_linear)

    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
                                                                         l2_reg_embedding, seed)

    fm_input = concat_func(sparse_embedding_list, axis=0)

    dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)

    dnn_output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed)(dnn_input)

    mmoe_outs = MMOELayer(num_tasks, num_experts, expert_dim)(dnn_output)
    if task_dnn_units != None:
        mmoe_outs = [DNN(task_dnn_units, dnn_activation, l2_reg_dnn, dnn_dropout, False, seed=seed)(mmoe_out) for mmoe_out in
                     mmoe_outs]

    task_outputs = []
    for mmoe_out, task in zip(mmoe_outs, tasks):
        dnn_logit = tf.keras.layers.Dense(
            1, use_bias=False, activation=None)(mmoe_out)
        final_logit = add_func([linear_logit, dnn_logit])
        if len(cin_layer_size) > 0:
            exFM_out = CIN(cin_layer_size, cin_activation,
                           cin_split_half, l2_reg_cin, seed)(fm_input)
            exFM_logit = tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.glorot_normal(seed))(exFM_out)
            final_logit = add_func([final_logit, exFM_logit])

        output = PredictionLayer(task)(final_logit)
        task_outputs.append(output)

    model = tf.keras.models.Model(inputs=inputs_list,
                                  outputs=task_outputs)

    return model