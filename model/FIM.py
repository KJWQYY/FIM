'''
implementation of FIM
'''

import math
import os, sys
import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from sklearn.metrics import accuracy_score
from time import time
import argparse
from dataloader import load_MM_Douban as DATA
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm


import json
import csv
firstWrite = 1
#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run.")
    parser.add_argument('--process', nargs='?', default='train',
                        help='Process type: train, evaluate.')
    parser.add_argument('--mla', type=int, default=0,
                        help='Set the experiment mode to be Micro Level Analysis or not: 0-disable, 1-enable.')
    parser.add_argument('--path', nargs='?', default='../data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='douban',
                        help='Choose a dataset.')
    parser.add_argument('--valid_dimen', type=int, default=3,
                        help='Valid dimension of the dataset. (e.g. frappe=10, ml-tag=3)')
    parser.add_argument('--epoch', type=int, default=1000,
                        help='Number of epochs.')
    parser.add_argument('--pretrain', type=int, default=-1,
                        help='flag for pretrain. 1: initialize from pretrain; 0: randomly initialize; -1: save to pretrain file; 2: initialize from pretrain and save to pretrain file')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size.')
    parser.add_argument('--attention', type=int, default=0,
                        help='flag for attention. 1: use attention; 0: no attention')
    # parser.add_argument('--hidden_factor', nargs='?', default='[36,36]',
    #                     help='Number of hidden factors.')
    parser.add_argument('--hidden_factor', nargs='?', default='[1,1]',
                        help='Number of hidden factors.')
    # parser.add_argument('--lamda_attention', type=float, default=1e+2,
    #                     help='Regularizer for attention part.')
    parser.add_argument('--lamda', type=float, default=0.01,
                        help='Regularizer for attention part.')
    parser.add_argument('--keep', nargs='?', default=1,
                        help='Keep probility (1-dropout) of each layer. 1: no dropout. The first index is for the attention-aware pairwise interaction layer.')
    # parser.add_argument('--keep', nargs='?', default='[1.0,0.5]',
    #                     help='Keep probility (1-dropout) of each layer. 1: no dropout. The first index is for the attention-aware pairwise interaction layer.')

    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate.')
    parser.add_argument('--freeze_fm', type=int, default=0,
                        help='Freese all params of fm and learn attention params only.')
    parser.add_argument('--optimizer', nargs='?', default='AdagradOptimizer',
                        help='Specify an optimizer type (AdamOptimizer, AdagradOptimizer, GradientDescentOptimizer, MomentumOptimizer).')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Whether to show the performance of each epoch (0 or 1)')
    parser.add_argument('--batch_norm', type=int, default=0,
                    help='Whether to perform batch normaization (0 or 1)')
    parser.add_argument('--decay', type=float, default=0.999,
                    help='Decay value for batch norm')
    parser.add_argument('--activation', nargs='?', default='relu',
                    help='Which activation function to use for deep layers: relu, sigmoid, tanh, identity')
    parser.add_argument('--init_seed', type=int, default=104,
                        help='init_seed')
    parser.add_argument('--std', type=float, default=0.1,
                        help='std in init')
    parser.add_argument('--trpanum', type=int, default=18,
                        help='fixseed')
    return parser.parse_args()

class FIM(BaseEstimator, TransformerMixin):
    def __init__(self, features_M, fields_M, pretrain_flag, save_file, attention, hidden_factor, valid_dimension, activation_function, num_variable,
                 freeze_fm, epoch, batch_size, learning_rate, lamda_bilinear, keep, optimizer_type, batch_norm, decay, verbose, micro_level_analysis, init_seed, std, r_1, trpanum, random_seed=1024):
        # bind params to class
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.attention = attention
        self. hidden_factor = hidden_factor
        self.valid_dimension = valid_dimension
        self.activation_function = activation_function
        self.num_variable = num_variable
        self.save_file = save_file
        self.pretrain_flag = pretrain_flag
        self.features_M = features_M
        self.fields_M = fields_M
        self.lamda_bilinear = lamda_bilinear
        self.keep = keep
        self.freeze_fm = freeze_fm
        self.epoch = epoch
        self.random_seed = random_seed
        self.init_seed = init_seed
        self.std = std
        self.trpanum = trpanum
        self.optimizer_type = optimizer_type
        self.batch_norm = batch_norm
        self.decay = decay
        self.verbose = verbose
        self.micro_level_analysis = micro_level_analysis
        # performance of each epoch
        self.train_mse, self.valid_mse, self.test_mse = [], [], []
        self.train_mae, self.valid_mae, self.test_mae = [], [], []
        self.r_1 = r_1
        # init all variables in a tensorflow graph
        self._init_graph()

    def _init_graph(self):
        '''
        Init a tensorflow Graph containing: input data, variables, model, loss, optimizer
        '''
        self.graph = tf.Graph()
        with self.graph.as_default():  # , tf.device('/cpu:0'):
            # Set graph level random seed
            tf.set_random_seed(self.random_seed)
            np.random.seed(self.random_seed)
            # Input data.
            self.train_features = tf.placeholder(tf.int32, shape=[None, None], name="train_features")  # None * features_M
            self.train_field = tf.placeholder(tf.int32, shape=[None, None],name="train_field")  # None * features_M
            self.train_labels = tf.placeholder(tf.float32, shape=[None, 1], name="train_labels")  # None * 1
            #self.fields_M = 5
            #
            self.train_f_index = tf.placeholder(tf.int32, shape=[None, None], name="train_f_index")

            self.dropout_keep = tf.placeholder(tf.float32, name="dropout_keep")
            self.train_phase = tf.placeholder(tf.bool, name="train_phase")

            # Variables.
            self.weights = self._initialize_weights()
            self.nonzero_features_embeddings = tf.nn.embedding_lookup(self.weights['feature_embeddings'],
                                                             self.train_features)  # None * M' * K  
            self.nonzero_fields_embeddings = tf.nn.embedding_lookup(self.weights['field_embeddings'],
                                                                    self.train_field)  # None * M' * K  
            self.featureMulField = tf.multiply(self.nonzero_features_embeddings, self.nonzero_fields_embeddings)

            self.nonzero_features_embeddings_branch = tf.nn.embedding_lookup(self.weights['feature_embeddings_branch'],
                                                             self.train_features)

            batch_size = tf.shape(self.train_features)[0]
            sliced_embeddings = tf.TensorArray(dtype=tf.float32, size=batch_size, infer_shape=False)
            check_embeddings = tf.TensorArray(dtype=tf.int32, size=batch_size, infer_shape=False)

            _, sliced_embeddings,  check_embeddings= tf.while_loop(
                cond=lambda i, _, __: i < batch_size,
                body=self.process_batch,
                loop_vars=(0, sliced_embeddings, check_embeddings)
            )

            self.sliced_embeddings111 = sliced_embeddings.stack()



            # Model.
            element_wise_product_list = []
            count = 0
            for i in range(0, self.num_variable):
                for j in range(i + 1, self.num_variable):
                    Fi = self.train_field[:, i]
                    Fj = self.train_field[:, j]
                    max_ij = tf.maximum(Fi,Fj)
                    min_ij = tf.minimum(Fi, Fj)
                    FiFj = min_ij*self.fields_M + max_ij
                    rFiFj = tf.nn.embedding_lookup(self.weights['field_r'],FiFj)
                    vivj = tf.multiply(self.featureMulField[:, i, :], self.featureMulField[:, j,:])
                    element_wise_product_list.append(vivj) # i = None * K
                    count += 1
            self.element_wise_product = tf.stack(element_wise_product_list)  # (M'*(M'-1)) * None * K
            self.element_wise_product = tf.transpose(self.element_wise_product, perm=[1, 0, 2], name="element_wise_product")  # None * (M'*(M'-1)) * K
            self.FMFFwFM = tf.reduce_sum(self.element_wise_product, 1, name="FMFFwFM")  # None * K
            #self.prediction = tf.matmul(self.FwFM, self.weights['prediction'])
            self.prediction = tf.reduce_sum(self.FMFFwFM, 1, keep_dims=True)
            self.nonzero_linear = tf.nn.embedding_lookup(self.weights['feature_bias'],self.train_features)  # None * M' * K
            self.linear = tf.reduce_sum(self.nonzero_linear, 1)  # None * 1
            Bias = self.weights['bias'] * tf.ones_like(self.train_labels)  # None * 1
            self.out = tf.add_n([self.prediction, self.linear, Bias], name="out_FMFFwFM")
            #self.out = tf.add_n([self.prediction, self.linear, Bias, self.c_i], name="out_FIM")# None * 1


            if self.lamda_bilinear > 0:

                self.loss = tf.nn.l2_loss(tf.subtract(self.train_labels, self.out)) + tf.contrib.layers.l2_regularizer(
                    self.lamda_bilinear)(self.weights['feature_bias']) + tf.contrib.layers.l2_regularizer(
                    self.lamda_bilinear)(self.weights['feature_embeddings'])+ tf.contrib.layers.l2_regularizer(
                    self.lamda_bilinear)(self.weights['field_r']) + tf.contrib.layers.l2_regularizer(
                    self.lamda_bilinear)(self.weights['field_embeddings'])# regulizer
            else:
                self.loss = tf.nn.l2_loss(tf.subtract(self.train_labels, self.out))

            # Optimizer.
            if self.optimizer_type == 'AdamOptimizer':
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(self.loss)
            elif self.optimizer_type == 'AdagradOptimizer':
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate, initial_accumulator_value=1e-8).minimize(self.loss)
            elif self.optimizer_type == 'GradientDescentOptimizer':
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == 'MomentumOptimizer':
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(self.loss)

            # init
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = self._init_session()
            self.sess.run(init)

            # number of params
            total_parameters = 0
            for variable in self.weights.values():
                shape = variable.get_shape() # shape is an array of tf.Dimension
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            if self.verbose > 0:
                print("#params: %d" %total_parameters)



    def dense_layer_with_relu(self, x_l, x_1, name="layer", i=0):
        """

        parameter：
            a_l: [batch_size, input_dim]
            output_dim:
            name:

        return：
            a_l_plus_1: [batch_size, output_dim]
        """
        input_dim = self.fields_M * self.hidden_factor[0]
        output_dim = input_dim
        with tf.variable_scope(name):
            glorot_init = tf.glorot_normal_initializer()
            zeros_init = tf.zeros_initializer()
            W_l = tf.get_variable(
                name="W",
                shape=[input_dim, output_dim],
                initializer=glorot_init
            )
            b_l = tf.get_variable(
                name="b",
                shape=[output_dim],
                initializer=zeros_init
            )
            r = tf.random_uniform(shape=[], minval=0, maxval=5, dtype=tf.int32)

            actual_shift = self.hidden_factor[0] * i
            c_l = tf.matmul(x_l, W_l) + b_l
            c_rolled = tf.roll(c_l, shift=actual_shift, axis=-1)
            # x^{l+1} = x^l ⊙ c^l + x^l
            x_l_plus_1 = x_1 * c_rolled + x_l
            x_l_plus_1 = tf.nn.sigmoid(x_l_plus_1)


            return x_l_plus_1
    def process_batch(self, i, sliced_embeddings, check_embeddings):

        indices = self.train_f_index[i]  #  [num_indices]
        embeddings = self.nonzero_features_embeddings_branch[i]  # [seq_len, embedding_dim]

        #  [0, 1, 5, 12, 20]
        full_indices = tf.concat([[0], indices, [tf.shape(embeddings)[0]]], axis=0)
        print(full_indices)

        #
        batch_slices = tf.TensorArray(dtype=tf.float32, size=self.fields_M, infer_shape=False)

        #
        def process_slice(j, batch_slices):
            start = full_indices[j]
            end = full_indices[j + 1]
            slice_emb = embeddings[start:end]

            slice_mean = tf.reduce_mean(slice_emb, axis=0)
            batch_slices = batch_slices.write(j, slice_mean)
            return j + 1, batch_slices

        #
        _, batch_slices = tf.while_loop(
            cond=lambda j, _: j < self.fields_M,
            body=process_slice,
            loop_vars=(0, batch_slices)
        )
        #
        all_slices = batch_slices.stack()

        check_embeddings = check_embeddings.write(i, full_indices)
        sliced_embeddings = sliced_embeddings.write(i, all_slices)
        return i + 1, sliced_embeddings, check_embeddings
    def _init_session(self):
        # adaptively growing video memory
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        return tf.Session(config=config)

    def _initialize_weights(self):
        all_weights = dict()
        # if freeze_fm, set all other params untrainable
        trainable = self.freeze_fm == 0
        if self.pretrain_flag > 0 or self.micro_level_analysis:
            from_file = self.save_file
            # if self.micro_level_analysis:
            from_file = self.save_file.replace('fim', 'fm')
            weight_saver = tf.train.import_meta_graph(from_file + '.meta')
            pretrain_graph = tf.get_default_graph()
            feature_embeddings = pretrain_graph.get_tensor_by_name('feature_embeddings:0')
            feature_bias = pretrain_graph.get_tensor_by_name('feature_bias:0')
            bias = pretrain_graph.get_tensor_by_name('bias:0')
            with self._init_session() as sess:
                weight_saver.restore(sess, from_file)
                fe, fb, b = sess.run([feature_embeddings, feature_bias, bias])
            # all_weights['feature_embeddings'] = tf.Variable(fe, dtype=tf.float32, name='feature_embeddings')
            all_weights['feature_embeddings'] = tf.Variable(fe, dtype=tf.float32, name='feature_embeddings', trainable=trainable)
            all_weights['feature_bias'] = tf.Variable(fb, dtype=tf.float32, name='feature_bias', trainable=trainable)
            all_weights['bias'] = tf.Variable(b, dtype=tf.float32, name='bias', trainable=trainable)
        else:
            zero_pad = tf.zeros([1, self.hidden_factor[1]], dtype=tf.float32)
            zero_pad = tf.stop_gradient(zero_pad)
            zero_pad_2 = tf.zeros([1, 1], dtype=tf.float32)
            zero_pad_2 = tf.stop_gradient(zero_pad_2)

            all_weights['feature_embeddings'] = tf.Variable(
                tf.random_normal([self.features_M, self.hidden_factor[1]], 0, self.std, seed=self.init_seed),
                name='feature_embeddings')  # features_M * K
            all_weights['feature_embeddings'] = tf.concat([all_weights['feature_embeddings'], zero_pad], axis=0)

            all_weights['feature_embeddings_branch'] = tf.Variable(
                tf.random_normal([self.features_M, self.hidden_factor[1]], 0, self.std, seed=self.init_seed),
                name='feature_embeddings')
            all_weights['feature_embeddings_branch'] = tf.concat([all_weights['feature_embeddings_branch'], zero_pad], axis=0)

            all_weights['field_embeddings'] = tf.Variable(
                tf.random_normal([self.fields_M, self.hidden_factor[0]], 0, self.std, seed=self.init_seed),
                name='field_embeddings')  # fields_M * K
            all_weights['field_embeddings'] = tf.concat([all_weights['field_embeddings'], zero_pad], axis=0)

            all_weights['field_r'] = tf.Variable(
                tf.random_normal([(self.fields_M+1)*(self.fields_M + 1), 1], 0, self.std, seed=self.init_seed),
                name='field_r')  # features_M * K

            all_weights['feature_bias'] = tf.Variable(
                tf.random_normal([self.features_M, 1], 6.45/self.num_variable, self.std, seed=self.init_seed), name='feature_bias')  # features_M * 1
            all_weights['feature_bias'] = tf.concat([all_weights['feature_bias'], zero_pad_2], axis=0)

            all_weights['bias'] = tf.Variable(tf.constant(0.0), name='bias', trainable=trainable)  # 1 * 1

        # prediction layer
        all_weights['prediction'] = tf.Variable(np.ones((self.hidden_factor[1], 1), dtype=np.float32))  # hidden_factor * 1

        return all_weights

    def batch_norm_layer(self, x, train_phase, scope_bn):
        bn_train = batch_norm(x, decay=self.decay, center=True, scale=True, updates_collections=None,
            is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=self.decay, center=True, scale=True, updates_collections=None,
            is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z

    def partial_fit(self, data):  # fit a batch
        feed_dict = {self.train_features: data['X'], self.train_field: data['X_f'], self.train_f_index: data['X_i'],  self.train_labels: data['Y'], self.dropout_keep: self.keep, self.train_phase: True}
        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
        return loss

    def get_random_block_from_data(self, data, batch_size):  # generate a random block of training data
        #np.random.seed(self.random_seed)
        start_index = np.random.randint(0, len(data['Y']) - batch_size)
        X , X_f, X_i, Y =[], [], [], []
        # forward get sample
        i = start_index
        while len(X) < batch_size and i < len(data['X']):
            if len(data['X'][i]) == len(data['X'][start_index]):
                Y.append([data['Y'][i]])
                X.append(data['X'][i])
                X_f.append(data['X_f'][i])
                X_i.append(data['X_i'][i])

                #mask.append(data['mask'][i])
                i = i + 1
            else:
                break
        # backward get sample
        i = start_index
        while len(X) < batch_size and i >= 0:
            if len(data['X'][i]) == len(data['X'][start_index]):
                Y.append([data['Y'][i]])
                X.append(data['X'][i])
                X_f.append(data['X_f'][i])
                X_i.append(data['X_i'][i])
                #mask.append(data['mask'][i])
                i = i - 1
            else:
                break
        return {'X': X, 'X_f': X_f, 'X_i': X_i, 'Y': Y}

    def get_ordered_block_from_data(self, data, batch_size, index):  # generate a ordered block of data
        start_index = index*batch_size
        X , X_f, X_i, Y =[], [], [], []
        # get sample
        i = start_index
        while len(X) < batch_size and i < len(data['X']):
            if len(data['X'][i]) == len(data['X'][start_index]):
                Y.append(data['Y'][i])
                X.append(data['X'][i])
                X_f.append(data['X_f'][i])
                X_i.append(data['X_i'][i])
                #mask.append(data['mask'][i])
                i = i + 1
            else:
                break
        return {'X': X, 'X_f': X_f, 'X_i': X_i, 'Y': Y}

    def shuffle_in_unison_scary(self, a, b, c): # shuffle two lists simutaneously
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        np.random.set_state(rng_state)
        np.random.shuffle(c)

    def train(self, Train_data, Validation_data, Test_data):  # fit a dataset
        # Check Init performance
        if self.verbose > 0:
            t2 = time()
            init_train_MSE, init_train_MAE = self.evaluate(Train_data)
            init_valid_MSE, init_valid_MAE = self.evaluate(Validation_data)
            print("Init: \t train_MSE=%.4f, validation_MSE=%.4f \t train_MAE=%.4f, validation_MAE=%.4f [%.1f s]" %(init_train_MSE, init_valid_MSE, init_train_MAE, init_valid_MAE, time()-t2))

        for epoch in range(self.epoch):
            t1 = time()
            #self.shuffle_in_unison_scary(Train_data['X'],Train_data['X_f'], Train_data['Y'])
            total_batch = int(len(Train_data['Y']) / self.batch_size)
            for i in range(total_batch):
                # generate a batch
                batch_xs = self.get_random_block_from_data(Train_data, self.batch_size) #获取了一个随机块，可能并不是所有数据都能取到？
                # Fit training
                self.partial_fit(batch_xs)
            t2 = time()

            # evaluate training and validation datasets
            train_result_MSE, train_result_MAE = self.evaluate(Train_data)
            valid_result_MSE, valid_result_MAE = self.evaluate(Validation_data)
            self.train_mse.append(train_result_MSE)
            self.valid_mse.append(valid_result_MSE)
            self.train_mae.append(train_result_MAE)
            self.valid_mae.append(valid_result_MAE)

            if self.verbose > 0 and epoch%self.verbose == 0:
                print("Epoch %d [%.1f s]\ttrain_MSE=%.4f, validation_MSE=%.4f \t train_MAE=%.4f, validation_MAE=%.4f[%.1f s]"
                      %(epoch+1, t2-t1, train_result_MSE, valid_result_MSE, train_result_MAE, valid_result_MAE, time()-t2))

            test_result_MSE, test_result_MAE = self.evaluate(Test_data)
            self.test_mse.append(test_result_MSE)
            self.test_mae.append(test_result_MAE)
            print("Epoch %d [%.1f s]\ttest_MSE=%.4f, test_MAE=%.4f [%.1f s]"
                  %(epoch+1, t2-t1, test_result_MSE, test_result_MAE, time()-t2))
            if self.eva_termination(self.valid_mse): #跳出
                break

        # if self.pretrain_flag < 0 or self.pretrain_flag == 2:
        #     print("Save model to file as pretrain.")
        #     self.saver.save(self.sess, self.save_file)

    def eva_termination(self, valid):
        if len(valid) > 5:
            if valid[-1] > valid[-2] and valid[-2] > valid[-3] and valid[-3] > valid[-4] and valid[-4] > valid[-5]:
                return True
        return False

    def evaluate(self, data):  # evaluate the results for an input set
        num_example = len(data['Y'])
        # fetch the first batch
        batch_index = 0
        batch_xs = self.get_ordered_block_from_data(data, self.batch_size, batch_index)
        # batch_xs = data
        y_pred = None
        # if len(batch_xs['X']) > 0:
        while len(batch_xs['X']) > 0:
            num_batch = len(batch_xs['Y'])
            feed_dict = {self.train_features: batch_xs['X'],  self.train_field: batch_xs['X_f'], self.train_f_index: data['X_i'], self.train_labels: [[y] for y in batch_xs['Y']], self.dropout_keep: 1.0, self.train_phase: False}
            batch_out, wight_temp,  sliced_embeddings111 = self.sess.run(
                [self.out, self.weights,self.sliced_embeddings111], feed_dict=feed_dict)

            if batch_index == 0:
                y_pred = np.reshape(batch_out, (num_batch,))
            else:
                y_pred = np.concatenate((y_pred, np.reshape(batch_out, (num_batch,))))
            # fetch the next batch
            batch_index += 1
            batch_xs = self.get_ordered_block_from_data(data, self.batch_size, batch_index)

        y_true = np.reshape(data['Y'], (num_example,))

        predictions_bounded = np.maximum(y_pred, np.ones(num_example) * min(y_true))  # bound the lower values
        predictions_bounded = np.minimum(predictions_bounded, np.ones(num_example) * max(y_true))  # bound the higher values
        MSE = mean_squared_error(y_true, predictions_bounded)
        MAE =mean_absolute_error(y_true, predictions_bounded)
        return MSE, MAE

    def dynamic_split(self, field, indices):

        valid_indices = tf.boolean_mask(indices, indices >= 0)

        split_sizes = tf.concat([
            [valid_indices[0]],
            valid_indices[1:] - valid_indices[:-1],
            [tf.shape(field)[0] - valid_indices[-1]]
        ], axis=0)

        return tf.split(field, num_or_size_splits=split_sizes, axis=0)
def make_save_file(args):
    pretrain_path = '../pretrain/%s_%d' %(args.dataset, eval(args.hidden_factor)[1])
    if args.mla:
        pretrain_path += '_mla'
    if not os.path.exists(pretrain_path):
        os.makedirs(pretrain_path)
    save_file = pretrain_path+'/%s_%d' %(args.dataset, eval(args.hidden_factor)[1])
    return save_file

def train(args):
    np.random.seed(4096)
    r_1 = np.random.randint(low=0, high=2, size=(5)).tolist()
    # Data loading
    data = DATA.LoadData(args.path, args.dataset)
    if args.verbose > 0:
        print("FIM: dataset=%s, factors=%s, attention=%d, freeze_fm=%d, #epoch=%d, batch=%d, lr=%.4f, lambda_attention=%.1e, keep=%s, optimizer=%s, batch_norm=%d, decay=%f, activation=%s"
              %(args.dataset, args.hidden_factor, args.attention, args.freeze_fm, args.epoch, args.batch_size, args.lr, args.lamda, args.keep, args.optimizer,
              args.batch_norm, args.decay, args.activation))
    activation_function = tf.nn.relu
    if args.activation == 'sigmoid':
        activation_function = tf.sigmoid
    elif args.activation == 'tanh':
        activation_function == tf.tanh
    elif args.activation == 'identity':
        activation_function = tf.identity

    save_file = make_save_file(args)
    # Training
    t1 = time()

    #num_variable = data.truncate_features()
    num_variable = data.padding_features(args.endlen)
    #np.random.seed(2048)

    if args.mla:
        args.freeze_fm = 1
    model = FIM(data.features_M, data.field_M, args.pretrain, save_file, args.attention, eval(args.hidden_factor), num_variable,
        activation_function, num_variable, args.freeze_fm, args.epoch, args.batch_size, args.lr, args.lamda, args.keep, args.optimizer,
        args.batch_norm, args.decay, args.verbose, args.mla, args.init_seed, args.std, r_1, args.trpanum)
    model.train(data.Train_data, data.Validation_data, data.Test_data)
    # Find the best validation result across iterations
    best_valid_score = 0
    best_valid_score = min(model.valid_mse)
    best_epoch = model.valid_mse.index(best_valid_score)

    best_test_score = min(model.test_mse)
    best_epoch_test = model.test_mse.index(best_test_score)

    print ("Best Iter(validation)= %d\t train_mse = %.4f, valid_mse = %.4f,train_mae = %.4f, valid_mae = %.4f [%.1f s]"
           %(best_epoch+1, model.train_mse[best_epoch], model.valid_mse[best_epoch],model.train_mae[best_epoch], model.valid_mae[best_epoch], time()-t1))




if __name__ == '__main__':

    para_index = 0 #
    hidden_factor = ['[32,32]']  # 
    lamda_attention = [0.8]
    optimizer = ['AdamOptimizer']
    seed = [1024]
    lr = [0.01]
    std = [0.1]
    for hidden_factor_i in hidden_factor:
        for lamda_attention_i in lamda_attention:
            for optimizer_i in optimizer:
                for lr_i in lr:
                    for seed_i in seed:
                        for std_i in std:
                            para_index += 1
                            parser = argparse.ArgumentParser(description="Run.")
                            parser.add_argument('--process', nargs='?', default='all',
                                                help='Process type: all.')
                            parser.add_argument('--mla', type=int, default=0,
                                                help='Set the experiment mode to be Micro Level Analysis or not: 0-disable, 1-enable.')
                            parser.add_argument('--path', nargs='?', default='../data/',
                                                help='Input data path.')
                            parser.add_argument('--dataset', nargs='?', default='MM-Douban',
                                                help='Choose a dataset.')
                            parser.add_argument('--valid_dimen', type=int, default=3,
                                                help='Valid dimension of the dataset. (e.g. frappe=10, ml-tag=3)')
                            parser.add_argument('--epoch', type=int, default=1,
                                                help='Number of epochs.')
                            parser.add_argument('--pretrain', type=int, default=-1,
                                                help='flag for pretrain. 1: initialize from pretrain; 0: randomly initialize; -1: save to pretrain file; 2: initialize from pretrain and save to pretrain file')
                            parser.add_argument('--batch_size', type=int, default=512,
                                                help='Batch size.')
                            parser.add_argument('--attention', type=int, default=0,
                                                help='flag for attention. 1: use attention; 0: no attention')
                            parser.add_argument('--hidden_factor', nargs='?', default=hidden_factor_i,
                                                help='Number of hidden factors.')
                            parser.add_argument('--lamda', type=float, default=lamda_attention_i,
                                                help='Regularizer for attention part.')
                            parser.add_argument('--keep', nargs='?', default=1,
                                                help='Keep probility (1-dropout) of each layer. 1: no dropout. The first index is for the attention-aware pairwise interaction layer.')
                            parser.add_argument('--lr', type=float, default=lr_i,
                                                help='Learning rate.')
                            parser.add_argument('--freeze_fm', type=int, default=0,
                                                help='Freese all params of fm and learn attention params only.')
                            parser.add_argument('--optimizer', nargs='?', default=optimizer_i,
                                                help='Specify an optimizer type (AdamOptimizer, AdagradOptimizer, GradientDescentOptimizer, MomentumOptimizer).')
                            parser.add_argument('--verbose', type=int, default=1,
                                                help='Whether to show the performance of each epoch (0 or 1)')
                            parser.add_argument('--batch_norm', type=int, default=0,
                                                help='Whether to perform batch normaization (0 or 1)')
                            parser.add_argument('--decay', type=float, default=0.999,
                                                help='Decay value for batch norm')
                            parser.add_argument('--activation', nargs='?', default='relu',
                                                help='Which activation function to use for deep layers: relu, sigmoid, tanh, identity')
                            parser.add_argument('--seed', type=int, default=seed_i,
                                                help='fixseed')
                            parser.add_argument('--init_seed', type=int, default=seed_i,
                                                help='init_seed')
                            parser.add_argument('--std', type=float, default=std_i,
                                                help='std in init')
                            parser.add_argument('--trpanum', type=int, default=20,
                                                help='trpanum')
                            parser.add_argument('--endlen', type=int, default=23,
                                                help='endlen')

                            args = parser.parse_args()
                            if args.process == 'train':
                                train(args)




