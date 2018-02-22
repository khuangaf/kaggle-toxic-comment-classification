from __future__ import absolute_import, division
# coding: utf-8

# In[108]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
import re
import string
# Any results you write to the current directory are saved as output.

from keras.models import Model
from keras.layers import Dense, Embedding, Input, LeakyReLU
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout, CuDNNLSTM, CuDNNGRU
from keras.layers import Dense, Embedding, Input, LeakyReLU, merge, Conv2D, Conv1D, PReLU,ELU,Concatenate, Convolution1D
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, GRU, Dropout, CuDNNGRU, Reshape, MaxPool2D,Flatten, Lambda, Activation
from keras.layers.core import SpatialDropout1D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.backend.tensorflow_backend import set_session
from keras import regularizers, constraints
from keras.optimizers import RMSprop, Adam,Nadam

import os
import tensorflow as tf


# In[2]:


os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


# In[3]:


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))


# In[158]:


import h5py
with h5py.File('../input/fasttext_processed_rmnum_extra.h5', 'r') as f:
    x_train = f['x_train'].value
    y_train = f['y_train'].value
    x_test = f['x_test'].value

print ("file loaded")
# In[5]:


embedding_dim = 300
maxlen = 100





# In[100]:





import sys
from os.path import dirname
# sys.path.append(dirname(dirname(__file__)))
from keras import initializers
from keras.engine import InputSpec, Layer
from keras import backend as K


class AttentionWeightedAverage(Layer):
    """
    Computes a weighted average of the different channels across timesteps.
    Uses 1 parameter pr. channel to compute the attention value for a single timestep.
    """

    def __init__(self, return_attention=False, **kwargs):
        self.init = initializers.get('uniform')
        self.supports_masking = True
        self.return_attention = return_attention
        super(AttentionWeightedAverage, self).__init__(** kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(ndim=3)]
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[2], 1),
                                 name='{}_W'.format(self.name),
                                 initializer=self.init)
        self.trainable_weights = [self.W]
        super(AttentionWeightedAverage, self).build(input_shape)

    def call(self, x, mask=None):
        # computes a probability distribution over the timesteps
        # uses 'max trick' for numerical stability
        # reshape is done to avoid issue with Tensorflow
        # and 1-dimensional weights
        logits = K.dot(x, self.W)
        x_shape = K.shape(x)
        logits = K.reshape(logits, (x_shape[0], x_shape[1]))
        ai = K.exp(logits - K.max(logits, axis=-1, keepdims=True))

        # masked timesteps have zero weight
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            ai = ai * mask
        att_weights = ai / (K.sum(ai, axis=1, keepdims=True) + K.epsilon())
        weighted_input = x * K.expand_dims(att_weights)
        result = K.sum(weighted_input, axis=1)
        if self.return_attention:
            return [result, att_weights]
        return result

    def get_output_shape_for(self, input_shape):
        return self.compute_output_shape(input_shape)

    def compute_output_shape(self, input_shape):
        output_len = input_shape[2]
        if self.return_attention:
            return [(input_shape[0], output_len), (input_shape[0], input_shape[1])]
        return (input_shape[0], output_len)

    def compute_mask(self, input, input_mask=None):
        if isinstance(input_mask, list):
            return [None] * len(input_mask)
        else:
            return None


# In[101]:


def Crop(dimension, start, end):
    # Crops (or slices) a Tensor on a given dimension from start to end
    # example : to crop tensor x[:, :, 5:10]
    # call slice(2, 5, 10) as you want to crop on the second dimension
    def func(x):
        if dimension == 0:
            return x[start: end]
        if dimension == 1:
            return x[:, start: end]
        if dimension == 2:
            return x[:, :, start: end]
        if dimension == 3:
            return x[:, :, :, start: end]
        if dimension == 4:
            return x[:, :, :, :, start: end]
    return Lambda(func)


# In[150]:


def get_model():
    global embedding_dim
    embed_size = embedding_dim
    inp = Input(shape=(maxlen, embedding_dim ))
#     x = Embedding(max_features, embed_size)(inp)
#     x = Dropout(0.4)(x)
    x = SpatialDropout1D(0.2)(inp)
    x = Bidirectional(CuDNNGRU(50, return_sequences=True))(x)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)

#     x = Dropout(0.4)(x)
#     x = Bidirectional(CuDNNLSTM(50, return_sequences=True))(x)
    A = AttentionWeightedAverage(name='attlayer', return_attention=False)(x)
#     A = AttentionWithContext()(x)
#     x = AttentionWithContext()(x)
#     C = Crop(2,-2,-1)(x)
#     C = Reshape([-1,])(C)
#     print(C)
    G = GlobalMaxPool1D()(x)
    x = Concatenate()([A,G])
    x = Dropout(0.1)(x)
    
    x = Dense(50, activation=None)(x)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
#     model.layers[1].set_weights([embedding_matrix])
#     model.layers[1].trainable = False
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(amsgrad=True),
                  metrics=['accuracy'])
    return model


# In[151]:


from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
def train_bagging(X, y, model_func, fold_count, batch_size, num_epoch, patience, verbose=0):
    
    best_auc_score = -np.inf
    best_weights = None
    kf = KFold(n_splits=fold_count, random_state=None, shuffle=False)
    fold_id = -1
    model_list = []
    for train_index, test_index in kf.split(X):
        fold_id +=1 
        model = model_func()
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        not_improve_count = 0
        current_best_auc_score = -np.inf
        for e in range(num_epoch):
            if verbose: print("Fold {0}, epoch {1}".format(fold_id, e))
            model.fit(X_train,y_train, batch_size=batch_size, verbose=0)
            y_pred = model.predict(X_test)
            auc = roc_auc_score(y_test, y_pred)
            
            if auc > current_best_auc_score:
                if verbose: print("Current AUC Score improved from {0} to {1}.".format(current_best_auc_score, auc))
                current_best_auc_score = auc
                current_best_weights = model.get_weights()
                not_improve_count = 0
            else:
                if verbose: print("Current AUC Score did not improved. {}".format(auc))
                not_improve_count += 1
                if not_improve_count >= patience:
                    model.set_weights(current_best_weights)
                    model_list.append(model)
                    if verbose: print ("Model appended.")
                    break
        if current_best_auc_score > best_auc_score:
            print("Best AUC Score improved from {0} to {1}.".format(best_auc_score, current_best_auc_score))
            best_weights = current_best_weights
            best_auc_score = current_best_auc_score
        else:
            print("Best AUC Score did not improved. {0}".format(current_best_auc_score))
        
            
#     model.set_weights(best_weights)
    return model_list
    
    


# In[159]:




patience=6
print ("start training..")
fname='sp_bigru_relu_ft100_d1_atten_global_amsgrad_rmnum_extra_all55'

batch_size= 512
epochs = 55
model = get_model()

model.fit(x_train, y_train, epochs =epochs, batch_size=batch_size)
# model.load_weights(filepath)
# y_pred = model.predict(x_test, verbose=1, batch_size=batch_size)


# In[42]:


# fname='sp_bigru_ft100atten_global_all'
# model = get_model()
# model.fit(x_train, y_train, epochs =13, batch_size=batch_size)
y_pred = model.predict(x_test, verbose=1, batch_size=batch_size)


# In[17]:


test_data = pd.read_csv('../input/sample_submission.csv')


# In[ ]:


test_ids = test_data["id"].values
test_ids = test_ids.reshape((len(test_ids), 1))

CLASSES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
test_predicts = pd.DataFrame(data=y_pred, columns=CLASSES)
test_predicts["id"] = test_ids
test_predicts = test_predicts[["id"] + CLASSES]
submit_path = os.path.join("output", fname+".csv")
test_predicts.to_csv(submit_path, index=False)

