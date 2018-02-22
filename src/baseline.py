
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output


# Any results you write to the current directory are saved as output.

from keras.models import Model
from keras.layers import Dense, Embedding, Input, LeakyReLU, merge, Conv2D, Conv1D, PReLU,ELU,Concatenate, Convolution1D
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, GRU, Dropout, CuDNNGRU, CuDNNLSTM,MaxPooling1D,MaxPool2D,MaxPooling1D
from keras.preprocessing import text, sequence
from keras.layers.core import *
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.backend.tensorflow_backend import set_session
import os
import tensorflow as tf
import h5py
import cPickle as pickle
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

max_features = 20000
maxlen = 100
embedding_dim = 300
TIME_STEPS = 100
SINGLE_ATTENTION_VECTOR = False
with h5py.File(''.join(['toxic_token.h5']), 'r') as hf:
    X_t = hf['X_t'].value
    X_te = hf['X_te'].value
    y = hf['y'].value
    embedding_matrix = hf['embedding_matrix'].value
with open('word_index.p', 'rb') as fp:
    word_index = pickle.load(fp)
    
    
# score_columns = ['toxic_level','attack','aggression']
# train_scores = pd.read_csv('train_with_convai.csv').loc[:,score_columns]
# test_scores = pd.read_csv('test_with_convai.csv').loc[:,score_columns]
    
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
# train = train.sample(frac=1)

list_sentences_train = train["comment_text"].fillna("CVxTz").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
list_sentences_test = test["comment_text"].fillna("CVxTz").values
tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
word_index = tokenizer.word_index
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_t = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)




embeddings_index={}
f = open( 'glove.42B.300d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()


embedding_matrix = np.zeros((max_features, embedding_dim))
for word, i in word_index.items():
    if i < max_features:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            
            
# import h5py
# with h5py.File('toxic_token_external_snowball.h5','w') as f:
#     f.create_dataset("X_t", data = X_t )
#     f.create_dataset("X_te", data = X_te )
#     f.create_dataset("y", data = y )
#     f.create_dataset("embedding_matrix", data=embedding_matrix)
    
# import cPickle as pickle
# with open('word_index_external_snowball.p', 'wb') as fp:
#     pickle.dump(word_index, fp)

# def attention_3d_block(inputs):
#     print inputs.shape
#     # inputs.shape = (batch_size, time_steps, input_dim)
#     input_dim = int(inputs.shape[2])
#     a = Permute((2, 1))(inputs)
#     # a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
#     a = Dense(TIME_STEPS, activation='softmax')(a)
#     if SINGLE_ATTENTION_VECTOR:
#         a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
#         a = RepeatVector(input_dim)(a)
#     a_probs = Permute((2, 1), name='attention_vec')(a)
#     output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
#     return output_attention_mul

def get_model():
    global embedding_matrix, embedding_dim
    embed_size = embedding_dim
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size)(inp)
    
    x = Bidirectional(CuDNNGRU(50, return_sequences=True))(x)
    x = Dropout(0.2)(x)
    
    x = GlobalMaxPool1D()(x)

    x = Dense(40)(x)
    score_input = Input(shape = (3,))
    score = Dense(10)(score_input)
    score = Dropout(0.1)(score)
    x = Concatenate()([x,score])
    x = LeakyReLU()(x)
    x = Dropout(0.2)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=[inp,score_input], outputs=x)
    model.layers[1].set_weights([embedding_matrix])
    model.layers[1].trainable = False
    
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model

def get_cnn_model():
    global embedding_matrix, embedding_dim
    filter_sizes = [3,8,5]
    num_filters = 10
    drop = 0.1
    
    embed_size = embedding_dim
    inputs = Input(shape=(maxlen, ), dtype='int32')
    embedding = Embedding(input_dim=max_features, output_dim=embedding_dim, input_length=maxlen)(inputs)
    reshape = Reshape((maxlen,embedding_dim,1))(embedding)

    conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
    conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
    conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
    conv_0 = Dropout(0.1)(conv_0)
    maxpool_0 = MaxPool2D(pool_size=(maxlen - filter_sizes[0] + 1, 1), strides=(1,1), padding='valid')(conv_0)
    # maxpool_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(maxpool_0)
    # maxpool_0 = MaxPool2D(pool_size=(maxlen - filter_sizes[0] + 1, 1), strides=(1,1), padding='valid')(maxpool_0)
    conv_1 = Dropout(0.1)(conv_1)
    maxpool_1 = MaxPool2D(pool_size=(maxlen - filter_sizes[1] + 1, 1), strides=(1,1), padding='valid')(conv_1)
    maxpool_2 = MaxPool2D(pool_size=(maxlen - filter_sizes[2] + 1, 1), strides=(1,1), padding='valid')(conv_2)

    concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1])
    # concatenated_tensor = maxpool_0
    flatten = Flatten()(concatenated_tensor)
    flatten = Dense(30, activation='relu')(flatten)
    dropout = Dropout(drop)(flatten)
    output = Dense(6, activation='sigmoid')(dropout)
    model = Model(inputs=inputs, outputs=output)
    model.layers[1].set_weights([embedding_matrix])
    model.layers[1].trainable = False
    
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model



model = get_model()
batch_size = 32
epochs = 10

file_path="weights_base.pre_trained.hdf5"
checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

early = EarlyStopping(monitor="val_loss", mode="min", patience=20)


callbacks_list = [checkpoint, early] #early
model.fit([X_t, train_scores], y, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=callbacks_list, shuffle=False)
intermetiate_X = model.predict(X_t)


model.load_weights(file_path)

y_test = model.predict(X_te, verbose=1)
sample_submission = pd.read_csv("sample_submission.csv")
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
sample_submission[list_classes] = y_test

sample_submission.to_csv("baseline_pretrained.csv", index=False)