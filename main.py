# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 15:49:29 2017

@author: Zayn
"""
from __future__ import print_function
import os
import sys
import pandas as pd
import numpy as np
import scipy as sp
from utils import *
from data import *
from model_utils import *
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from keras.callbacks import ModelCheckpoint
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.callbacks import Callback, ModelCheckpoint
#prepare data
data_dir = 'data'
pick_class = 9
# Set batch size and number of training epochs
batch_size = 24
nb_epoch = 50

df_text = pd.read_csv(os.path.join('data','df_text_for_cnn_UTF8.csv'))
df, go = read_images_and_metadata(data_dir, pick_class)
df_good, sorted_df, label_PR, label_PR_cutted = clean_data(data_dir, df)
sorted_df, msk = split_data(sorted_df, label_PR_cutted)
df_title = df_text[df_text.productID.isin(df_good.productID)].CHN_NAME
title_list = split_title(df_title)
voc_size, word2index = build_vocb(title_list)
filter_title_list, title_max_len = filter_words(title_list, word2index)
X = text2seq(word2index, filter_title_list, title_max_len)
#X = text2mat(word2index, title_max_len, filter_title_list)

X_train = X[msk]
X_test = X[~msk]
y_train = label_PR[msk]
y_test = label_PR[~msk]

#prepare wordembedding
word2vec = Word2Vec.load(os.path.join('model', 'wiki_w2v_100.bin'))
embedding_weights = np.zeros((voc_size, word2vec.vector_size))
for word, index in word2index.items():
    try:
        embedding_weights[index] = word2vec.wv[word]
    except KeyError:
        pass
#build model
model = Sequential()
model.add(Embedding(input_dim=voc_size, output_dim=word2vec.vector_size,
                    weights=[embedding_weights], mask_zero=True, trainable=False))
model.add(Bidirectional(LSTM(128, return_sequences = True),input_shape = (title_max_len,word2vec.vector_size), merge_mode="concat"))
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(128), merge_mode="concat"))
model.add(Dropout(0.5))
"""
model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(32))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))"""

model.add(Dense(1,activation = 'sigmoid'))

model.summary()

#set model file name
opt = '_original' + "0" + "pick_class" + str(pick_class) + '_LSTM'
model_file_name = "./model/booksTW_sales12wk_regression" + opt + ".h5" # add dynamic naming rule here
checkpoint = ModelCheckpoint(model_file_name,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='auto')
loss_history = LossHistory()
Peason_R = PeasonR()

myoptimizer = Adam()

model.compile(loss='mean_absolute_error', 
              optimizer=myoptimizer)

print_time_info("Start training...")
history_model = model.fit(X_train, y_train,
                batch_size= batch_size,
                epochs =nb_epoch,
                #validation_split = 0.1,
                validation_data = (X_test, y_test),
                shuffle=True,
                callbacks = [loss_history, checkpoint, Peason_R])
print_time_info("Complete training...")

train_loss = history_model.history.get("loss")
val_loss = history_model.history.get("val_loss")
train_mae = history_model.history.get("mean_absolute_error")
val_mae = history_model.history.get("val_mean_absolute_error")
val_corr = np.hstack(history_model.history.get('val_corr')).tolist()

## Save training and validation loss
recorder = pd.DataFrame.from_dict(history_model.history)
orig_stdout = sys.stdout

#save model summary 
with open(model_file_name + ".txt", "w+") as f:
	sys.stdout = f
	print(model.summary())
	print("------------------")
	print('loss, train:', min(train_loss), 'test:', min(val_loss))
	print('cor max, test: ', max(val_corr))
	sys.stdout = orig_stdout

#save model ealuation per iteration
recorder.to_csv(model_file_name + '_record.csv', sep = "\t", encoding='utf-8')


model = load_model(model_file_name)
print_time_info("Restoring model from {}...".format(model_file_name))


train_pred = model.predict(X_train, batch_size=batch_size)
test_pred = model.predict(X_test, batch_size=batch_size)

perform_check_train = pd.DataFrame([
        np.array(y_train, dtype="float32"),
        train_pred.squeeze()
        ]).transpose()
perform_check_test = pd.DataFrame([
        np.array(y_test, dtype="float32"),
        test_pred.squeeze()
        ]).transpose()

perform_check_train.columns = [["y", "yp"]]
perform_check_test.columns = [["y", "yp"]]

perform_check_train['y_diff'] = perform_check_train['y'].sub(perform_check_train['yp'], axis = 0)
print(np.square(perform_check_train['y_diff']).mean())

perform_check_test['y_diff'] = perform_check_test['y'].sub(perform_check_test['yp'], axis = 0)
print(np.square(perform_check_test['y_diff']).mean())

result_train_cor = sp.stats.pearsonr(train_pred.squeeze(), y_train)
result_test_cor = sp.stats.pearsonr(test_pred.squeeze(), y_test)
print(result_train_cor, result_test_cor)

pic_base_name = "./pics/" + model_file_name.split("/")[2].split(".h5")[0]
print(pic_base_name)

plt.figure()
plt.plot(range(recorder.shape[0]), recorder.loss, label = "Training")
plt.plot(range(recorder.shape[0]), recorder.val_loss, label = "Validation")
plt.title("loss (MAE)")
plt.legend()
plt.plot()
plt.savefig(pic_base_name + "_learning_curve.png")
#plt.show()

# plot correlation between predict and real value (in training set)
a = [np.min(y_train), np.max(y_train), np.min(train_pred), np.max(train_pred)]
plt_train_min, plt_train_max = np.min(a), np.max(a)
print(plt_train_min, plt_train_max)
pt = np.arange(plt_train_min, plt_train_max, 0.1)

plt.figure()
plt.scatter(y_train, train_pred, alpha = 0.25, )
plt.plot(pt, pt, 'r--')
plt.title("training set, actual value to predicted value, cor = " + str(np.round(result_train_cor[0],3)) )
plt.ylabel("predicted percentile")
plt.xlabel("real percentile")
axes = plt.gca()
axes.set_xlim([0, plt_train_max])
axes.set_ylim([0, plt_train_max])
plt.plot()
plt.savefig(pic_base_name + "_corplot_train.png")

# plot correlation between predict and real value (in test set)
a = [np.min(y_test), np.max(y_test), np.min(test_pred), np.max(test_pred)]
plt_test_min, plt_test_max = np.min(a), np.max(a)
print(plt_test_min, plt_test_max)
pt = np.arange(plt_test_min, plt_test_max, 0.1)

plt.figure()
plt.scatter(y_test, test_pred, alpha = 0.25)
plt.plot(pt, pt, 'r--')
plt.title("testing set, actual value to predicted value, cor = " + str(np.round(result_test_cor[0],3)) )
plt.ylabel("predicted percentile")
plt.xlabel("real percentile")
axes = plt.gca()
axes.set_xlim([0, plt_test_max])
axes.set_ylim([0, plt_test_max])
plt.plot()
plt.savefig(pic_base_name + "_corplot_test.png")


