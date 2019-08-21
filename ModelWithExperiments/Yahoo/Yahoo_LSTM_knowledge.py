
# coding: utf-8

# In[1]:


import numpy as np
import re
import csv
import pandas as pd
from sklearn.utils import shuffle
import os


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Flatten, Activation
from keras.layers import Convolution1D, MaxPooling1D, Embedding, LSTM
from keras.models import Model, load_model
from keras.layers import Input, Dropout
from keras.optimizers import SGD, Adadelta
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from sklearn import metrics
from sklearn import *
from sklearn.model_selection import train_test_split, KFold
# import theano
import csv
import h5py

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from scipy import interp
from itertools import cycle


# In[2]:


# Function To plot the model history

import matplotlib.pyplot as plt
plt.style.use('ggplot')

# Source : https://realpython.com/python-keras-text-classification/

def plot_history(history):
    valid_acc = history['val_acc']    
    train_acc = history['acc']
    valid_loss = history['val_loss']
    train_loss = history['loss']
    r = range(1, len(train_acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(r, valid_acc, 'r', label='Validation Accuracy')
    plt.plot(r, train_acc, 'b', label='Training Accuracy')
    plt.title('Training and validation Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(r, valid_loss, 'r', label='Validation Loss')    
    plt.plot(r, train_loss, 'b', label='Training Loss')
    plt.title('Training and validation loss')
    plt.legend()
    
    plt.show()


# #### Construct the vector for Label Data

# In[3]:


with open('Data/yahoo_label.txt',encoding='utf-8') as f:
    labels = f.read().splitlines()

labels = list(map(int, labels))
labels = to_categorical(np.asarray(labels))


# #### Set the Hyperparameters

# In[4]:


EMBEDDING_DIM = 50
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.3
max_seq_length = 1000
max_word_cnt = 20000
Classes = 10
DROP_OUT = 0.3
BATCH_SIZE = 50
EPOCH = 10
optimizer="adam"
loss="categorical_crossentropy"

GLOVE_DIR = 'glove.6B/'
FILENAME = 'glove.6B.' + str(EMBEDDING_DIM) + 'd.txt'


# ---------------------

# ---------------------

# #### Without Knowledge

# In[5]:


with open('Data/yahoo_raw.txt',encoding='utf-8') as f:
    texts_without_knowledge = f.read().splitlines()


# In[6]:


print('Found %s texts.' % len(texts_without_knowledge))

# Tokenize the sentences to create dictionary
tokenizer = Tokenizer(num_words=max_word_cnt)
tokenizer.fit_on_texts(texts_without_knowledge)
sequences = tokenizer.texts_to_sequences(texts_without_knowledge)

# Create Word Index
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

# Make Data
data = pad_sequences(sequences, maxlen=max_seq_length)

print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)


# In[7]:


## Prepare the Glove Embedding data
embeddings_index = {}
gloveFileName = os.path.join(GLOVE_DIR, FILENAME)
file = open(gloveFileName, encoding="utf8")
for line in file:
    values = line.split()
    word = values[0]
    coeffs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coeffs
f.close()

print('Found %s word vectors.' % len(embeddings_index))


# In[8]:


## Prepare Embedding Matrix
words_nb = min(max_word_cnt, len(word_index))
embedding_matrix = np.zeros((words_nb + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    if i > max_word_cnt:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


# In[9]:


embedding_matrix.shape


# In[10]:


def create_model():
    print('Number of class: ||%d||' % (Classes))
    model = Sequential()
    model.add(Embedding( 
        input_length=max_seq_length,name = "embedding_layer",
        weights=[embedding_matrix],input_dim=words_nb + 1,                   
        output_dim= EMBEDDING_DIM,trainable=False))
    

    model.add(LSTM(128, name="lstm_layer", dropout=DROP_OUT, recurrent_dropout=DROP_OUT))
    model.add(Dense(Classes, activation = 'sigmoid', name = "dense_one"))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    return model


# In[11]:


model = None 
model = create_model()

train_X, test_X, train_Y, test_Y = train_test_split(data, labels, test_size=TEST_SPLIT, random_state = 42)


# In[12]:


model.summary()


# In[13]:


history = model.fit(train_X, train_Y, validation_split=VALIDATION_SPLIT, epochs=EPOCH, 
                    batch_size=BATCH_SIZE) 


training_loss, training_accuracy = model.evaluate(train_X, train_Y)
print ("Training Loss: ", training_loss)
print ("Training Accuracy: ", training_accuracy)

eval_loss, eval_accuracy = model.evaluate(test_X, test_Y)
print ("Testing Loss: ", eval_loss)
print ("Testing Accuracy: ", eval_accuracy)

model_history = history.history


# In[14]:


plot_history(model_history)


# In[15]:


test_Y_predProb = model.predict(test_X, verbose=0)
test_Y_predClass = np.argmax(test_Y_predProb,axis=1)
test_Y_class = np.argmax(test_Y,axis=1)


# In[16]:


print("Precision (macro): %f" % precision_score(test_Y_class, test_Y_predClass, average='macro'))
print("Recall (macro):    %f" % recall_score(test_Y_class, test_Y_predClass, average='macro'))
print("F1 score (macro):  %f" % f1_score(test_Y_class, test_Y_predClass, average='macro'), end='\n\n')
print("Precision (weighted): %f" % precision_score(test_Y_class, test_Y_predClass, average='weighted'))
print("Recall (weighted):    %f" % recall_score(test_Y_class, test_Y_predClass, average='weighted'))
print("F1 score (weighted):  %f" % f1_score(test_Y_class, test_Y_predClass, average='weighted'))


# ----------------------

# ----------------------

# #### With Knowledge

# In[5]:


with open('Data/yahoo_know.txt',encoding='utf-8') as f:
    texts_with_knowledge = f.read().splitlines()


# In[6]:


print('Found %s texts.' % len(texts_with_knowledge))
global word_index, tokenizer

tokenizer = Tokenizer(num_words=max_word_cnt)
tokenizer.fit_on_texts(texts_with_knowledge)
sequences = tokenizer.texts_to_sequences(texts_with_knowledge)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=max_seq_length)

print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)


# In[7]:


global embeddings_index 
embeddings_index = {}
fname = os.path.join(GLOVE_DIR, FILENAME)
f = open(fname, encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))


# In[8]:


print('Preparing embedding matrix.')
words_nb = min(max_word_cnt, len(word_index))
embedding_matrix = np.zeros((words_nb + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    if i > max_word_cnt:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


# In[9]:


embedding_matrix.shape


# In[10]:


def create_LSTM_model():
    print('Number of class: ||%d||' % (Classes))
    model = Sequential()
    model.add(Embedding( 
        input_length=max_seq_length,name = "embedding_layer",
        weights=[embedding_matrix],input_dim=words_nb + 1,                   
        output_dim= EMBEDDING_DIM,trainable=False))
    
    model.add(LSTM(128, name="lstm_layer", recurrent_dropout=DROP_OUT, dropout=DROP_OUT))
    model.add(Dense(Classes, name = "dense_one", activation = 'sigmoid'))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    return model


# In[11]:


model = None 
model = create_LSTM_model()

train_X, test_X, train_Y, test_Y = train_test_split(data, labels, test_size=TEST_SPLIT, random_state = 42)
model.summary()


# In[12]:


history = model.fit(train_X, train_Y, validation_split=VALIDATION_SPLIT, epochs=EPOCH, 
                    batch_size=BATCH_SIZE) 

training_loss, training_accuracy = model.evaluate(train_X, train_Y)
print ("Training Loss: ", training_loss)
print ("Training Accuracy: ", training_accuracy)

eval_loss, eval_accuracy = model.evaluate(test_X, test_Y)
print ("Testing Loss: ", eval_loss)
print ("Testing Accuracy: ", eval_accuracy)

model_history = history.history


# In[13]:


plot_history(model_history)


# In[14]:


test_Y_predProb = model.predict(test_X, verbose=0)
test_Y_predClass = np.argmax(test_Y_predProb,axis=1)
test_Y_class = np.argmax(test_Y,axis=1)


# In[15]:


print("Precision (macro): %f" % precision_score(test_Y_class, test_Y_predClass, average='macro'))
print("Recall (macro):    %f" % recall_score(test_Y_class, test_Y_predClass, average='macro'))
print("F1 score (macro):  %f" % f1_score(test_Y_class, test_Y_predClass, average='macro'), end='\n\n')
print("Precision (weighted): %f" % precision_score(test_Y_class, test_Y_predClass, average='weighted'))
print("Recall (weighted):    %f" % recall_score(test_Y_class, test_Y_predClass, average='weighted'))
print("F1 score (weighted):  %f" % f1_score(test_Y_class, test_Y_predClass, average='weighted'))

