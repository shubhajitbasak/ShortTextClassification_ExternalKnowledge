
# coding: utf-8

# In[1]:


import numpy as np
import re
import csv
import pandas as pd
from sklearn.utils import shuffle
import os

from keras.models import Model,Sequential
from keras.layers import Input, Dense, Flatten
from keras.layers import Convolution1D, Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Embedding
from keras.layers import ThresholdedReLU
from keras.layers import Dropout, SpatialDropout1D, concatenate


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint

import tensorflow as tf
import json

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc


# In[2]:


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


# In[3]:


MAX_SEQUENCE_LENGTH = 1000
MAX_WORD_COUNT = 20000
EMBEDDING_DIM = 50
BATCH_SIZE = 50
TEST_SPLIT = 0.3
VALIDATION_SPLIT = 0.1
DROP_OUT = 0.3
EPOCH = 10
Classes = 7
optimizer="adam"
loss="categorical_crossentropy"

GLOVE_DIR = 'glove.6B/'
FILENAME = 'glove.6B.' + str(EMBEDDING_DIM) + 'd.txt'


# #### Construct the vector for Label Data

# In[4]:


with open('Data/Kaggle/Kaggle_labels.txt',encoding='utf-8') as f:
    labels = f.read().splitlines()

labels = list(map(int, labels))
labels = to_categorical(np.asarray(labels))


# In[5]:


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


# #### Create Word Model

# In[6]:


def createWordModel():
    # Input layer
    inputs_words = Input(shape=(MAX_SEQUENCE_LENGTH,), name='sent_input', dtype='int64')
    # Embedding layers
    x = Embedding(                          
        input_dim=word_count + 1, output_dim= EMBEDDING_DIM,           
        weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH,
        name = "embedding_layer", trainable=False)(inputs_words)
    l_cov1= Conv1D(128, 5, activation='relu')(x)
    l_pool1 = MaxPooling1D(5)(l_cov1)
    l_cov2 = Conv1D(128, 5, activation='relu')(l_pool1)
    spac_dropout1 = SpatialDropout1D(0.5)(l_cov2)
    l_pool2 = MaxPooling1D(5)(spac_dropout1)
    drop_out1 = Dropout(0.5)(l_pool2)
    l_cov3 = Conv1D(128, 5, activation='relu')(drop_out1)
    l_pool3 = MaxPooling1D(35)(l_cov3)  # global max pooling
    l_flat = Flatten()(l_pool3)
    drop_out2 = Dropout(0.5)(l_flat)
    l_dense = Dense(64, activation='relu')(drop_out2)
    predictions_words = Dense(Classes, activation='softmax')(l_dense)
    # Build and compile model
    model_words = Model(inputs=inputs_words, outputs=predictions_words)
    model_words.compile(optimizer=optimizer, loss=loss,metrics=['accuracy'])
    print(model_words.summary())
    
    return model_words


# ---------------

# #### With Raw Data

# In[7]:


with open('Data/Kaggle/Kaggle_raw.txt',encoding='utf-8') as f:
    texts_without_knowledge = f.read().splitlines()


# In[8]:


print(len(texts_without_knowledge))
print(len(labels))


# In[9]:


print('Found %s texts.' % len(texts_without_knowledge))

tokenizer = Tokenizer(num_words=MAX_WORD_COUNT)
tokenizer.fit_on_texts(texts_without_knowledge)

sequences = tokenizer.texts_to_sequences(texts_without_knowledge)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
# labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)


# In[10]:


# Preparing embedding matrix
word_count = min(MAX_WORD_COUNT, len(word_index))
embedding_matrix = np.zeros((word_count + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    if i > MAX_WORD_COUNT:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


# In[11]:


model = None 
model = createWordModel()

train_X, test_X, train_Y, test_Y = train_test_split(data, labels, test_size=TEST_SPLIT, random_state = 20)


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


# -----------------------
