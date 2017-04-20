
# coding: utf-8

#@author Nitin Bansal
#Change Rescurrent function to SimpleRNN and GRU for their results

from __future__ import division
from functools import reduce
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

#import tensorflow as tf
import re
import numpy as np
import os
import keras

# In[40]:

from keras.layers.core import Dense, Dropout
from keras.utils.data_utils import get_file
from keras.layers.embeddings import Embedding
from keras import layers
from keras.layers import recurrent
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.legacy.layers import Merge
#from keras.layers import Dense, Activation



"""Pass all the strings of a file to this function. This function would divide each in to seperate 
Sentences and each sentences in to seperate tokens, Giving a result which consists of List of List"""
def tokenize_sentence(file_str):
    """Return a List of Tokens"""
    sent_tokenize_list = sent_tokenize(file_str)
    for i in range(0,1):
        word_token = word_tokenize(sent_tokenize_list[i])
        return word_token

"""Going through the Story and Parsing them Accordingly, Verifying
    If we need to do Strong Supervision or Weak Supervision depending 
    upon if we consider the supporting Facts. Returns a List of tuples 
    which consist of (Story, Query and Answer)"""
def parse_stories(lines, only_supporting=False):
    '''Parse stories provided in the bAbi tasks format
    If only_supporting is true,
    only the sentences that support the answer are kept.
    '''
    data = []
    story = []
    for line in lines:
        line = line.strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            substory = None
            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]
            data.append((substory, q, a))
            story.append('')
        else:
            sent = tokenize(line)
            story.append(sent)
    return data
            
        


# In[42]:


def get_stories(f, only_supporting=False, max_length=None):
    '''Given a file name, read the file, retrieve the stories,
    and then convert the sentences into a single story.
    If max_length is supplied,
    any stories longer than max_length tokens will be discarded.
    '''
    data = parse_stories(f.readlines(), only_supporting=only_supporting)
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(story), q, answer) for story, q, answer in data if not max_length or len(flatten(story)) < max_length]
    return data


# In[43]:

"""Converting the Story Query in to Vector form"""
def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
    #temp_story is a list, which consist of all the stories for a given Task
    #Similarly for Query and Answer
    temp_story = []
    temp_query = []
    temp_answer = []
    
    for story, query, ans in data:
        x = [word_idx[w] for w in story]
        xq = [word_idx[w] for w in query]
        #Index 0 is reserved for UNKNOWN TOKENS or Padded Sequence
        a = np.zeros(len(word_idx) + 1)
        a[word_idx[ans]] = 1
        temp_story.append(x)
        temp_query.append(xq)
        temp_answer.append(a)
    #Returning  a Paded Sequence according to maxlen provided or the longest sequence  present 
    #which is 2-D Numpy Array
    return pad_sequences(temp_story,maxlen = story_maxlen), pad_sequences(temp_query, maxlen = query_maxlen), np.array(temp_answer)


# In[44]:

#Starting with the main Block
#Defining the Selected Parameters, which can be changed for  
# obtaining the result for different selections.
RNN = recurrent.LSTM
EMBED_HIDDEN_SIZE = 50
BATCH_SIZE = 32
EPOCHS = 40

#Accessing the Saved Dataset
#Obtaining the file
path = os.getcwd()
path = path + '/tasks_1-20_v1-2/en/'
files_train = [f for f in os.listdir(path) if re.match(r'.*train.*', f)]
files_train = sorted(files_train)
files_test = [f for f in os.listdir(path) if re.match(r'.*test.*', f)]
files_test = sorted(files_test)

files_train


# In[45]:

#Selecting the first file of Both Train and test
for i in range(0,20):
	tra = files_train[0]
	tes = files_test[0]

	print (tra)
	print (tes)
	os.chdir(path)

	train = open(tra, "r")
	test = open(tes,"r")

	#Getting all the words present in the Train and Test file respectively
	train = get_stories(train,True)
	test = get_stories(test)

	vocab = set()
	for story, q, answer in train + test:
    		vocab |= set(story + q + [answer])
	vocab = sorted(vocab)
	#Now we are ready to build One-hot vector representation for it

	#Reserving 0 for masking via pad_sequence
	vocab_size = len(vocab) + 1
	word_idx = dict((c, i+ 1) for i,c in enumerate(vocab))
	story_maxlen = max(map(len, (x for x,_,_ in train + test)))
	query_maxlen = max(map(len, (x for _,x,_ in train + test)))

	x,xq,y = vectorize_stories(train,word_idx,story_maxlen,query_maxlen)
	tx,txq,ty = vectorize_stories(test,word_idx,story_maxlen,query_maxlen)

	print('vocab = {}'.format(vocab))
	print('Build model...')

	#Main Model Building Using the Merge Model
	#Defined in Smerity Article

	sentrnn = Sequential()
	sentrnn.add(Embedding(vocab_size, EMBED_HIDDEN_SIZE, input_length=story_maxlen, mask_zero=True))
	sentrnn.add(Dropout(0.3))
	sentrnn.add(RNN(EMBED_HIDDEN_SIZE, return_sequences = False))

	qrnn = Sequential()
	qrnn.add(Embedding(vocab_size, EMBED_HIDDEN_SIZE, input_length=query_maxlen,  mask_zero=True))
	qrnn.add(Dropout(0.3))
	qrnn.add(RNN(EMBED_HIDDEN_SIZE,  return_sequences = False))

	model = Sequential()
	model.add(Merge([sentrnn, qrnn], mode='concat'))
	model.add(Dense(vocab_size,activation='softmax'))

	model.compile(optimizer='adam',
                	loss='categorical_crossentropy',
                	metrics=['accuracy'])

	print('Training')
	model.fit([x, xq], y,
            	batch_size=BATCH_SIZE,
            	epochs=EPOCHS,
            	validation_split=0.05)
	acc = model.evaluate([tx, txq], ty,
                        batch_size=BATCH_SIZE)
	print('Test accuracy = {:.4f}'.format(acc))







