
'''
@Author Nitin Bansal
Many of Preprocessing function and Idea from https://github.com/fchollet/keras/blob/master/examples/babi_memnn.py
'''
from __future__ import print_function
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, Permute, Dropout, add, dot, concatenate
from keras.layers import LSTM

from keras.preprocessing.sequence import pad_sequences
from functools import reduce
import numpy as np
import re
import os


"""Pass all the strings of a file to this function. This function would divide each in to seperate 
Sentences and each sentences in to seperate tokens, Giving a result which consists of List of List"""
def tokenize_sentence(file_str):
    """Return a List of Tokens"""
    sent_tokenize_list = sent_tokenize(file_str)
    for i in range(0,1):
        word_token = word_tokenize(sent_tokenize_list[i])
        return word_token

def parse_stories(lines, only_supporting=False):
    '''Parse stories provided in the bAbi tasks format
    If only_supporting is true, only the sentences
    that support the answer are kept.
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


def get_stories(f, only_supporting=False, max_length=None):
    '''Given a file name, read the file,
    retrieve the stories,
    and then convert the sentences into a single story.
    If max_length is supplied,
    any stories longer than max_length tokens will be discarded.
    '''
    data = parse_stories(f.readlines(), only_supporting=only_supporting)
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(story), q, answer) for story, q, answer in data if not max_length or len(flatten(story)) < max_length]
    return data


def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
    X = []
    Xq = []
    Y = []
    for story, query, answer in data:
        x = [word_idx[w] for w in story]
        xq = [word_idx[w] for w in query]
        # let's not forget that index 0 is reserved
        y = np.zeros(len(word_idx) + 1)
        y[word_idx[answer]] = 1
        X.append(x)
        Xq.append(xq)
        Y.append(y)
    return (pad_sequences(X, maxlen=story_maxlen),
            pad_sequences(Xq, maxlen=query_maxlen), np.array(Y))

#Obtaining the file
path = os.getcwd()
path = path + '/tasks_1-20_v1-2/en/'

#We Can Work on 1000 Sample Dataset(1000 Questions Per Task), 10K Dataset
#English Dataset or Hindi Dataset, or Even Shuffled Data set(which shows the 
#Model used is Language Agnostic!)
files_train = [f for f in os.listdir(path) if re.match(r'.*train.*', f)]
files_train = sorted(files_train)
files_test = [f for f in os.listdir(path) if re.match(r'.*test.*', f)]
files_test = sorted(files_test)

files_train


# In[45]:

#Selecting the first file of Both Train and test
for i in range(0,20):
	tra = files_train[i]
	tes = files_test[i]

	print (tra)
	print (tes)
	os.chdir(path)

	train = open(tra, "r")
	test = open(tes,"r")

	#Getting all the words present in the Train and Test file respectively
	train_stories = get_stories(train)
	test_stories = get_stories(test)


	vocab = set()
	for story, q, answer in train_stories + test_stories:
	    vocab |= set(story + q + [answer])
	vocab = sorted(vocab)

	# Reserve 0 for masking via pad_sequences
	vocab_size = len(vocab) + 1
	story_maxlen = max(map(len, (x for x, _, _ in train_stories + test_stories)))
	query_maxlen = max(map(len, (x for _, x, _ in train_stories + test_stories)))

	print('-')
	print('Vocab size:', vocab_size, 'unique words')
	print('Story max length:', story_maxlen, 'words')
	print('Query max length:', query_maxlen, 'words')
	print('Number of training stories:', len(train_stories))
	print('Number of test stories:', len(test_stories))
	print('-')
	print('Here\'s what a "story" tuple looks like (input, query, answer):')
	print(train_stories[0])
	print('-')
	print('Vectorizing the word sequences...')

	word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
	inputs_train, queries_train, answers_train = vectorize_stories(train_stories,
                                                               word_idx,
                                                               story_maxlen,
                                                               query_maxlen)
	inputs_test, queries_test, answers_test = vectorize_stories(test_stories,
                                                            word_idx,
                                                            story_maxlen,
                                                            query_maxlen)

	print('-')
	print('inputs: integer tensor of shape (samples, max_length)')
	print('inputs_train shape:', inputs_train.shape)
	print('inputs_test shape:', inputs_test.shape)
	print('-')
	print('queries: integer tensor of shape (samples, max_length)')
	print('queries_train shape:', queries_train.shape)
	print('queries_test shape:', queries_test.shape)
	print('-')
	print('answers: binary (1 or 0) tensor of shape (samples, vocab_size)')
	print('answers_train shape:', answers_train.shape)
	print('answers_test shape:', answers_test.shape)
	print('-')
	print('Compiling...')

	#Making a Model for K-HOP End to End Networks
	# where the value of K =3
	# placeholders
	input_sequence = Input((story_maxlen,))
	question = Input((query_maxlen,))

	# encoders
	# embed the input sequence into a sequence of vectors
	input_encoder_m = Sequential()
	input_encoder_m.add(Embedding(input_dim=vocab_size,
				output_dim=64))
	input_encoder_m.add(Dropout(0.3))
	# output: (samples, story_maxlen, embedding_dim)

	# embed the input into a sequence of vectors of size query_maxlen
	input_encoder_c = Sequential()
	input_encoder_c.add(Embedding(input_dim=vocab_size,
				output_dim=64))
	input_encoder_c.add(Dropout(0.3))
	# output: (samples, story_maxlen, query_maxlen)

	# embed the question into a sequence of vectors
	question_encoder = Sequential()
	question_encoder.add(Embedding(input_dim=vocab_size,
				output_dim=64,
				input_length=query_maxlen))
	question_encoder.add(Dropout(0.3))
	# output: (samples, query_maxlen, embedding_dim)

	# encode input sequence and questions (which are indices)
	# to sequences of dense vectors
	input_encoded_m = input_encoder_m(input_sequence)
	input_encoded_c = input_encoder_c(input_sequence)
	question_encoded = question_encoder(question)

	# compute a 'match' between the first input vector sequence
	# and the question vector sequence
	# shape: `(samples, story_maxlen, query_maxlen)`
	match = dot([input_encoded_m, question_encoded], axes=(2, 2))
	match = Activation('softmax')(match)

	# add the match matrix with the second input vector sequence
	match = Permute((2,1))(match)
	input_encoded_c = Permute((2,1))(input_encoded_c)
	response = dot([match, input_encoded_c],axes = (2,2))  # (samples, story_maxlen, query_maxlen)

	# concatenate the match matrix with the question vector sequence
	answer = add([response, question_encoded])

	#########################################
	# Going By Layer-Wise Binding of the 3 Hops 
	# Keeping All the Embedding Matrix Same
	# u_k+1 = u_k + o_k

	question_encoded1 = answer
	# compute a 'match' between the first input vector sequence
	# and the question vector sequence
	# shape: `(samples, story_maxlen, query_maxlen)`
	match = dot([input_encoded_m, question_encoded1], axes=(2, 2))
	match = Activation('softmax')(match)


	#add the match matrix with the second input vector sequence
	match = Permute((2,1))(match)
	response = dot([match, input_encoded_c], axes = (2,2))  # (samples, story_maxlen, query_ma    xlen)

	# concatenate the match matrix with the question vector sequence
	answer = add([response, question_encoded1])
	question_encoded2 = answer
	# compute a 'match' between the first input vector sequence
	# and the question vector sequence
	# shape: `(samples, story_maxlen, query_maxlen)`
	match = dot([input_encoded_m, question_encoded2], axes=(2, 2))
	match = Activation('softmax')(match)

	match = Permute((2,1))(match)
	response = dot([match, input_encoded_c], axes = (2,2))
	answer = add([response,question_encoded2])
	# the original paper uses a matrix multiplication for this reduction step.
	# we choose to use a RNN instead.
	answer = LSTM(32)(answer)  # (samples, 32)

	# one regularization layer -- more would probably be needed.
	answer = Dropout(0.3)(answer)
	answer = Dense(vocab_size)(answer)  # (samples, vocab_size)
	# we output a probability distribution over the vocabulary
	answer = Activation('softmax')(answer)

	# build the final model
	model = Model([input_sequence, question], answer)
	model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
			metrics=['accuracy'])

	# train
	model.fit([inputs_train, queries_train], answers_train,
			batch_size=32,
			epochs=40,
			#validation_data=([inputs_test, queries_test], answers_test))
			validation_split=0.05)
	loss, acc = model.evaluate([inputs_test, queries_test], answers_test,
			batch_size=32)
	print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))



