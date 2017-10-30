'''Example script adopted from keras examples to 
implement/try different model structures for text generation
'''

from __future__ import print_function
from keras.layers import Dense, Activation, Embedding, Dropout, Reshape, Input, Concatenate
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from keras.preprocessing import sequence
from keras.models import Model
from keras import metrics

import numpy as np
import random
import sys

#added character embeddings
EMBEDDING_DIM = 40

path = get_file('nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
text = open(path).read().lower()
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
#maxlen is the total 'scope' lstm will be seeing
maxlen = 1000
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
	sentences.append(text[i: i + maxlen])
	next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
x = np.zeros((len(sentences), maxlen), dtype=np.int32)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
	for t, char in enumerate(sentence):
		x[i, t]= char_indices[char]
	y[i, char_indices[next_chars[i]]] = 1

#must split in order to do divide and conquer architecture
x = np.split(x,100,axis=1)
# build the model: a single LSTM
print('Build model...')

#smallest length of chars model will look at
LOWEST_DIV = 10
embedding_layer = Embedding(len(chars),
							EMBEDDING_DIM,
							input_length=LOWEST_DIV)

#list that will be fed as input to the model
input_list = []
index = 0
for i in range(0,maxlen,LOWEST_DIV):
	input_list.append(Input(shape=(LOWEST_DIV,), dtype='int32'))
	index += 1

#shared lstm layer
lstm_1 = LSTM(128)

first_layer_dict = {}
embedding_dict = {}
count = 0
for i in range(0,maxlen,LOWEST_DIV):
	embedding_dict["lstm{0}".format(count)] = embedding_layer(input_list[count])
	first_layer_dict["lstm{0}".format(count)] = (lstm_1)(embedding_dict["lstm{0}".format(count)])
	count += 1

concat_layer = Concatenate(axis =1)([Reshape((1,128))(first_layer_dict["lstm0"]),Reshape((1,128))(first_layer_dict["lstm1"])])
for i in range(2,len(input_list)):
	concat_layer = Concatenate(axis=1)([concat_layer, Reshape((1,128))(first_layer_dict["lstm{0}".format(i)])])

lstm_2 = LSTM(128)(concat_layer)
lstm_2 = Dropout(.2)(lstm_2)
dense = Dense(100, activation = 'relu')(lstm_2)
dense = Dropout(.2)(dense)
out = Dense(len(chars))(dense)
output = Activation('softmax')(out)

model = Model(input_list, output)
optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

#save pic of model
with open('./model_summary.txt','w+') as fh:
	model.summary(print_fn=lambda t: fh.write(t + '\n'))


def sample(preds, temperature=1.0):
	# helper function to sample an index from a probability array
	preds = np.asarray(preds).astype('float64')
	preds = np.log(preds) / temperature
	exp_preds = np.exp(preds)
	preds = exp_preds / np.sum(exp_preds)
	probas = np.random.multinomial(1, preds, 1)
	return np.argmax(probas)

# train the model, output generated text after each iteration
for iteration in range(1, 60):
	print()
	print('-' * 50)
	print('Iteration', iteration)
	model.fit(x, y,
			batch_size=128,
			epochs=1)
	start_index = random.randint(0, len(text) - maxlen - 1)
	for diversity in [0.2, 0.5, 1.0, 1.2]:
		print()
		print('----- diversity:', diversity)
		generated = ''
		sentence = text[start_index: start_index + maxlen]
		generated += sentence
		print('----- Generating with seed: "' + sentence + '"')
		sys.stdout.write(generated)
		for i in range(400):
			x_pred = np.zeros((1, maxlen))
			#change x_pred to account for divide and conquer architecture
			
			for t, char in enumerate(sentence):
				x_pred[0,t] = char_indices[char]
			
			x_pred = np.split(x_pred, 100, axis=1)
			preds = model.predict(x_pred, verbose=0)[0]
			next_index = sample(preds, diversity)
			next_char = indices_char[next_index]
			generated += next_char
			sentence = sentence[1:] + next_char
			sys.stdout.write(next_char)
			sys.stdout.flush()
		print()