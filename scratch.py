# By Hongyu Xiong
# construct vocabulary and represented by GloVe word vectors
import math
import os
import random
import sys
import time

import numpy as np
import data_utils_tag
import tensorflow as tf

def constructGloveDict(glove_path, max_vocabulary_size):
	count = 0
	word_vectors = dict()
	for line in open(glove_path).readlines():
		if count > max_vocabulary_size: 
			break
		sp = line.strip().split()
		word_vectors[sp[0]] = [float(x) for x in sp[1:]]	# dictionary, token ---> Glove pretrained vector
		count += 1
		if count % 10000 == 0:
			print "		reading %d lines from GloVe file" % count
	return word_vectors

def generateEmbedMatrix(vocabulary_path, max_vocabulary_size, glove_path = None, unit_var = False):
	print "Importing GloVe pretrained word vectors"
    
	vocab, _ = data_utils_tag.initialize_vocabulary(vocabulary_path)	# (token ---> idx)

	if glove_path is not None:
		word_vectors = constructGloveDict(glove_path, max_vocabulary_size)
		glove_matrix = np.zeros((len(word_vectors), 300))
		count = 0
		for k,v in word_vectors.items(): 
			glove_matrix[count] = np.array(v)
			count += 1
	else:
		word_vectors = None
	# random initialize word vectors, with a larger radius than glove vector, in order to deal with outside-vocab words
	if word_vectors is not None:
		print "Replacing GloVe word vectors as initialization"
		stat = 0
		mu = np.mean(glove_matrix, axis=0)
		sigma = np.std(glove_matrix, axis=0)
		embedding_matrix = np.concatenate([np.random.normal(mu[x], sigma[x], (len(vocab), 1)) for x in range(300)], axis=1)
		for token in vocab:
			if token in word_vectors:
				embedding_matrix[vocab[token], :] = np.array(word_vectors[token]) 
				stat += 1
		print "Coverage = %.4f" % (1.0 * stat / len(vocab))
		if unit_var:
			embedding_matrix /= np.reshape(np.std(embedding_matrix, axis=1), (embedding_matrix.shape[0], 1))
	else: 
		embedding_matrix = np.asarray(np.random.normal(0, 1, (len(vocab), 300)), dtype = 'float32')
		print "Coverage = NaN (no pretrained vector used)"
	return embedding_matrix, vocab, word_vectors
