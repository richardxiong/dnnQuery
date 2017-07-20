# By Hongyu Xiong
# construct vocabulary and represented by GloVe word vectors
import math
import os
import random
import sys
import time

import numpy as np
import data_utils
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

def generateEmbedMatrix(vocabulary_path, max_vocabulary_size, glove_path = None):
	print "Importing GloVe pretrained word vectors"
    
	vocab, _ = data_utils.initialize_vocabulary(vocabulary_path)	# (token ---> idx)

	if glove_path is not None:
		word_vectors = constructGloveDict(glove_path, max_vocabulary_size)
	else:
		word_vectors = None
	# random initialize word vectors, with a larger radius than glove vector, in order to deal with outside-vocab words
	# embedding_matrix = np.asarray(np.random.uniform(-1.99, 1.99, (len(vocab), 100)), dtype = 'float32')
	embedding_matrix = np.asarray(np.random.normal(0, 1, (len(vocab), 300)), dtype = 'float32')

	if word_vectors is not None:
		print "Replacing GloVe word vectors as initialization"
		for token in vocab:
			if token in word_vectors:
				embedding_matrix[vocab[token], :] = np.array(word_vectors[token]) 
	return embedding_matrix, vocab, word_vectors

# def bucketStat():
# 	_buckets = [(10, 8), (15, 12), (19, 16), (23, 21)]
# 	max_size = None
# 	data_set = [[] for _ in _buckets]

# 	with tf.gfile.GFile('../data/rand_train.qu', mode="r") as source_file:
#     	with tf.gfile.GFile('../data/rand_train.lo', mode="r") as target_file:
#         	with tf.gfile.GFile('../data/rand_train.ta', mode="r") as tag_file:
#             	source, target, tag = source_file.readline(), target_file.readline(), tag_file.readline()
#             	counter = 0
#             	while source and target and tag and (not max_size or counter < max_size):
#                 	counter += 1
#                 	if counter % 200 == 0:
#                     	print("  reading data line %d" % counter)
#                     	sys.stdout.flush()
#                 	source_ids = [x for x in source.split()]
#                 	arget_ids = [x for x in target.split()]
#                		tag_ids = [x for x in tag.split()]
#                 	target_ids.append(data_utils_tag.EOS_ID)
#                 	for bucket_id, (source_size, target_size) in enumerate(_buckets):
#                     	if len(source_ids) < source_size and len(target_ids) < target_size:
#                         	data_set[bucket_id].append([source_ids, tag_ids, target_ids])
#                         	break
#                 	source, target, tag = source_file.readline(), target_file.readline(), tag_file.readline()

# 	print "bucket 0: %d" % len(data_set[0])		# 801
# 	print "bucket 1: %d" % len(data_set[1])		# 732
# 	print "bucket 2: %d" % len(data_set[2])		# 711
# 	print "bucket 3: %d" % len(data_set[3])		# 556
# 	return

# def randomizeData():
# 	Fqu = []
# 	Flo = []
# 	Ffi = []

# 	with open('../data/rand_dev.fi') as f_fi:
#     	with open('../data/rand_dev.qu') as f_qu:
#         	with open('../data/rand_dev.lo') as f_lo:
#             	schema, query, logic = f_fi.readline(), f_qu.readline(), f_lo.readline()
#             	idx = 0
#             	while schema and query and logic:
#                 	Fqu.append(query)
#                 	Flo.append(logic)
#                 	Ffi.append(schema)
#                 	schema, query, logic = f_fi.readline(), f_qu.readline(), f_lo.readline()
                
# 	num = np.random.permutation(len(Fqu))
# 	print num
# 	f_fi1 = open('../data/rand_dev2.fi', 'w')
# 	f_lo1 = open('../data/rand_dev2.lo', 'w')
# 	f_qu1 = open('../data/rand_dev2.qu', 'w')

# 	for i in range(len(num)):
#     	f_lo1.write(Flo[num[i]])
#     	f_fi1.write(Ffi[num[i]])
#     	f_qu1.write(Fqu[num[i]])

# 	f_fi1.close()
# 	f_qu1.close()
# 	f_lo1.close()
# 	return