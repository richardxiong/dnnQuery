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

def generateEmbedMatrix(vocabulary_path, max_vocabulary_size, glove_path = None):
	print "Importing GloVe pretrained word vectors"
    
	vocab, _ = data_utils.initialize_vocabulary(vocabulary_path)	# (token ---> idx)

	if glove_path is not None:
		word_vectors = constructGloveDict(glove_path, max_vocabulary_size)
	else:
		word_vectors = None
	embedding_matrix = np.asarray(np.random.normal(0, 0.9, (len(vocab), 100)), dtype = 'float32')

	if word_vectors is not None:
		print "Replacing GloVe word vectors as initialization"
		for token in vocab:
			if token in word_vectors:
				embedding_matrix[vocab[token], :] = np.array(word_vectors[token]) 
	return embedding_matrix, vocab, word_vectors

# _buckets = [(10, 8), (15, 12), (20, 16), (24, 21)]

# max_size = None
# data_set = [[] for _ in _buckets]
# with tf.gfile.GFile('./data6/rand_train.qu', mode="r") as source_file:
#     with tf.gfile.GFile('./data6/rand_train.lo', mode="r") as target_file:
#         with tf.gfile.GFile('./data6/rand_train.ta', mode="r") as tag_file:
#             source, target, tag = source_file.readline(), target_file.readline(), tag_file.readline()
#             counter = 0
#             while source and target and tag and (not max_size or counter < max_size):
#                 counter += 1
#                 if counter % 200 == 0:
#                     print("  reading data line %d" % counter)
#                     sys.stdout.flush()
#                 source_ids = [x for x in source.split()]
#                 target_ids = [x for x in target.split()]
#                 tag_ids = [x for x in tag.split()]
#                 target_ids.append(data_utils.EOS_ID)
#                 for bucket_id, (source_size, target_size) in enumerate(_buckets):
#                     if len(source_ids) < source_size and len(target_ids) < target_size:
#                         data_set[bucket_id].append([source_ids, tag_ids, target_ids])
#                         break
#                 source, target, tag = source_file.readline(), target_file.readline(), tag_file.readline()

# print "bucket 0: %d" % len(data_set[0])
# print "bucket 1: %d" % len(data_set[1])
# print "bucket 2: %d" % len(data_set[2])
# print "bucket 3: %d" % len(data_set[3])

#embedding_matrix = generateEmbedMatrix('./data4/wikicompose_train.qu.ids10000', './glove.6B/glove.6B.50d.txt', 100000)
#print embedding_matrix

#f_qu = open('wikicompose_1_random.qu', 'w')
#f_lo = open('wikicompose_1_random.lo', 'w')

# f_qu1 = open('wikicompose_train.qu', 'w')
# f_lo1 = open('wikicompose_train.lo', 'w')
# f_qu2 = open('wikicompose_dev.qu', 'w')
# f_lo2 = open('wikicompose_dev.lo', 'w')

# Fqu = [[] for i in range(97)]
# Flo = [[] for i in range(97)]

# count = 0
# hash_func = dict()
# with open('wikicompose_1_random.lo') as f1:
#     for line in f1:
#         num = random.randint(0, 96)
#         hash_func[count] = num
#         Flo[num].append(line)
#         count += 1

# count2 = 0
# with open('wikicompose_1_random.qu') as f2:
#     for line in f2:
#         num = hash_func[count2]
#         Fqu[num].append(line)
#         count2 += 1

# for i in range(97):
#     if i < 81:
#         for line in Flo[i]:
#             f_lo1.write(line)
#         for line in Fqu[i]:
#             f_qu1.write(line)
#     else:
#         for line in Flo[i]:
#             f_lo2.write(line)
#         for line in Fqu[i]:
#             f_qu2.write(line)
        
# # f_qu.close()
# # f_lo.close()
# f_qu1.close()
# f_lo1.close()
# f_qu2.close()
# f_lo2.close()