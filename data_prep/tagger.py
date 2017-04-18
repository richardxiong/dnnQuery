# Generating a tagged sentence for input query
# By Hongyu Xiong, 02/02/2017

import math
import os
import random
import sys
import time
import re

import editdistance as ed
import numpy as np
import tag_utils as tu

def buildDictionary(schema):
  ''' Build a search dictionary based on a given schema
      tu.Config() contains a huge SCHEMA database, which contains all the fields in schema given
      return --- query_dict, mapping between words and possible field names they refer
                 string_dict, mapping between values and possible field names they refer
  '''
    config = tu.Config()
    query_dict = dict()
    string_dict = dict()
    #num_dict = dict()
    #for key in config.field2word:
    for key in schema:
        # build query_dict: for field names
        if key not in config.field2word:
            continue
        value = config.field2word[key]
        for query_word in value['query_word']:
            if query_word not in query_dict:
                # one value could potentially corresponds to several field names
                query_dict[query_word] = [] 
            query_dict[query_word].append(key)
        if value['value_type'] is 'string':
            # build string_dict: for field values
            for word in value['value_range']:
                if word not in string_dict:
                    string_dict[word] = []
                string_dict[word].append(key)
        # else:
        #     # build num_dict
    return query_dict, string_dict#, num_dict

def fromWordtoVec():
  ''' generate the field to vector dictionary from the field to words dictionary
    # field2vec = {"nation": [vec1, vec2],...}    # vec1 is average vec for query words;
    #                                             # vec2 is average vec for value words; 
                 # vec could be weighted by words frequency used for future work
  '''  
    config = tu.Config()
    field2vec = dict()
    for key in config.field2word:
        field2vec[key] = []
        if config.field2word[key]['value_type'] == 'string':
            value_vector = np.array([0.0 for i in range(config.word_dim)])
            count = 0
            for word in config.field2word[key]['value_range']:
                if word.lower() not in config.word_vector:
                    continue
                vec = config.word_vector[word.lower()]
                value_vector += vec
                count += 1
            if count > 0:
                # consider it is possible that no values is in the range
                # so no value_vector is added
                value_vector /= count
                field2vec[key].append(value_vector)
        query_vector = np.array([0.0 for i in range(config.word_dim)])
        count = 0
        for word in config.field2word[key]['query_word']:
            if word.lower() not in config.word_vector:
                continue
            vec = config.word_vector[word.lower()]
            query_vector += vec
            count += 1
        if count > 0:
            # consider it is possible that no query words
            # so no query_vector is appended
            query_vector /= count
            field2vec[key].append(query_vector)
        #if len(field2vec[key]) == 0:
            # no vectors added, then delete
            # del field2vec[key]
    return field2vec

def fromWordtoVecList(schema):
  ''' generate the field to vector dictionary from the field to words dictionary
    # field2vecQuery = {"nation": [vec1, vec2,...]}    # vec are centroids; 
    # field2vecString = {"nation": [vec1, vec2,...]}    # vec are centroids; 
                 # vec could be weighted by words frequency used for future work
  '''
    config = tu.Config()
    field2vecQuery = dict()
    field2vecString = dict()
    # for key in config.field2word:
    for key in schema:
        field2vecString[key] = []
        field2vecQuery[key] = []
        if key not in config.field2word:
            continue
        if config.field2word[key]['value_type'] == 'string':
            for word in config.field2word[key]['value_range']:
                if word.lower() not in config.word_vector:
                    continue
                value_vector = config.word_vector[word.lower()]
                field2vec[key].append(value_vector)

        for word in config.field2word[key]['query_word']:
            if word.lower() not in config.word_vector:
                continue
            query_vector = config.word_vector[word.lower()]
            field2vec[key].append(query_vector)
        #if len(field2vec[key]) == 0:
            # no vectors added, then delete
            # del field2vec[key]
    return field2vecQuery, field2vecString

def strSimilarity(word1, word2):
  ''' Measure the similarity based on Edit Distance'''
    ### Measure how similar word1 is with respect to word2
    diff = ed.eval(word1.lower(), word2.lower())   #search
    # lcs = LCS(word1, word2)   #search
    length = max(len(word1), len(word2))
    if diff >= length:
        similarity = 0.0
    else:
        similarity = 1.0 * (length-diff) / length
    return similarity

def EuDisSqu(vector1, vector2):
  ''' Calculate the square of Euclidean distance between two vectors'''
    dist_square = 0.0
    dist_square += np.dot(vector1, vector1) + np.dot(vector2, vector2) - 2*np.dot(vector1, vector2)
    return np.sqrt(dist_square)

def spanInSpace(vectorArray):
  ''' calculate the maximum distance within a list of word vectors '''
    dist = 0.0
    for i in range(len(vectorArray)):
        for j in range(i+1, len(vectorArray)):
            x = EuDisSqu(vectorArray[i], vectorArray[j])
            if dist < x:
                dist = x
    span = dist
    return span

def spanInSpace2(vectorArray):
  ''' calculate 1 the average distance 
                2 the standard deviation within a list of word vectors '''
    dist = 0.0
    std = 0.0
    center = np.array([0.0 for i in range(len(vectorArray[0]))])
    dist_arr = []
    for i in range(len(vectorArray)):
        for j in range(i+1, len(vectorArray)):
            x = EuDisSqu(vectorArray[i], vectorArray[j])
            dist_arr.append(x)
      center += vectorArray[i]/len(vectorArray)
    dist_arr = np.array(dist_arr)
    dist = np.mean(dist_arr)
    std = np.std(dist_arr)
    return center, dist, std

def findNearestNeighbor(word, vectorArray, diameter):
  ''' return the word vector which is the nearest neighbor to the query word
      return None is word is a new vocab or the distance between word and any of the vecs is larger than diameter
  '''
    config = tu.Config()
    if word in config.word_vector:
        char_vec = config.word_vector[word]
    else:
        char_vec = config.embedding_matrix[config.vocab[word], :]
    nearest = diameter
    idx = None
    for i in range(len(vectorArray)):
        x = EuDisSqu(vectorArray[i], char_vec)
        if x > diameter:
            continue
        if x < nearest:
            nearest = x
            idx = i
    if idx is None:
        return None
    else:
        #print idx
        #print nearest
        return vectorArray[idx]

def findKNearestNeighbor(word, vectorArray, k=3):
  ''' return the list of word vectors which are the k nearest neighbors to the query word with corresponding 
              weights (1 / distance), list of tuples
      return None is word is a new vocab or the distance between word and any of the vecs is larger than diameter
  '''
    config = tu.Config()
    if word in config.word_vector:
        char_vec = config.word_vector[word]
    else:
        char_vec = config.embedding_matrix[config.vocab[word], :]
    center, diameter, std = spanInSpace2(vectorArray)
    if EuDisSqu(center, char_vec) > 0.5*diameter+1.5*std:
        return None
    
    dist_array = np.array([0.0 for i in range(len(vectorArray))])
    for i in range(len(vectorArray)):
        dist_array[i] = EuDisSqu(vectorArray[i], char_vec)
    idx = np.argsort(dist_array)
    
    result = []
    if k > len(vectorArray):
        k = len(vectorArray)
    for i in range(k):
        result.append((vectorArray[idx[i]], 1.0 / dist_array[idx[i]]))
    return result

def strIsNum(s):
    ### verify if the word represent a numerical value
    if not isinstance(s, basestring):
        return 0
    if s.isdigit():
        return True #return int(s)
    # for float value
    try:
        x = float(s)
        return True
    except ValueError:
        return False
    return False

def basic_tokenizer(sentence):
    ### Very basic tokenizer: split the sentence into a list of tokens.
    words = []
    WORD_SPLIT = re.compile(b"([,!?\"':;)(])")   # get rid of '.'
    for space_separated_fragment in sentence.strip().split():
        words.extend(WORD_SPLIT.split(space_separated_fragment))
    return [w for w in words if w]


def sentTagging(query, schema):
    #schema = ["Nation", "Rank", "Gold", "Silver", "Bronze", "#_participant", "Total"]
    #query = 'how many gold medals the nation ranked 14 won'
    # config = tu.Config()
    # print config.word_dim
    field2vec = fromWordtoVec()
    query = query.lower()
    words = basic_tokenizer(query)
    tag = ["<nan>" for x in words]
    schema0 = ['sum', 'diff','less','greater','argmax', 'argmin'] #, 'mean', ]
    schema += schema0
    if 'Average' not in schema:
        schema.append('mean')
    # construct schema_vec, a dict with (key = vector for centroids, value = schema field name)
    schema_vec = dict()
    for i in range(len(schema)):
        vec_list = field2vec[schema[i]]
        for j in range(len(vec_list)):
            vec_sign = tuple(vec_list[j])
            #vec_sign = vec_list[j]
            schema_vec[vec_sign] = schema[i]
    # go over words, find nearest neighbor
    diameter = spanInSpace(schema_vec.keys())
    # 0th pass find devided words, such as country_gdp
    for j in range(len(schema)):
        # split around '_'
        div_fields = schema[j].split('_')
        k = len(div_fields)
        if k == 1:
            continue
        for i in range(len(words) - k + 1):
            if tag[i] is not "<nan>":
                continue
            check_combine = '_'.join(words[i:i+k])
            if strSimilarity(check_combine, schema[j].lower()) >= 0.7:
                for l in range(k):
                    tag[i+l] = schema[j]
    
    filter_words = [',','the','a','an','for','of','in','on','with','than','and',\
    'is','are','do','does','did','has','have','had','what','how','many','number','get']
    count_f = 0
    for i in range(len(words)):
        # 0th pass eliminate non-sense words
        if words[i] in filter_words:
        	continue
        # 1st pass normalize all the numbers to 00/000, but label them for decoding process
        if tag[i] is not "<nan>":
            continue
        if strIsNum(words[i]):
            tag[i] = '<num>'
    
    	# 2nd pass find field name according to string similarity (field with numerical values)
        if tag[i] is not "<nan>":
            continue
        for j in range(len(schema)):
            if strSimilarity(words[i], schema[j].lower()) >= 0.7:
                tag[i] = schema[j]
    
    	# 3rd pass find values to closest field name in semantic space (name entities)
        #          find query words to closest field name in semantic space
        if tag[i] is not "<nan>":
            continue
        # Nearest neighbor: finding closest clustroid
        the_one = findNearestNeighbor(words[i], schema_vec.keys(), diameter/3.5)
        if the_one is not None:
            tag[i] = schema_vec[the_one]
    
    	# (4th pass NER for un-tagged name entity field string value, tag for schema[0])
    tag_sentence = ' '.join(tag)
    return tag_sentence


def sentTagging_value(query, schema, logic=None):
  ''' Tag each word in a query with one of the three possible tokens:
          1. <nan>
          2. <field: i>
          3. <value: j>
          where i, j are the position according to the schema
      return -- tagged query
                tagged logical form: where <field: 1> equal <value: 1>, select <field: 2>
                field correspondence: a list of the corresponding field names in seuquence [Gold, Nation], 
                                      if <field: 1> is Gold and <field: 2> is Nation
                value correspondence: a list of the corresponding field values in seuquence ['13', 'China'],
                                      corresponding to the field at the same position in field_corr
  '''
    ### construct dictionaries ###
    field2vecField, field2vecValue = fromWordtoVecList(schema)
    field_dict, value_dict = buildDictionary(schema)
    
    ### prepare query, schema, and initialize tag with <nan> ###
    query = query.lower()
    words = basic_tokenizer(query)
    tag = ["<nan>" for x in words]
    # schema0 = ['sum', 'diff','less','greater','argmax', 'argmin'] #, 'mean', ]
    # schema += schema0
    
    ### construct schema_vec, a dict with (key = vector for centroids, value = schema field name)###
    schema_vec = dict()
    for i in range(len(schema)):
        vec_list = field2vecField[schema[i]]
        for j in range(len(vec_list)):
            vec_sign = tuple(vec_list[j])
            #vec_sign = vec_list[j]
            schema_vec[vec_sign] = schema[i]
    
    ### construct schema_vec, a dict with (key = vector for centroids, value = schema field name)###
    schema_vec2 = dict()
    for i in range(len(schema)):
        vec_list2 = field2vecValue[schema[i]]
        for j in range(len(vec_list2)):
            vec_sign2 = tuple(vec_list2[j])
            #vec_sign = vec_list[j]
            schema_vec2[vec_sign] = schema[i]
    
    filter_words = [',','the','a','an','for','of','in','on','with','than','and',\
    'is','are','do','does','did','has','have','had','what','how','many','number','get']
    
    field_corr = []
    value_corr = []
    count = 0
    for i in range(len(words)):
      # 0th pass eliminate non-sense words, label <num> and standby
        if words[i] in filter_words:
            continue
        if strIsNum(words[i]):
            tag[i] = '<num>'
    
      # 1st pass exact match of field name
        if tag[i] is not "<nan>":
            continue
        if words[i] in field_dict:
            if len(field_dict[words[i]]) == 1:
                tag[i] = field_dict[words[i]][0]
            else:
                tag[i] = ';'.join(field_dict[words[i]])
    
      # 2nd pass exact match of field values (CURRENTLY assume one word can NOT be both value and name)
        if tag[i] is not "<nan>":
            continue
        if words[i] in value_dict:
            if len(value_dict[words[i]]) == 1:
                tag[i] = value_dict[words[i]][0]
            else:
                tag[i] = ';'.join(value_dict[words[i]])
    
      # 3rd pass find field name according to string similarity (field with numerical values)
        if tag[i] is not "<nan>":
            continue
        for j in range(len(schema)):
            if strSimilarity(words[i], schema[j].lower()) >= 0.7:
                tag[i] = schema[j]
    
      # 4th pass find values to closest field name in semantic space (name entities)
        #          find query words to closest field name in semantic space
        if tag[i] is not "<nan>":
            continue
        # Nearest neighbor: finding closest clustroid
        kneighbors = findNearestNeighbor(words[i], schema_vec.keys() + schema_vec2.keys(), k = len(schema))
        stat = dict()
        for (vector, weight) in kneighbors:
            if vector in schema_vec:
                candidate = schema_vec[vector]
            else:
                candidate = schema_vec2[vector]
            if candidate not in stat:
                stat[candidate] = 0.0
            stat[candidate] += weight
        def getvalue(item):
            return item[1]
        weightlist = sorted(stat.items(), key=getvalue, reverse=True)
        tag[i] = weightlist[0][0]
        # if the_one is not None:
        #     tag[i] = schema_vec[the_one]
    
    tag_sentence = ' '.join(tag)

    # further change the logical forms to new_logical forms
    return tag_sentence


f_ta = open('../data/rand_train.ta', 'w')
schemas = []
with open('../data/rand_train.fi') as f_fi:
    for sent in f_fi:
        schema = sent.split()
        schemas.append(schema)
with open('../data/rand_train.qu') as f_qu:
    idx = 0
    for query in f_qu:
        print query
        tagged = sentTagging_value(query, schemas[idx])
        print tagged
        f_ta.write(tagged + '\n')
        idx += 1
f_ta.close()