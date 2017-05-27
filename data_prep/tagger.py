# Generating a tagged sentence for input query
# By Hongyu Xiong, 02/02/2017

import math
import os
import random
import sys
import time
import re
import inspect

import editdistance as ed
import numpy as np
import tag_utils as tu

from nltk.parse import stanford
from nltk import tree
os.environ['STANFORD_PARSER'] = '/Users/richard_xiong/Documents/DeepLearningMaster/deep_parser'
os.environ['STANFORD_MODELS'] = '/Users/richard_xiong/Documents/DeepLearningMaster/deep_parser'
    
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)

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
            if query_word.lower() not in query_dict:
                # one value could potentially corresponds to several field names
                query_dict[query_word.lower()] = [] 
            query_dict[query_word.lower()].append(key)
        if value['value_type'] == 'string':
            # build string_dict: for field values
            for word in value['value_range']:
                if word.lower() not in string_dict:
                    string_dict[word.lower()] = []
                string_dict[word.lower()].append(key)
        # else:
        #     # build num_dict
    return query_dict, string_dict#, num_dict


def buildDictionary880(schema):
    ''' Build a search dictionary based on a given schema in GeoQuery880
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
        if key not in config.geo880_dict:
            continue
        value = config.geo880_dict[key]
        for query_word in value['query_word']:
            if query_word.lower() not in query_dict:
                # one value could potentially corresponds to several field names
                query_dict[query_word.lower()] = [] 
            query_dict[query_word.lower()].append(key)
        if value['value_type'] == 'string':
            # build string_dict: for field values
            for word in value['value_range']:
                if word.lower() not in string_dict:
                    string_dict[word.lower()] = []
                string_dict[word.lower()].append(key)
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
                field2vecString[key].append(value_vector)

        for word in config.field2word[key]['query_word']:
            if word.lower() not in config.word_vector:
                continue
            query_vector = config.word_vector[word.lower()]
            field2vecQuery[key].append(query_vector)
        #if len(field2vec[key]) == 0:
            # no vectors added, then delete
            # del field2vec[key]
    return field2vecQuery, field2vecString


def strSimilarity(word1, word2):
    ''' Measure the similarity based on Edit Distance
    ### Measure how similar word1 is with respect to word2
    '''
    diff = ed.eval(word1.lower(), word2.lower())   #search
    # lcs = LCS(word1, word2)   #search
    length = max(len(word1), len(word2))
    if diff >= length:
        similarity = 0.0
    else:
        similarity = 1.0 * (length-diff) / length
    return similarity


def EuDisSqu(vector1, vector2):
    ''' Calculate the square of Euclidean distance between two vectors
    '''
    dist_square = 0.0
    dist_square += np.dot(vector1, vector1) + np.dot(vector2, vector2) - 2*np.dot(vector1, vector2)
    return np.sqrt(dist_square)


def spanInSpace(vectorArray):
    ''' calculate the maximum distance within a list of word vectors 
    '''
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
                2 the standard deviation within a list of word vectors 
    '''
    dist = 0.0
    std = 0.0
    center = np.array([0.0 for i in range(len(vectorArray[0]))])
    dist_arr = []
    for i in range(len(vectorArray)):
        for j in range(i+1, len(vectorArray)):
            x = EuDisSqu(vectorArray[i], vectorArray[j])
            dist_arr.append(x)
        center += vectorArray[i] / len(vectorArray)
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
    '''verify if the word represent a numerical value
    ''' 
    if not isinstance(s, basestring):
        return 0
    ones = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"}
    tens = {"twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"}
    teens = {"ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", \
            "seventeen", "eighteen", "nineteen"}
    levels = {"hundred", "thousand", "million", "billion", "trillion"}
    if s in ones:
        return True 
    if s in tens:
        return True 
    if s in teens:
        return True 
    if s in levels:
        return True 
    if s.isdigit():
        return True #return int(s)
    # for float value
    try:
        x = float(s)
        return True
    except ValueError:
        return False
    return False


def strIsOrdinal(s):
    '''verify if the word represent a numerical value
    ''' 
    if not isinstance(s, basestring):
        return 0
    ones = {'zeroth','first', 'second', 'third', 'fourth', 'fifth', 'sixth','seventh', 'eighth',\
           'ninth', 'tenth','last','top','bottom'}
    nums = {'0th','1st', '2nd', '3rd', '4th', '5th','6th','7th','8th','9th'}
    if s in ones:
        return True 
    if s in nums:
        return True 
    return False


def basic_tokenizer(sentence):
    '''Very basic tokenizer: split the sentence into a list of tokens.
    ''' 
    words = []
    WORD_SPLIT = re.compile(b"([,!?\")(])")   # get rid of '.':;'
    for space_separated_fragment in sentence.strip().split():
        words.extend(WORD_SPLIT.split(space_separated_fragment))
    return [w for w in words if w]


def sentTagging_knn_semantic(query, schema, logic=None):
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
    ### construct schema_vec, a dict with (key = vector for centroids, value = schema field name)###
    # schema_vec = dict()
    # for i in range(len(schema)):
    #     vec_list = field2vecField[schema[i]]
    #     for j in range(len(vec_list)):
    #         vec_sign = tuple(vec_list[j]) # vec_list[j]
    #         #vec_sign = vec_list[j]
    #         schema_vec[vec_sign] = schema[i]
    
    # ### construct schema_vec, a dict with (key = vector for centroids, value = schema field name)###
    # schema_vec2 = dict()
    # for i in range(len(schema)):
    #     vec_list2 = field2vecValue[schema[i]]
    #     for j in range(len(vec_list2)):
    #         vec_sign2 = tuple(vec_list2[j]) # vec_list2[j]
    #         #vec_sign = vec_list[j]
    #         schema_vec2[vec_sign] = schema[i]
    
    # 4th pass find values to closest field name in semantic space (name entities)
        #          find query words to closest field name in semantic space
        # if tag[i] is not "<nan>":
        #     continue
        # # Nearest neighbor: finding closest clustroid
        # kneighbors = findKNearestNeighbor(words[i], [np.array(x) for x in (schema_vec.keys() + schema_vec2.keys())], k = len(schema))
        # if kneighbors is None:
        #     continue
        # stat = dict()
        # for (vector, weight) in kneighbors:
        #     if tuple(vector) in schema_vec:
        #         candidate = schema_vec[tuple(vector)]
        #     else:
        #         candidate = schema_vec2[tuple(vector)]
        #     if candidate not in stat:
        #         stat[candidate] = 0.0
        #     stat[candidate] += weight
        # def getvalue(item):
        #     return item[1]
        # weightlist = sorted(stat.items(), key=getvalue, reverse=True)
        # tag[i] = weightlist[0][0]
        # if the_one is not None:
        #     tag[i] = schema_vec[the_one]
    
    return None


def sentTagging_value(query, fields, logic=None):
    ''' Tag each word in a query with one of the three possible tokens:
          1. <nan>
          2. <field: i>
          3. <value: j>
          where i, j are the position according to the schema
      return -- tag
                field correspondence: a list of the corresponding field names in seuquence [Gold, Nation], 
                                      if <field: 1> is Gold and <field: 2> is Nation
                value correspondence: a list of the corresponding field values in seuquence ['13', 'China'],
                                      corresponding to the field at the same position in field_corr
                tagged query: how many <field>:0 the <field>:1 <field>:2 <value>:2 won
                tagged logical form: where <field>:1 equal <value>:1, select <field>:2
                      for example:
                .ficorr -- [Gold, Nation]
                .vacorr -- [14, italy;japan]
      deprecate the previous thought:
                .ficorr -- [Gold, Nation, Nation]
                .vacorr -- [14, italy, japan]
    '''
    ### prepare query, schema, and initialize tag with <nan> ###
    query = query.lower()
    words = basic_tokenizer(query)
    schema = fields.split()
    tag = ["<nan>" for x in words]
    
    ### construct dictionaries ###
    config = tu.Config()
    field2vecField, field2vecValue = fromWordtoVecList(schema)
    field_dict, value_dict = buildDictionary(schema)
    # print "value_dict:"
    # print value_dict
    
    filter_words = [',','the','a','an','for','of','in','on','with','than','and',\
    'is','are','do','does','did','has','have','had','what','how','many','get'] #,'number'
    
    ### TAG WITH <field> & <value>
    for i in range(len(words)):
      # 0th pass eliminate non-sense words, label <num> and standby
        if words[i] in filter_words:
            continue
        if strIsNum(words[i]):
            if len(words[i]) == 4:
            # find Year like fields
                the_one = None
                for j in range(len(schema)):
                    if config.field2word[schema[j]]['value_type'] == 'date':
                        # find year_like field
                        the_one = schema[j]
                if the_one == None:
                    tag[i] = '<value>:<num>'
                else: 
                    tag[i] = '<value>:' + the_one
            else: 
                tag[i] = '<value>:<num>'
        if strIsOrdinal(words[i]):
            # tag[i] = '<value>:<order>'
            the_one = None
            for j in range(len(schema)):
                if config.field2word[schema[j]]['value_type'] == 'ordinal':
                    # find ordinal field
                    the_one = schema[j]
            if the_one == None:
                tag[i] = '<value>:<order>'
            else: 
                tag[i] = '<value>:' + the_one
    
      # 1st pass exact match of field name
        if tag[i] is not "<nan>":
            continue
        if words[i] in field_dict:
            tag[i] = '<field>:'
            if len(field_dict[words[i]]) == 1:
                tag[i] += field_dict[words[i]][0]
            else:
                tag[i] += ';'.join(field_dict[words[i]])
            
      # 2nd pass exact match of field values (CURRENTLY assume one word can NOT be both value and name)
      # TO DO: later update to Bloom filter, for strings with high similarity to certain value
        if tag[i] is not "<nan>":
            continue
        if words[i] in value_dict:
            tag[i] = '<value>:'
            if len(value_dict[words[i]]) == 1:
                tag[i] += value_dict[words[i]][0]
            else:
                tag[i] += ';'.join(value_dict[words[i]])
            
      # 3rd pass find field name according to string similarity (field with numerical values)
        if tag[i] is not "<nan>":
            continue
        baseline = 0.55
        for j in range(len(schema)):
            if strSimilarity(words[i], schema[j].lower()) >= baseline:
                tag[i] = schema[j]
                baseline = strSimilarity(words[i], schema[j].lower())
        if tag[i] is not '<nan>':
            tag[i] = '<field>:' + tag[i]
      
    #tag_sentence = ' '.join(tag)
    tag2 = ["<nan>" for x in tag]
    field_corr = []
    value_corr = []
    num_field_position = []
    num_value_position = []
    # count = 0

    ### CORRESPOND <field> with <value>
    for i in range(len(tag2)):
        if tag[i] == "<nan>":
            continue
        reference = tag[i].split(':')
        if reference[0] == '<field>':
            print reference
            if reference[1] in field_corr:
                idx = field_corr.index(reference[1])
                tag2[i] = '<field>:'+str(idx)
            else:
                field_corr.append(reference[1])
                value_corr.append("<nan>")
                idx = len(field_corr) - 1
                tag2[i] = '<field>:'+str(idx)
            
            refers = reference[1].split(';')
            if config.field2word[refers[0]]['value_type'] != 'string':
                num_field_position.append((i, idx))
    for i in range(len(tag2)):
        if tag[i] == "<nan>":
            continue
        reference = tag[i].split(':')
        if reference[0] == '<value>':
            print reference
            # check reference[1] == '<num>'
            if reference[1] in field_corr:
                idx = field_corr.index(reference[1])
                tag2[i] = '<value>:'+str(idx)
                if value_corr[idx] is "<nan>":
                    value_corr[idx] = words[i]
                else:
                    value_corr[idx] += ';'+words[i]

            else:
                if reference[1] == '<num>' or reference[1] == '<order>':
                    num_value_position.append(i)
                    continue
                field_corr.append(reference[1])
                value_corr.append(words[i])
                idx = len(field_corr) - 1
                tag2[i] = '<value>:'+str(idx)
    
    print num_field_position    #[(4, 1), (7, 2)]
    print num_value_position    #[9]
    for i in num_value_position:
        # TO DO: find corresponding field name (dependency tree LCA)
        # find nearest neighbor for this classification model
        idx = None
        nearest_dist = len(tag2)
        for (j,m) in num_field_position:
            if np.abs(j - i) < nearest_dist:
                idx = m
                nearest_dist = np.abs(j - i)
        tag2[i] = '<value>:'+str(idx)
        if idx is not None:
            if value_corr[idx] is "<nan>":
                value_corr[idx] = words[i]
            else:
                value_corr[idx] += ';'+words[i]
                    
    field_corr_sentence = ' '.join(field_corr)
    value_corr_sentence = ' '.join(value_corr)
    tag2_sentence = ' '.join(tag2)

    newquery = [x for x in words]
    for i in range(len(tag2)):
        if tag2[i] == '<nan>':
            continue
        newquery[i] = tag2[i]

    newquery_sentence = ' '.join(newquery)
    if logic is not None:
        tokens = logic.split()
        newlogic = ['<nan>' for x in tokens]
        for i in range(len(tokens)):
            if tokens[i] in field_corr:
                idx = field_corr.index(tokens[i])
                newlogic[i] = '<field>:'+str(idx)
                continue
            for idx in range(len(value_corr)):
                if tokens[i].lower() in value_corr[idx].split(';'):
                    newlogic[i] = '<value>:'+str(idx)
                    continue
            if newlogic[i] == '<nan>':
                newlogic[i] = tokens[i]
        newlogic_sentence = ' '.join(newlogic)
    else:
        newlogic_sentence = None
    # further change the logical forms to new_logical forms
    return tag2_sentence, field_corr_sentence, value_corr_sentence, newquery_sentence, newlogic_sentence


def treeLCAdepth(deptree, value_position, field_position):
    ''' Measure the depth of Lowest Common Ancestor (LCA) in a tree
        arguments --- deptree: dependency tree of a certain parse_query
                      value_position (int): the index of value in original query
                      field_position (int): the index of field in original query
        call function nltk.tree.treeposition_spanning_leaves(start, end), return the sequence of common ancestors
        return --- length of common ancestor sequence, the depth of LCA
    '''
    if value_position > field_position:
        start = field_position
        end = value_position
    else:
        start = value_position
        end = field_position
    treespan = deptree.treeposition_spanning_leaves(start, end)
    return len(treespan)


def sentTagging_tree(parser, query, fields, logic=None):
    ''' Tag each word in a query with one of the three possible tokens:
          BASED ON Dependency Tree
          1. <nan>
          2. <field>:i
          3. <value>:j
          where i, j are the position according to the schema
        argument --- parser: Stanford Dependency parser using NLTK python interface, output a dependency tree 
                             of a parse_query
        return --- tag
                   field correspondence: a list of the corresponding field names in seuquence [Gold, Nation], 
                                        if <field: 1> is Gold and <field: 2> is Nation
                   value correspondence: a list of the corresponding field values in seuquence ['13', 'China'],
                                        corresponding to the field at the same position in field_corr
                   tagged query: how many <field>:0 the <field>:1 <field>:2 <value>:2 won
                   tagged logical form: where <field>:1 equal <value>:1, select <field>:2
                   .ficorr -- [Gold, Nation]
                   .vacorr -- [14, italy;japan]
        deprecate the previous thought:
                   .ficorr -- [Gold, Nation, Nation]
                   .vacorr -- [14, italy, japan]
    '''
    ### prepare query, schema, and initialize tag with <nan> ###
    query = query.lower()
    words = basic_tokenizer(query)
    schema = fields.split()
    tag = ["<nan>" for x in words]
    
    ### construct dictionaries ###
    config = tu.Config()
    #field2vecField, field2vecValue = fromWordtoVecList(schema)
    field_dict, value_dict = buildDictionary(schema)
    # print "value_dict:"
    # print value_dict
    
    filter_words = [',','the','a','an','for','of','in','on','with','than','and',\
    'is','are','do','does','did','has','have','had','what','how','many','get','same','as'] #,'number'
    
    ### TAG WITH <field> & <value>
    for i in range(len(words)):
      # 0th pass eliminate non-sense words, label <num> and standby
        if words[i] in filter_words:
            continue
        if strIsNum(words[i]):
            if len(words[i]) == 4:
            # find Year like fields
                the_one = None
                for j in range(len(schema)):
                    if config.field2word[schema[j]]['value_type'] == 'date':
                        # find year_like field
                        the_one = schema[j]
                if the_one == None:
                    tag[i] = '<value>:<num>'
                else: 
                    tag[i] = '<value>:' + the_one
            else: 
                tag[i] = '<value>:<num>'
        if strIsOrdinal(words[i]):
            # tag[i] = '<value>:<order>'
            the_one = None
            for j in range(len(schema)):
                if config.field2word[schema[j]]['value_type'] == 'ordinal':
                    # find ordinal field
                    the_one = schema[j]
            if the_one == None:
                continue
                #tag[i] = '<value>:<order>'
            else: 
                tag[i] = '<value>:' + the_one
    
      # 1st pass exact match of field name
        if tag[i] is not "<nan>":
            continue
        if words[i] in field_dict:
            tag[i] = '<field>:'
            if len(field_dict[words[i]]) == 1:
                tag[i] += field_dict[words[i]][0]
            else:
                tag[i] += ';'.join(field_dict[words[i]])
            
      # 2nd pass exact match of field values (CURRENTLY assume one word can NOT be both value and name)
      # TO DO: later update to Bloom filter, for strings with high similarity to certain value
        if tag[i] is not "<nan>":
            continue
        if words[i] in value_dict:
            tag[i] = '<value>:'
            if len(value_dict[words[i]]) == 1:
                tag[i] += value_dict[words[i]][0]
            else:
                tag[i] += ';'.join(value_dict[words[i]])
            
      # 3rd pass find field name according to string similarity (field with numerical values)
        if tag[i] is not "<nan>":
            continue
        baseline = 0.55
        for j in range(len(schema)):
            if strSimilarity(words[i], schema[j].lower()) >= baseline:
                tag[i] = schema[j]
                baseline = strSimilarity(words[i], schema[j].lower())
        if tag[i] is not '<nan>':
            tag[i] = '<field>:' + tag[i]
      
    #tag_sentence = ' '.join(tag)
    tag2 = ["<nan>" for x in tag]
    field_corr = []
    value_corr = []
    num_field_position = []
    num_value_position = []
    str_field_position = []
    str_value_position = []
    # count = 0

    ### CORRESPOND <field> with <value>
    for i in range(len(tag2)):
        if tag[i] == "<nan>":
            continue
        reference = tag[i].split(':')
        if reference[0] == '<field>':
            print reference
            if reference[1] in field_corr:
                idx = field_corr.index(reference[1])
                tag2[i] = '<field>:'+str(idx)
            else:
                field_corr.append(reference[1])
                value_corr.append("<nan>")
                idx = len(field_corr) - 1
                tag2[i] = '<field>:'+str(idx)
            
            refers = reference[1].split(';')
            if config.field2word[refers[0]]['value_type'] != 'string':
                num_field_position.append((i, idx))
            else:
                if len(refers) > 1:
                    continue
                str_field_position.append((i, idx))
    for i in range(len(tag2)):
        if tag[i] == "<nan>":
            continue
        reference = tag[i].split(':')
        if reference[0] == '<value>':
            print reference
            # check reference[1] == '<num>'
            if reference[1] in field_corr:
                if len(reference[1].split(';')) > 1:
                    # Overlapping value range for field with 'string' type (e.g. 1st_venue)
                    str_value_position.append(i)
                    continue
                idx = field_corr.index(reference[1])
                tag2[i] = '<value>:'+str(idx)
                if value_corr[idx] is "<nan>":
                    value_corr[idx] = words[i]
                else:
                    value_corr[idx] += ';'+words[i]

            else:
                if reference[1] == '<num>' or reference[1] == '<order>':
                    num_value_position.append(i)
                    continue
                elif len(reference[1].split(';')) > 1:
                    # Overlapping value range for field with 'string' type (e.g. 1st_venue)
                    str_value_position.append(i)
                    continue
                field_corr.append(reference[1])
                value_corr.append(words[i])
                idx = len(field_corr) - 1
                tag2[i] = '<value>:'+str(idx)
    
    print num_field_position    #[(4, 1), (7, 2)]
    print num_value_position    #[9]
    print str_field_position    #[(4, 1), (7, 2)]
    print str_value_position    #[9]
    
    # (dependency tree LCA) build parse query for str_position and num_position
    if len(num_value_position) > 0:
        parsequery_num = [x for x in words]
        for (j,m) in num_field_position:
            parsequery_num[j] = '<field:'+str(m)+'>'
        #for i in num_value_position:
          #  parsequery_num[i] = '<value>'
        dependency_tree_num = parser.raw_parse_sents(('Hello, My name is Melroy', ' '.join(parsequery_num)))
        # find corresponding field (dependency tree LCA)
        # only take dependency_tree[1]
        for i in num_value_position:
            # find largest ancestor depth for (value, field) pair in the tree
            idx = None
            largest_depth = 0
            for (j,m) in num_field_position:
                depth = treeLCAdepth(dependency_tree_num[1], i, j)
                if depth > largest_depth:
                    idx = m
                    largest_depth = depth
            if idx is not None:
                tag2[i] = '<value>:'+str(idx)
                if value_corr[idx] is "<nan>":
                    value_corr[idx] = words[i]
                else:
                    value_corr[idx] += ';'+words[i]
    
    ### 0508 newly added, to deal with overlap range in string-type field                
    if len(str_value_position) > 0:    
        parsequery_str = [x for x in words]
        for (j,m) in str_field_position:
            parsequery_str[j] = '<field:'+str(m)+'>'
        #for i in str_value_position:
          #  parsequery_str[i] = '<value>'
        dependency_tree_str = parser.raw_parse_sents(('Hello, My name is Melroy', ' '.join(parsequery_str)))
        for i in str_value_position:
            # find largest ancestor depth for (value, field) pair in the tree
            idx = None
            largest_depth = 0
            for (j,m) in str_field_position:
                depth = treeLCAdepth(dependency_tree_str[1], i, j)
                if depth > largest_depth:
                    idx = m
                    largest_depth = depth
            if idx is not None:
                tag2[i] = '<value>:'+str(idx)
                if value_corr[idx] is "<nan>":
                    value_corr[idx] = words[i]
                else:
                    value_corr[idx] += ';'+words[i]

    field_corr_sentence = ' '.join(field_corr)
    value_corr_sentence = ' '.join(value_corr)
    tag2_sentence = ' '.join(tag2)

    newquery = [x for x in words]
    for i in range(len(tag2)):
        if tag2[i] == '<nan>':
            continue
        newquery[i] = tag2[i]

    newquery_sentence = ' '.join(newquery)
    if logic is not None:
        tokens = logic.split()
        newlogic = ['<nan>' for x in tokens]
        for i in range(len(tokens)):
            if tokens[i] in field_corr:
                idx = field_corr.index(tokens[i])
                newlogic[i] = '<field>:'+str(idx)
                continue
            for idx in range(len(value_corr)):
                if tokens[i].lower() in value_corr[idx].split(';'):
                    newlogic[i] = '<value>:'+str(idx)
                    continue
            if newlogic[i] == '<nan>':
                newlogic[i] = tokens[i]
        newlogic_sentence = ' '.join(newlogic)
    else:
        newlogic_sentence = None
    # further change the logical forms to new_logical forms
    return tag2_sentence, field_corr_sentence, value_corr_sentence, newquery_sentence, newlogic_sentence


def sentTagging_tree880(parser, query, fields, logic=None):
    ''' Tag each word in a query with one of the three possible tokens:
          BASED ON Dependency Tree
          1. <nan>
          2. <field>:i
          3. <value>:j
          where i, j are the position according to the schema
        argument --- parser: Stanford Dependency parser using NLTK python interface, output a dependency tree 
                             of a parse_query
        return --- tag
                   field correspondence: a list of the corresponding field names in seuquence [Gold, Nation], 
                                        if <field: 1> is Gold and <field: 2> is Nation
                   value correspondence: a list of the corresponding field values in seuquence ['13', 'China'],
                                        corresponding to the field at the same position in field_corr
                   tagged query: how many <field>:0 the <field>:1 <field>:2 <value>:2 won
                   tagged logical form: where <field>:1 equal <value>:1, select <field>:2
                   .ficorr -- [Gold, Nation]
                   .vacorr -- [14, italy;japan]
        deprecate the previous thought:
                   .ficorr -- [Gold, Nation, Nation]
                   .vacorr -- [14, italy, japan]
    '''
    ### prepare query, schema, and initialize tag with <nan> ###
    query = query.lower()
    words = basic_tokenizer(query)
    schema = fields.split()
    tag = ["<nan>" for x in words]
    
    ### construct dictionaries ###
    config = tu.Config()
    #field2vecField, field2vecValue = fromWordtoVecList(schema)
    field_dict, value_dict = buildDictionary880(schema)
    
    filter_words = [',','the','a','an','in','for','of','on','with','than','and',\
    'is','are','do','does','did','has','have','had','what','how','many','get','same','as'] #,'number'
    
    ### TAG WITH <field> & <value>
    for i in range(len(words)):
      # 0th pass eliminate non-sense words, label <num> and standby
        if words[i] in filter_words:
            continue
        if strIsNum(words[i]):
            if len(words[i]) == 4:
            # find Year like fields
                the_one = None
                for j in range(len(schema)):
                    if config.geo880_dict[schema[j]]['value_type'] == 'date':
                        # find year_like field
                        the_one = schema[j]
                if the_one == None:
                    tag[i] = '<value>:<num>'
                else: 
                    tag[i] = '<value>:' + the_one
            else: 
                tag[i] = '<value>:<num>'
        if strIsOrdinal(words[i]):
            # tag[i] = '<value>:<order>'
            the_one = None
            for j in range(len(schema)):
                if config.geo880_dict[schema[j]]['value_type'] == 'ordinal':
                    # find ordinal field
                    the_one = schema[j]
            if the_one == None:
                continue
                #tag[i] = '<value>:<order>'
            else: 
                tag[i] = '<value>:' + the_one
    
      # 1st pass exact match of field name
        if tag[i] is not "<nan>":
            continue
        if words[i] in field_dict:
            tag[i] = '<field>:'
            if len(field_dict[words[i]]) == 1:
                tag[i] += field_dict[words[i]][0]
            else:
                tag[i] += ';'.join(field_dict[words[i]])
            
      # 2nd pass exact match of field values (CURRENTLY assume one word can NOT be both value and name)
      # TO DO: later update to Bloom filter, for strings with high similarity to certain value
        if tag[i] is not "<nan>":
            continue
        if words[i] in value_dict:
            tag[i] = '<value>:'
            if len(value_dict[words[i]]) == 1:
                tag[i] += value_dict[words[i]][0]
            else:
                tag[i] += ';'.join(value_dict[words[i]])
            
      # 3rd pass find field name according to string similarity (field with numerical values)
        if tag[i] is not "<nan>":
            continue
        baseline = 0.75
        for j in range(len(schema)):
            if strSimilarity(words[i], schema[j].lower()) >= baseline:
                tag[i] = schema[j]
                baseline = strSimilarity(words[i], schema[j].lower())
        if tag[i] is not '<nan>':
            tag[i] = '<field>:' + tag[i]
      
    #tag_sentence = ' '.join(tag)
    tag2 = ["<nan>" for x in tag]
    field_corr = []
    value_corr = []
    num_field_position = []
    num_value_position = []
    str_field_position = []
    str_value_position = []
    # count = 0

    ### CORRESPOND <field> with <value>
    for i in range(len(tag2)):
        if tag[i] == "<nan>":
            continue
        reference = tag[i].split(':')
        if reference[0] == '<field>':
            print reference
            if reference[1] in field_corr:
                idx = field_corr.index(reference[1])
                tag2[i] = '<field>:'+str(idx)
            else:
                field_corr.append(reference[1])
                value_corr.append("<nan>")
                idx = len(field_corr) - 1
                tag2[i] = '<field>:'+str(idx)
            
            refers = reference[1].split(';')
            if config.geo880_dict[refers[0]]['value_type'] != 'string':
                num_field_position.append((i, idx))
            else:
                if len(refers) > 1:   #multiple fields, did not add into
                    continue
                str_field_position.append((i, idx))
    moderation = set()  #0526 newly added
    for i in range(len(tag2)):
        if tag[i] == "<nan>":
            continue
        reference = tag[i].split(':')
        if reference[0] == '<value>':
            print reference
            # check reference[1] == '<num>'
            if reference[1] in field_corr:
                if len(reference[1].split(';')) > 1:
                    # Overlapping value range for field with 'string' type (e.g. 1st_venue)
                    str_value_position.append(i)
                    continue
                idx = field_corr.index(reference[1])
                tag2[i] = '<value>:'+str(idx)
                if value_corr[idx] is "<nan>":
                    value_corr[idx] = words[i]
                else:
                    value_corr[idx] += ';'+words[i]

            else:
                if reference[1] == '<num>' or reference[1] == '<order>':
                    num_value_position.append(i)
                    continue
                elif len(reference[1].split(';')) > 1:
                    # Overlapping value range for field with 'string' type (e.g. 1st_venue)
                    str_value_position.append(i)
                    #0526 newly added
                    for refer in reference[1].split(';'):
                        moderation.add(refer)
                    continue
                field_corr.append(reference[1])
                value_corr.append(words[i])
                idx = len(field_corr) - 1
                tag2[i] = '<value>:'+str(idx)
    
    print num_field_position    #[(4, 1), (7, 2)]
    print num_value_position    #[9]
    print str_field_position    #[(4, 1), (7, 2)]
    print str_value_position    #[9]
    print moderation
    # (dependency tree LCA) build parse query for str_position and num_position
    if len(num_value_position) > 0:
        parsequery_num = [x for x in words]
        for (j,m) in num_field_position:
            parsequery_num[j] = '<field:'+str(m)+'>'
        #for i in num_value_position:
          #  parsequery_num[i] = '<value>'
        dependency_tree_num = parser.raw_parse_sents(('Hello, My name is Melroy', ' '.join(parsequery_num)))
        # find corresponding field (dependency tree LCA)
        # only take dependency_tree[1]
        for i in num_value_position:
            # find largest ancestor depth for (value, field) pair in the tree
            idx = None
            largest_depth = 0
            for (j,m) in num_field_position:
                depth = treeLCAdepth(dependency_tree_num[1], i, j)
                if depth > largest_depth:
                    idx = m
                    largest_depth = depth
            if idx is not None:
                tag2[i] = '<value>:'+str(idx)
                if value_corr[idx] is "<nan>":
                    value_corr[idx] = words[i]
                else:
                    value_corr[idx] += ';'+words[i]
    
    ### 0508 newly added, to deal with overlap range in string-type field                
    if len(str_value_position) > 0:    
        parsequery_str = [x for x in words]
        for (j,m) in str_field_position:
            parsequery_str[j] = '<field:'+str(m)+'>'
        #for i in str_value_position:
          #  parsequery_str[i] = '<value>'
        dependency_tree_str = parser.raw_parse_sents(('Hello, My name is Melroy', ' '.join(parsequery_str)))
        for i in str_value_position:
            # find largest ancestor depth for (value, field) pair in the tree
            idx = None
            largest_depth = 0
            for (j,m) in str_field_position:
                if field_corr[m] not in moderation: #0526 newly added
                    continue
                depth = treeLCAdepth(dependency_tree_str[1], i, j)
                if depth > largest_depth:
                    idx = m
                    largest_depth = depth
            if idx is not None:
                tag2[i] = '<value>:'+str(idx)
                if value_corr[idx] is "<nan>":
                    value_corr[idx] = words[i]
                else:
                    value_corr[idx] += ';'+words[i]
            else: #0526 newly added
                #new correspondence found
                reference = tag[i].split(':')
                refers = reference[1].split(';')
                if refers[0] in field_corr:
                    idx = field_corr.index(refers[0])
                    if value_corr[idx] is "<nan>":
                        value_corr[idx] = words[i]
                    else:
                        value_corr[idx] += ';'+words[i]
                else:
                    field_corr.append(refers[0])
                    idx = len(field_corr) - 1
                    value_corr.append(words[i])
                tag2[i] = '<value>:'+str(idx)

    field_corr_sentence = ' '.join(field_corr)
    value_corr_sentence = ' '.join(value_corr)
    tag2_sentence = ' '.join(tag2)

    newquery = [x for x in words]
    for i in range(len(tag2)):
        if tag2[i] == '<nan>':
            continue
        newquery[i] = tag2[i]

    newquery_sentence = ' '.join(newquery)
    if logic is not None:
        tokens = logic.split()
        newlogic = ['<nan>' for x in tokens]
        for i in range(len(tokens)):
            if tokens[i] in field_corr:
                idx = field_corr.index(tokens[i])
                newlogic[i] = '<field>:'+str(idx)
                continue
            for idx in range(len(value_corr)):
                if tokens[i].lower() in value_corr[idx].split(';'):
                    newlogic[i] = '<value>:'+str(idx)
                    continue
            if newlogic[i] == '<nan>':
                newlogic[i] = tokens[i]
        newlogic_sentence = ' '.join(newlogic)
    else:
        newlogic_sentence = None
    # further change the logical forms to new_logical forms
    return tag2_sentence, field_corr_sentence, value_corr_sentence, newquery_sentence, newlogic_sentence


def templateToLogicalfrom(field_corr_sentence, value_corr_sentence, newlogic_sentence):
    ''' given newlogical template with field_corr and value_corr
        output the declarative logical form
    '''
    
    logic = []
    newlogical = newlogic_sentence.split()
    field_corr = field_corr_sentence.split()
    value_corr = value_corr_sentence.split()
    # go over each token in newlogical and replace with corresponding field name
    checkcount = dict()   # used for keep track field with multiple values appeared
    for i in range(len(newlogical)):
        reference = newlogical[i].split(':')
        if len(reference) == 1:
            logic.append(newlogical[i])
            continue
        if reference[0] == '<field>':
            print reference
            idx = int(reference[1])
            logic.append(field_corr[idx])
            if field_corr[idx] not in checkcount:
                checkcount[field_corr[idx]] = 0
            else: 
                checkcount[field_corr[idx]] += 1
        else:
            print reference
            idx = int(reference[1])
            # check whether value_corr[idx] is single value
            value_choice = value_corr[idx].split(';')
            if len(value_choice) == 1:
                logic.append(value_corr[idx])
                continue
            pick = checkcount[field_corr[idx]]
            logic.append(value_choice[pick])
    #print logic
    logic_sentence = ' '.join(logic)
    return logic_sentence


def main2():
    ''' convert .lox back to .lo, the final stage of NLIDB pipeline
              (template)->(declarative)
        using for the final accuracy check
    '''
    f_lo = open(parentdir + '/evaluation/forms_train.lo', 'w')
    with open(parentdir + '/evaluation/rand_train.ficorr') as f_ficorr:
        with open(parentdir + '/evaluation/rand_train.vacorr') as f_vacorr:
            with open(parentdir + '/data/rand_train.lox') as f_lox:   # final test change to logicalForm.out
                field_corr, value_corr, newlogical = f_ficorr.readline(), f_vacorr.readline(), f_lox.readline()
                idx = 0
                while field_corr and value_corr and newlogical:
                    idx += 1
                    # if idx == 20:
                    #     break
                    print '### example: %d ###' % idx
                    logic = templateToLogicalfrom(field_corr, value_corr, newlogical)
                    print newlogical
                    print field_corr
                    print value_corr
                    print logic
                    f_lo.write(logic + '\n')
                    field_corr, value_corr, newlogical = f_ficorr.readline(), f_vacorr.readline(), f_lox.readline()
    f_lo.close()
    return


def main1():
    ''' process data, from .qu, .lo, and .fi
        to .ta, .lox, .qux
        and .ficorr, .vacorr
    '''
    parser = stanford.StanfordParser(model_path='/Users/richard_xiong/Documents/DeepLearningMaster/deep_parser/englishPCFG.ser.gz')
    f_ta = open(parentdir + '/data/rand_train.ta', 'w')
    f_lox = open(parentdir + '/data/rand_train.lox', 'w')
    f_qux = open(parentdir + '/data/rand_train.qux', 'w')
    f_ficorr = open(parentdir + '/evaluation/rand_train.ficorr', 'w')
    f_vacorr = open(parentdir + '/evaluation/rand_train.vacorr', 'w')
    with open(parentdir + '/data/rand_train.fi') as f_fi:
        with open(parentdir + '/data/rand_train.qu') as f_qu:
            with open(parentdir + '/data/rand_train.lo') as f_lo:
                schema, query, logic = f_fi.readline(), f_qu.readline(), f_lo.readline()
                idx = 0
                while schema and query and logic:
                    idx += 1
                    # if idx == 15:
                    #     break
                    print '### example: %d ###' % idx
                    #tagged2, field_corr, value_corr, newquery, newlogical = sentTagging_value(query, schema, logic)
                    print schema, query,logic
                    tagged2, field_corr, value_corr, newquery, newlogical = sentTagging_tree(parser, query, schema, logic)
                    print field_corr
                    print value_corr
                    print tagged2
                    print newquery
                    print newlogical
                    print '\n'
                    f_qux.write(newquery + '\n')
                    f_lox.write(newlogical + '\n')
                    f_ficorr.write(field_corr +'\n')
                    f_vacorr.write(value_corr + '\n')
                    f_ta.write(tagged2 + '\n')
                    schema, query, logic = f_fi.readline(), f_qu.readline(), f_lo.readline()
    f_ta.close()
    f_lox.close()
    f_qux.close()
    f_vacorr.close()
    f_ficorr.close()
    return

def main880():
    ''' process data, from .qu, .lo, and .fi
        to .ta, .lox, .qux
        and .ficorr, .vacorr
    '''
    parser = stanford.StanfordParser(model_path='/Users/richard_xiong/Documents/DeepLearningMaster/deep_parser/englishPCFG.ser.gz')
    schema = "state area population population_density city lake mountain height river length border_state major_city major_lake major_river highest_point \
            highest_elevation lowest_point lowest_elevation capital country"
    f_ta = open(parentdir + '/data_prep/GeoQuery/geo880.ta', 'w')
    f_lox = open(parentdir + '/data_prep/GeoQuery/geo880.lox', 'w')
    f_qux = open(parentdir + '/data_prep/GeoQuery/geo880.qux', 'w')
    f_ficorr = open(parentdir + '/evaluation/Geo/geo880.ficorr', 'w')
    f_vacorr = open(parentdir + '/evaluation/Geo/geo880.vacorr', 'w')
    with open(parentdir + '/data_prep/GeoQuery/geo880.qu') as f_qu:
        with open(parentdir + '/data_prep/GeoQuery/geo880.lo') as f_lo:
            query, logic = f_qu.readline(), f_lo.readline()
            idx = 0
            while query and logic:
                idx += 1
                # if idx < 40:
                #     continue
                # if idx == 80:
                #     break
                print '### example: %d ###' % idx
                #tagged2, field_corr, value_corr, newquery, newlogical = sentTagging_value(query, schema, logic)
                print query
                print logic
                tagged2, field_corr, value_corr, newquery, newlogical = sentTagging_tree880(parser, query, schema, logic)
                print field_corr
                print value_corr
                print tagged2
                print newquery
                print newlogical
                print '\n'
                f_qux.write(newquery + '\n')
                f_lox.write(newlogical + '\n')
                f_ficorr.write(field_corr +'\n')
                f_vacorr.write(value_corr + '\n')
                f_ta.write(tagged2 + '\n')
                query, logic = f_qu.readline(), f_lo.readline()
    f_ta.close()
    f_lox.close()
    f_qux.close()
    f_vacorr.close()
    f_ficorr.close()
    return

#main1()
#main2()
main880()