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
import scratch

class Config(object):
    word_dim = 100
    vocabulary_path = './data/vocab10000.from'
    glove_path = '.。/glove.6B/glove.6B.100d.txt'
    max_vocabulary_size = 500000

    embedding_matrix, vocab, word_vector = scratch.generateEmbedMatrix(vocabulary_path, glove_path, max_vocabulary_size)
    # all keys for word_vector are in lowercase

    field2word = {
			'sum': {'value_type': 'string', 
                       'value_range': ['total', 'combined'], 
                       'query_word': ['sum','summation']},
			'diff': {'value_type': 'string', 
                       'value_range': [], 
                       'query_word': ['difference']},
			'less': {'value_type': 'string', 
                       'value_range': [], 
                       'query_word': ['less']},
			'greater': {'value_type': 'string', 
                       'value_range': ['larger'], 
                       'query_word': ['more']},
            'mean': {'value_type': 'string', 
                       'value_range': [], 
                       'query_word': ['mean', 'average']},
			'argmax': {'value_type': 'string', 
                       'value_range': ['maximum','max'],# ,'last', 'previous','before'
                       'query_word': ['most','greatest']}, #,'greatest'
			'argmin': {'value_type': 'string', 
                       'value_range': ['minimum'], #,'first', 'next','after'
                       'query_word': ['least']},
			'Masters': {'value_type':'int',
						'query_word': ['masters']},
			'Country':{'value_type': 'string', 
                     'value_range': ['Brazil', 'Canada', 'Italy', 'France','Nigeria', 'Norway', 'Argentina', 
                     				'South_Korea', 'Israel', 'Australia', 'Iceland', 'Slovenia', 'China', 
                        			'Belgium', 'Germany', 'Poland', 'Spain', 'Ukraine', 'Hungary','Finland', 'Sweden', 
                        			'Vietnam', 'Thailand', 'Switzerland', 'Russia', 'Mexico', 'Egypt', 'Singapore',
                        			'India', 'US', 'Czech', 'Austria', 'UK','Greece', 'Japan'], 
                     'query_word': ['country','countries']},
            'U.S._Open': {'value_type':'int',
						'query_word': ['us','open']},
			'The_Open': {'value_type':'int',
						'query_word': ['the', 'open']},
			'PGA': {'value_type':'int',
						'query_word': ['pga']},
			'Team':{'value_type': 'string', 
                     'value_range': ['Brazil', 'Canada', 'Italy', 'France','Nigeria', 'Norway', 'Argentina', 
                     				'South_Korea', 'Israel', 'Australia', 'Iceland', 'Slovenia', 'China', 
                        			'Belgium', 'Germany', 'Poland', 'Spain', 'Ukraine', 'Hungary','Finland', 'Sweden', 
                        			'Vietnam', 'Thailand', 'Switzerland', 'Russia', 'Mexico', 'Egypt', 'Singapore',
                        			'India', 'US', 'Czech', 'Austria', 'UK','Greece', 'Japan'], 
                     'query_word': ['team','nation']},
            'Years': {'value_type':'int',
						'query_word': ['years']},
			'Wins': {'value_type':'int',
						'query_word': ['wins']},
			'Areas': {'value_type':'int',
						'query_word': ['area','areas']},
			'Prices': {'value_type':'int',
						'query_word': ['prices', 'price']},
            'Swara': {'value_type': 'string', 
                     'value_range': [], 
                     'query_word': ['swara']},
            'Short_name': {'value_type': 'string', 
                     'value_range': [], 
                     'query_word': ['short','name']},
            'Notation': {'value_type': 'string', 
                     'value_range': [], 
                     'query_word': ['notation']},
            'Mnemonic': {'value_type': 'string', 
                     'value_range': [], 
                     'query_word': ['mnemonic']},
			'Player': {'value_type': 'string', 
                     'value_range': [], 
                     'query_word': ['player']},
			'Matches': {'value_type':'int',
						'query_word': ['matches','match']},
			'Innings': {'value_type':'int',
						'query_word': ['innings']},
			'Runs': {'value_type':'int',
						'query_word': ['runs','run']},
			'Average': {'value_type':'int',
						'query_word': ['average']},
			'100s': {'value_type':'int',
						'query_word': ['100s']},
			'50s': {'value_type':'int',
						'query_word': ['50s']},
			'Games_Played': {'value_type':'int',
						'query_word': ['games']},
			'Field_Goals': {'value_type':'int',
						'query_word': ['field', 'goals']},
			'Free_Throws': {'value_type':'int',
						'query_word': ['free', 'throws']},
			'Points': {'value_type':'int',
						'query_word': ['points']},
			'Menteri_Besar': {'value_type': 'string', 
                     'value_range': [], 
                     'query_word': ['menteri','besar']},
			'Party':{'value_type': 'string', 
                     'value_range': ['Conservatives', 'Green', 'Independent', 'Labour'], 
                     'query_word': ['party']},
			'Took_office':{'value_type': 'string', 
                     'value_range': ['Conservatives', 'Green', 'Independent', 'Labour'], 
                     'query_word': ['take', 'took']},
            'Left_office':{'value_type': 'string', 
                     'value_range': [], 
                     'query_word': ['leave', 'left']},
			'Nation':{'value_type': 'string', 
                     'value_range': ['Brazil', 'Canada', 'Italy', 'France','Nigeria', 'Norway', 'Argentina', 
                     				'South_Korea', 'Israel', 'Australia', 'Iceland', 'Slovenia', 'China', 
                        			'Belgium', 'Germany', 'Poland', 'Spain', 'Ukraine', 'Hungary','Finland', 'Sweden', 
                        			'Vietnam', 'Thailand', 'Switzerland', 'Russia', 'Mexico', 'Egypt', 'Singapore',
                        			'India', 'US', 'Czech', 'Austria', 'UK','Greece', 'Japan'], 
                     'query_word': ['country','nation']},
			'Rank': {'value_type':'int',
						'query_word': ['rank', 'ranked']},
			'Gold': {'value_type':'int',
						'query_word': ['gold']},
			'Silver': {'value_type':'int',
						'query_word': ['silver']},
			'Bronze': {'value_type':'int',
						'query_word': ['bronze']},
			'Total': {'value_type':'int',
						'query_word': ['total']},
			'Name': {'value_type':'string',
						'value_range':[],
						'query_word': ['name', 'who']},
			'Position': {'value_type':'string',
						'value_range':['goalkeeper','defender','midfielder'],
						'query_word': ['position']},
			'Year_inducted': {'value_type':'int',
						'query_word': ['year']},
			'Apps': {'value_type':'int',
						'query_word': ['appearance','appearances']},
			'Goals': {'value_type':'int',
						'query_word': ['goal','goals']},
			'Total_Apps': {'value_type':'int',
						'query_word': ['appearance','appearances']},
			'Total_Goals': {'value_type':'int',
						'query_word': ['goal','goals']},
			'League_Apps': {'value_type':'int',
						'query_word': ['appearance','appearances']},
			'League_Goals': {'value_type':'int',
						'query_word': ['goal','goals']},
			'FA_Cup_Apps': {'value_type':'int',
						'query_word': ['appearance','appearances']},
			'FA_Cup_Goals': {'value_type':'int',
						'query_word': ['goal','goals']},
			'State': {'value_type':'string',
						'value_range':['California','Texas','Florida','Louisiana'],
						'query_word': ['state']},
			'No._of_elected': {'value_type':'int',
						'query_word': ['elected']},
			'No._of_candidates': {'value_type':'int',
						'query_word': ['candidate', 'candidates']},
			'Total_no._of_seats_in_Assembly': {'value_type':'int',
						'query_word': ['seat', 'seats']},
			'Year_of_Election': {'value_type':'int',
						'query_word': ['year']},
			'Year': {'value_type':'int',
						'query_word': ['year']},
			'1st_Venue': {'value_type': 'string', 
                       'value_range': ['Chicago','Dallas', 'Florence', 'London', 
                       					'Moscow', 'Paris', 'Philadelphia', 'Prague', 'Seattle', 'Seoul', 'Macau', 'Sydney', 'Tokyo', 
                       					'Toronto', 'Venice', 'Vienna'], 
                       'query_word': ['city', 'cities','venue']}, 
            '2nd_Venue': {'value_type': 'string', 
                       'value_range': ['Amsterdam', 'Beijing', 'Berlin','Chicago','Dallas', 'Florence', 'London', 
                       					'Moscow', 'Paris', 'Philadelphia', 'Prague', 'Seattle', 'Seoul',
                       					'Toronto', 'Venice', 'Vienna'], 
                       'query_word': ['city', 'venue']}, 
            '3rd_Venue': {'value_type': 'string', 
                       'value_range': ['Amsterdam', 'Beijing', 'Berlin','Chicago','Dallas', 'Florence', 'London', 
                       					'Prague', 'Seattle', 'Seoul', 'Macau', 'Sydney', 'Tokyo', 
                       					'Toronto', 'Venice', 'Vienna'], 
                       'query_word': ['city', 'venue']}, 
            '4th_Venue': {'value_type': 'string', 
                       'value_range': ['Amsterdam', 'Beijing', 'Berlin','Chicago',
                       					'Moscow', 'Paris', 'Philadelphia', 'Prague', 'Seattle', 'Seoul', 'Macau', 'Sydney', 'Tokyo', 
                       					'Toronto', 'Venice', 'Vienna'], 
                       'query_word': ['city', 'venue']}, 
            '5th_Venue': {'value_type': 'string', 
                       'value_range': ['Amsterdam', 'Beijing', 'Berlin','Chicago','Dallas', 'Florence', 'London', 
                       					'Moscow', 'Paris', 'Philadelphia', 'Macau', 'Sydney', 'Tokyo', 
                       					'Toronto', 'Venice', 'Vienna'], 
                       'query_word': ['city', 'venue']}, 
            '6th_Venue': {'value_type': 'string', 
                       'value_range': ['Amsterdam', 'Beijing', 'Berlin','Chicago','Dallas', 'Florence', 'London', 
                       					'Moscow', 'Paris', 'Philadelphia', 'Prague', 'Seattle', 'Seoul', 'Macau', 'Sydney', 'Tokyo'], 
                       'query_word': ['city', 'venue']}, 
			'host_city':{'value_type': 'string', 
                       'value_range': ['Amsterdam', 'Beijing', 'Berlin','Chicago','Dallas', 'Florence', 'London', 
                       					'Moscow', 'Paris', 'Philadelphia', 'Prague', 'Seattle', 'Seoul', 'Macau', 'Sydney', 'Tokyo', 
                       					'Toronto', 'Venice', 'Vienna'], 
                       'query_word': ['city', 'cities']}, 
        	'#_participants':{'value_type': 'int', 
                            'query_word': ['participate', 'participates', 'participated', 'participant', 'participants']},
        	'#_audience':{'value_type': 'int', 
                        'query_word': ['audience']},
        	'#_medals':{'value_type': 'int', 
                      'query_word': ['medal', 'medals']},
        	'country_size':{'value_type': 'int', 
                          'query_word': ['large', 'larger', 'largest']}, 
        	'country_gdp':{'value_type': 'int', 
                         'query_word': ['gdp', 'wealth', 'wealthy', 'wealthier']}, 
        	'country_population':{'value_type': 'int', 
                                'query_word': ['population', 'people']}, 
        	'#_duration':{'value_type': 'int', 
                        'query_word': ['long', 'longer', 'longest', 'duration', 'day', 'days']},
        	'year':{'value_type': 'int', 
                  'query_word': ['year', 'years']}, 
        	'country':{'value_type': 'string', 
                     'value_range': ['Brazil', 'Canada', 'Italy', 'France','Nigeria', 'Norway', 'Argentina', 
                     				'South_Korea', 'Israel', 'Australia', 'Iceland', 'Slovenia', 'China', 
                        			'Belgium', 'Germany', 'Poland', 'Spain', 'Ukraine', 'Hungary','Finland', 'Sweden', 
                        			'Vietnam', 'Thailand', 'Switzerland', 'Russia', 'Mexico', 'Egypt', 'Singapore',
                        			'India', 'US', 'Czech', 'Austria', 'UK','Greece', 'Japan'], 
                     'query_word': ['country','countries']}}

# Method 1: find direct match
def buildDictionary(fields):
    query_dict = dict()
    string_dict = dict()
    num_dict = dict()
    for key in fields:
        # build query_dict
        value = fields[key]
        for query_word in value['query_word']:
            if query_word not in query_dict:
                query_dict[query_word] = [] # one value could potentially corresponds to several field names
            query_dict[query_word].append(key)
        if value['value_type'] is 'string':
            # build string_dict
            for word in value['value_range']:
                if word not in string_dict:
                    string_dict[word] = []
                string_dict[word].append(key)
        else:
            # build value_dict
            for num in value['value_range']:
                if num not in num_dict:
                    num_dict[num] = []
                num_dict[num].append(key)
    return query_dict, string_dict, num_dict

def fromWordtoVec(config):
    ### generate the field to vector dictionary from the field to words dictionary
    # field2vec = {"nation": [vec1, vec2],...}    # vec are centroids; 
                 # vec could be weighted by words frequency used for future work
    field2vec = dict()
    for key in config.field2word:
        field2vec[key] = []
        if config.field2word[key]['value_type'] == 'string':
            value_vector = np.array([0.0 for i in range(word_dim)])
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
        query_vector = np.array([0.0 for i in range(word_dim)])
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
        if len(field2vec[key]) == 0:
            # no vectors added, then delete
            del field2vec[key]
    return field2vec

def strSimilarity(word1, word2):
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
    ### Calculate the square of Euclidean distance between two vectors
    dist_square = 0.0
    for i in range(len(vector1)):
        dist_square += (vector1[i] - vector2[i]) * (vector1[i] - vector2[i])
    return dist_square

def spanInSpace(vectorArray):
    ### calculate the half of maximum distance within a list of word vectors
    dist_square = 0.0
    for i in range(len(vectorArray)):
        for j in range(i+1, len(vectorArray)):
            x = EuDisSqu(vectorArray[i], vectorArray[j])
            if dist_square < x:
                dist_square = x
    diameter = dist_square
    return diameter

def findNearestNeighbor(config, word, vectorArray, diameter):
    ### return the word vector which is the nearest neighbor to the query word
    ### return None is word is a new vocab or the distance between word and any of the vecs is larger than diameter
    char_vec = config.word_vector[word]
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
        print idx
        print nearest
        return vectorArray[idx]

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
    config = Config()
    field2vec = fromWordtoVec(config)
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
    
    filter_words = [',','the','a','an','for','of','in','on','with','than','and','is','are','do','does','has']
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
        the_one = findNearestNeighbor(config, words[i], schema_vec.keys(), diameter/2.7)
        if the_one is not None:
            tag[i] = schema_vec[the_one]
    
    	# (4th pass NER for un-tagged name entity field string value, tag for schema[0])
    tag_sentence = ' '.join(tag)
    return tag_sentence
