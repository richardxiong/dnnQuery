'''
A simple script for logical form verification
'''
import re
import numpy as np
import editdistance as ed
#from data_utils import basic_tokenizer
# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"([,!?\"')(])")   # get rid of '.':;
_DIGIT_RE = re.compile(br"\d")

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

'''
input files
'''
subset = 'restaurants'

# truth_path = "../one_learn/ba_%s" % subset
# truth_path = "../one_learn/ca_%s" % subset
# truth_path = "../one_learn/ho_%s" % subset
# truth_path = "../one_learn/re_%s" % subset
# truth_path = "../two_learn/ba_ca_%s" % subset
# truth_path = "../two_learn/ba_ho_%s" % subset
# truth_path = "../two_learn/ba_re_%s" % subset
# truth_path = "../two_learn/re_ca_%s" % subset
# truth_path = "../two_learn/re_ho_%s" % subset
# truth_path = "../two_learn/ca_ho_%s" % subset
truth_path = "../three_learn/ba_ca_ho_%s" % subset
# truth_path = "../three_learn/ba_ca_re_%s" % subset
# truth_path = "../three_learn/ba_ho_re_%s" % subset
# truth_path = "../three_learn/ca_ho_re_%s" % subset


# output_path = "../evaluation/one_learn/ba_%s" % subset
# output_path = "../evaluation/one_learn/ca_%s" % subset
# output_path = "../evaluation/one_learn/ho_%s" % subset
# output_path = "../evaluation/one_learn/re_%s" % subset
# output_path = "../evaluation/two_learn/ba_ca_%s" % subset
# output_path = "../evaluation/two_learn/ba_ho_%s" % subset
# output_path = "../evaluation/two_learn/ba_re_%s" % subset
# output_path = "../evaluation/two_learn/re_ca_%s" % subset
# output_path = "../evaluation/two_learn/re_ho_%s" % subset
# output_path = "../evaluation/two_learn/ca_ho_%s" % subset
output_path = "../evaluation/three_learn/ba_ca_ho_%s" % subset
# output_path = "../evaluation/three_learn/ba_ca_re_%s" % subset
# output_path = "../evaluation/three_learn/ba_ho_re_%s" % subset
# output_path = "../evaluation/three_learn/ca_ho_re_%s" % subset


test_truth = truth_path + "/%s_test.lox" % subset
test_output = output_path + "/%s_test.out" % subset

geo_truth = truth_path + "/%s_train.lox" % subset
geo_output = output_path + "/%s_train.out" % subset

prime = ['where', 'select', 'max', 'min', 'equal', 'less', 'greater', 'neq', 'ng', 'nl', \
         'avg', 'count', 'sum', 'between', 'and', 'or', '<field>:0', '<field>:1', '<field>:2', \
         '<field>:3', '<value>:0', '<value>:1', '<value>:2', '<value>:3','<count>','true'
        ]

correct = 0
truth = []
with open(test_truth) as infile:
    for line0 in infile:
        # wordsList = basic_tokenizer(line)
        line = line0.strip()
        '''1. Strip Parentheses'''
        line = line.replace('( ', '')
        line = line.replace(' )', '')
        '''2. Strip value types'''
        wordsList = line.split(' ')
        wordsList = [x[:x.find(':')]+x[x.rfind(':'):] for x in wordsList]
        line = ' '.join(wordsList)
        # for i in range(len(wordsList)):
        #     wordsList[i] = _DIGIT_RE.sub(b"0", wordsList[i])
        # truth.append(' '.join(wordsList))
        truth.append(line.lower())
index = 0
with open(test_output) as infile:
    for line0 in infile:
        line = line0.strip()
        '''1. Strip Parentheses'''
        line = line.replace('( ', '')
        line = line.replace(' )', '')
        '''2. Strip value types'''
        wordsList = line.split(' ')
        wordsList = [x[:x.find(':')]+x[x.rfind(':'):] for x in wordsList]
        line = ' '.join(wordsList)
        length = len(truth[index])
        if line.lower()[:length] == truth[index]:
            correct += 1
        else:
            # compare line with all possible forms, and choose the most similar one
            # case 1, different sequence
            #wordsList = line.split(' ')
            if len(wordsList) > 8 and wordsList[5] == 'and':
                newwordsList = [x for x in wordsList]
                newwordsList[2] = wordsList[6]
                newwordsList[6] = wordsList[2]
                newwordsList[4] = wordsList[8]
                newwordsList[8] = wordsList[4]
                newline = ' '.join(newwordsList)
                if newline.lower() == truth[index]:
                    correct += 1
                    index += 1
                    continue
            # case 2, replace unseen vocabulary
            # truthlist = truth[index].split(' ')
            # if len(wordsList) == len(truthlist):
            #     newwordsList = [x for x in wordsList]
            #     for i in range(len(wordsList)):
            #         if newwordsList[i].lower() == '_unk':
            #             newwordsList[i] = truthlist[i]
            #     newline = ' '.join(newwordsList)
            #     if newline.lower() == ' '.join(truthlist):
            #         correct += 1
            #         index += 1
            #         continue
            print "wrong examples: %d" %(index + 1)
            print truth[index]
            print line.lower()
            # dists = np.ones(len(loTemp))
            # for j in range(len(loTemp)):
            #     dists[j] = strSimilarity(line, loTemp[j])
            # newline = loTemp[np.argmax(dists)]
            # if newline == truth[index]:
            #     correct += 1
            # else:
            #     print "wrong examples: %d" %(index + 1)
            #     print truth[index]
            #     print line.lower()
        index += 1
print "test accuracy: " + str(correct * 1.0 / len(truth))

correct = 0
truth = []
with open(geo_truth) as infile:
    for line0 in infile:
        # wordsList = basic_tokenizer(line)
        line = line0.strip()
        '''1. Strip Parentheses'''
        line = line.replace('( ', '')
        line = line.replace(' )', '')
        '''2. Strip value types'''
        wordsList = line.split(' ')
        wordsList = [x[:x.find(':')]+x[x.rfind(':'):] for x in wordsList]
        line = ' '.join(wordsList)
        # for i in range(len(wordsList)):
        #     wordsList[i] = _DIGIT_RE.sub(b"0", wordsList[i])
        # truth.append(' '.join(wordsList))
        truth.append(line.lower())
index = 0
with open(geo_output) as infile:
    for line0 in infile:
        line = line0.strip()
        '''1. Strip Parentheses'''
        line = line.replace('( ', '')
        line = line.replace(' )', '')
        '''2. Strip value types'''
        wordsList = line.split(' ')
        wordsList = [x[:x.find(':')]+x[x.rfind(':'):] for x in wordsList]
        line = ' '.join(wordsList)
        length = len(truth[index])
        if line.lower()[:length] == truth[index]:
            correct += 1
        else:
            # compare line with all possible forms, and choose the most similar one
            # case 1, different sequence
            #wordsList = line.split(' ')
            if len(wordsList) > 8 and wordsList[5] == 'and':
                newwordsList = [x for x in wordsList]
                newwordsList[2] = wordsList[6]
                newwordsList[6] = wordsList[2]
                newwordsList[4] = wordsList[8]
                newwordsList[8] = wordsList[4]
                newline = ' '.join(newwordsList)
                if newline.lower() == truth[index]:
                    correct += 1
                    index += 1
                    continue
            # case 2, replace unseen vocabulary
            # truthlist = truth[index].split(' ')
            # if len(wordsList) == len(truthlist):
            #     newwordsList = [x for x in wordsList]
            #     for i in range(len(wordsList)):
            #         if newwordsList[i].lower() == '_unk':
            #             newwordsList[i] = truthlist[i]
            #     newline = ' '.join(newwordsList)
            #     if newline.lower() == ' '.join(truthlist):
            #         correct += 1
            #         index += 1
            #         continue
            print "wrong examples: %d" %(index + 1)
            print truth[index]
            print line.lower()
            # dists = np.ones(len(loTemp))
            # for j in range(len(loTemp)):
            #     dists[j] = strSimilarity(line, loTemp[j])
            # newline = loTemp[np.argmax(dists)]
            # if newline == truth[index]:
            #     correct += 1
            # else:
            #     print "wrong examples: %d" %(index + 1)
            #     print truth[index]
            #     print line.lower()
        index += 1
print "train accuracy: " + str(correct * 1.0 / len(truth))
