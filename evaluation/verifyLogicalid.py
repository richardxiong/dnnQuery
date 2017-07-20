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

truth_path = "../dataover/except_%s" % subset
output_path = "../evaluation/dataover/except_%s" % subset


# train_truth = truth_path + "/rand_train.lox"
# train_output = output_path + "/logicalTemp_train.out"
# dev_truth = truth_path + "/rand_dev.lox"
# dev_output = output_path + "/logicalTemp_dev.out"
test_truth = truth_path + "/%s_test.lox" % subset
test_output = output_path + "/%s_test.out" % subset

geo_truth = truth_path + "/%s_train.lox" % subset
geo_output = output_path + "/%s_train.out" % subset

# train_truth = truth_path + "/rand_train.lo"
# train_output = output_path + "/forms_train.lo"
# dev_truth = truth_path + "/rand_dev.lo"
# dev_output = output_path + "/forms_dev.lo"
# test_truth = truth_path + "/rand_test.lo"
# test_output = output_path + "/forms_test.lo"
prime = ['where', 'select', 'max', 'min', 'equal', 'less', 'greater', 'neq', 'ng', 'nl', \
         'avg', 'count', 'sum', 'between', 'and', 'or', '<field>:0', '<field>:1', '<field>:2', \
         '<field>:3', '<value>:0', '<value>:1', '<value>:2', '<value>:3','<count>'
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
            ## case 1, different sequence
            wordsList = line.split(' ')
            if len(wordsList) > 8 and wordsList[5] == 'and':
                temp = wordsList[2]
                wordsList[2] = wordsList[6]
                wordsList[6] = temp
                temp = wordsList[4]
                wordsList[4] = wordsList[8]
                wordsList[8] = temp
                if ' '.join(wordsList).lower() == truth[index]:
                    correct += 1
                    continue
            ## case 2, replace unseen vocabulary
            truthlist = truth[index].split(' ')
            wordsList = line.split(' ')
            if len(truthlist) == len(wordsList):
                for i in range(len(truthlist)):
                    if truthlist[i] not in prime:
                        truthlist[i] = wordsList[i]
                if line.lower() == ' '.join(truthlist):
                    correct += 1
                    continue
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
            ## case 1, different sequence
            wordsList = line.split(' ')
            if len(wordsList) > 8 and wordsList[5] == 'and':
                temp = wordsList[2]
                wordsList[2] = wordsList[6]
                wordsList[6] = temp
                temp = wordsList[4]
                wordsList[4] = wordsList[8]
                wordsList[8] = temp
                if ' '.join(wordsList).lower() == truth[index]:
                    correct += 1
                    continue
            ## case 2, replace unseen vocabulary
            truthlist = truth[index].split(' ')
            wordsList = line.split(' ')
            if len(truthlist) == len(wordsList):
                for i in range(len(truthlist)):
                    if truthlist[i] not in prime:
                        truthlist[i] = wordsList[i]
                if line.lower() == ' '.join(truthlist):
                    correct += 1
                    continue
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
