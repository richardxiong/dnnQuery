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
subset = socialnetwork

truth_path = "../data/Overnight/except_%s" % subset
output_path = "../evaluation/Overnight/except_%s" % subset


# train_truth = truth_path + "/rand_train.lox"
# train_output = output_path + "/logicalTemp_train.out"
# dev_truth = truth_path + "/rand_dev.lox"
# dev_output = output_path + "/logicalTemp_dev.out"
test_truth = truth_path + "/except_%s_test.lox" % subset
test_output = output_path + "/except_%s_test.out" % subset

geo_truth = truth_path + "/except_%s_train.lox" % subset
geo_output = output_path + "/except_%s_train.out" % subset

# train_truth = truth_path + "/rand_train.lo"
# train_output = output_path + "/forms_train.lo"
# dev_truth = truth_path + "/rand_dev.lo"
# dev_output = output_path + "/forms_dev.lo"
# test_truth = truth_path + "/rand_test.lo"
# test_output = output_path + "/forms_test.lo"

# loTemp = ['sum <field>:0','avg <field>:0','select <field>:0 argmax <field>:1','select <field>:0 argmin <field>:1',\
#           'select <field>:0 argmax <field>:0','select <field>:0 argmin <field>:0','select <field>:0 where <field>:1 equal <value>:1',\
#           'select <field>:0 where <field>:1 less <value>:1','select <field>:0 where <field>:1 greater <value>:1',\
#           'count <field>:0 where <field>:1 equal <value>:1','count <field>:0 where <field>:1 less <value>:1',\
#           'count <field>:0 where <field>:1 greater <value>:1','select <field>:0 prev <field>:0 equal <value>:0',\
#           'select <field>:0 next <field>:0 equal <value>:0','select <field>:0 argmin <field>:1 where <field>:2 equal <value>:2',\
#           'select <field>:0 argmin <field>:1 where <field>:2 less <value>:2',
#           'select <field>:0 argmin <field>:1 where <field>:2 greater <value>:2',
#           'select <field>:0 argmax <field>:1 where <field>:2 equal <value>:2',
#           'select <field>:0 argmax <field>:1 where <field>:2 less <value>:2',
#           'select <field>:0 argmax <field>:1 where <field>:2 greater <value>:2',
#           'select <field>:0 argmin <field>:0 where <field>:1 equal <value>:1',
#           'select <field>:0 argmin <field>:0 where <field>:1 less <value>:1',
#           'select <field>:0 argmin <field>:0 where <field>:1 greater <value>:1',
#           'select <field>:0 argmax <field>:0 where <field>:1 equal <value>:1',
#           'select <field>:0 argmax <field>:0 where <field>:1 less <value>:1',
#           'select <field>:0 argmax <field>:0 where <field>:1 greater <value>:1',
#           'select <field>:0 where <field>:1 equal <value>:1 and <field>:2 equal <value>:2',
#           'select <field>:0 where <field>:1 equal <field>:2 where <field>:0 equal <value>:0',
#           'select <field>:0 where <field>:1 equal <field>:1 where <field>:0 equal <value>:0',
#           'select <field>:1 where <field>:0 equal <field>:0 where <field>:1 equal <value>:1',
#           'sum <field>:0 where <field>:1 equal <value>:1 and where <field>:1 equal <value>:1',
#           'diff <field>:0 where <field>:1 equal <value>:1 and where <field>:1 equal <value>:1',
#           'select <field>:0 argmax <field>:1 where <field>:2 equal <value>:2 and <field>:3 equal <value>:3',
#           'select <field>:0 argmin <field>:1 where <field>:2 equal <value>:2 and <field>:3 equal <value>:3']

# correct = 0
# truth = []
# with open(train_truth) as infile:
#     for line in infile:
#         # wordsList = basic_tokenizer(line)
#         wordsList = line.strip()
#         # for i in range(len(wordsList)):
#         #     wordsList[i] = _DIGIT_RE.sub(b"0", wordsList[i])
#         # truth.append(' '.join(wordsList))
#         truth.append(wordsList.lower())
# index = 0
# with open(train_output) as infile:
#     for line0 in infile:
#         line = line0.strip()
#         if line.lower() == truth[index]:
#             correct += 1
#         else:
#             # compare line with all possible forms, and choose the most similar one
#             dists = np.ones(len(loTemp))
#             for j in range(len(loTemp)):
#                 dists[j] = strSimilarity(line, loTemp[j])
#             newline = loTemp[np.argmax(dists)]
#             if newline == truth[index]:
#                 correct += 1
#             else:
#                 print "wrong examples: %d" %(index + 1)
#                 print truth[index]
#                 print line.lower()
#         index += 1
# print "train accuracy: " + str(correct * 1.0 / len(truth))

# correct = 0
# truth = []
# with open(dev_truth) as infile:
#     for line in infile:
#         # wordsList = basic_tokenizer(line)
#         wordsList = line.strip()
#         # for i in range(len(wordsList)):
#         #     wordsList[i] = _DIGIT_RE.sub(b"0", wordsList[i])
#         # truth.append(' '.join(wordsList))
#         truth.append(wordsList.lower())
# index = 0
# with open(dev_output) as infile:
#     for line0 in infile:
#         line = line0.strip()
#         if line.lower() == truth[index]:
#             correct += 1
#         else:
#             # compare line with all possible forms, and choose the most similar one
#             dists = np.ones(len(loTemp))
#             for j in range(len(loTemp)):
#                 dists[j] = strSimilarity(line, loTemp[j])
#             newline = loTemp[np.argmax(dists)]
#             if newline == truth[index]:
#                 correct += 1
#             else:
#                 print "wrong examples: %d" %(index + 1)
#                 print truth[index]
#                 print line.lower()
#         index += 1
# print "dev accuracy: " + str(correct * 1.0 / len(truth))

correct = 0
truth = []
with open(test_truth) as infile:
    for line in infile:
        # wordsList = basic_tokenizer(line)
        wordsList = line.strip()
        # for i in range(len(wordsList)):
        #     wordsList[i] = _DIGIT_RE.sub(b"0", wordsList[i])
        # truth.append(' '.join(wordsList))
        truth.append(wordsList.lower())
index = 0
with open(test_output) as infile:
    for line0 in infile:
        line = line0.strip()
        if line.lower() == truth[index]:
            correct += 1
        else:
            # compare line with all possible forms, and choose the most similar one
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
    for line in infile:
        # wordsList = basic_tokenizer(line)
        wordsList = line.strip()
        # for i in range(len(wordsList)):
        #     wordsList[i] = _DIGIT_RE.sub(b"0", wordsList[i])
        # truth.append(' '.join(wordsList))
        truth.append(wordsList.lower())
index = 0
with open(geo_output) as infile:
    for line0 in infile:
        line = line0.strip()
        if line.lower() == truth[index]:
            correct += 1
        else:
            # compare line with all possible forms, and choose the most similar one
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
