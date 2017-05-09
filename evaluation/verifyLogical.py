'''
A simple script for logical form verification
'''
import re
#from data_utils import basic_tokenizer
# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"([,!?\"')(])")   # get rid of '.':;
_DIGIT_RE = re.compile(br"\d")


'''
input files
'''
truth_path = "../data"
output_path = "../evaluation"


train_truth = truth_path + "/rand_train.lox"
train_output = output_path + "/logicalTemp_train.out"
dev_truth = truth_path + "/rand_dev.lox"
dev_output = output_path + "/logicalTemp_dev.out"
test_truth = truth_path + "/rand_test.lox"
test_output = output_path + "/logicalTemp_test.out"

# train_truth = truth_path + "/rand_train.lo"
# train_output = output_path + "/forms_train.lo"
# dev_truth = truth_path + "/rand_dev.lo"
# dev_output = output_path + "/forms_dev.lo"
# test_truth = truth_path + "/rand_test.lo"
# test_output = output_path + "/forms_test.lo"

correct = 0
truth = []
with open(train_truth) as infile:
    for line in infile:
        # wordsList = basic_tokenizer(line)
        wordsList = line.strip()
        # for i in range(len(wordsList)):
        #     wordsList[i] = _DIGIT_RE.sub(b"0", wordsList[i])
        # truth.append(' '.join(wordsList))
        truth.append(wordsList.lower())
index = 0
with open(train_output) as infile:
    for line0 in infile:
        line = line0.strip()
        if line.lower() == truth[index]:
            correct += 1
        else:
            print "wrong examples: %d" %(index + 1)
            print line0
        index += 1
print "train accuracy: " + str(correct * 1.0 / len(truth))

correct = 0
truth = []
with open(dev_truth) as infile:
    for line in infile:
        # wordsList = basic_tokenizer(line)
        wordsList = line.strip()
        # for i in range(len(wordsList)):
        #     wordsList[i] = _DIGIT_RE.sub(b"0", wordsList[i])
        # truth.append(' '.join(wordsList))
        truth.append(wordsList.lower())
index = 0
with open(dev_output) as infile:
    for line0 in infile:
        line = line0.strip()
        if line.lower() == truth[index]:
            correct += 1
        else:
            print "wrong examples: %d" %(index + 1)
            print line0
        index += 1
print "dev accuracy: " + str(correct * 1.0 / len(truth))

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
            print "wrong examples: %d" %(index + 1)
            print line0
        index += 1
print "test accuracy: " + str(correct * 1.0 / len(truth))