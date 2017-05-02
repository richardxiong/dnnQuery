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
test_truth_path = "../data"
test_output_path = "../evaluation"


# test_truth = test_truth_path + "/rand_test.lox"
# test_output = test_output_path + "/logicalForm_test.out"

test_truth = test_truth_path + "/rand_train.lo"
test_output = test_output_path + "/forms_train.lo"

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

print "total accuracy: " + str(correct * 1.0 / len(truth))