# Transfer-learnable Deep Neural Natural Language Interface for Database


1 Introduction 

Question-Answering (QA) problem has been a long interest to the machine learning community. In this paper, we propose a deep neural parser for QA on table/database queries and search. We transform this problem into building and training a sequence-to-sequence (seq2seq) deep neural network translating natural language queries to a SQL-like logical forms, which could then be used as an intermediate state to query into the table/database. To improve the model, we use Glove pretrained vectors to represent words and tag each word in the queries with possible field name in the corresponding table. We augment a small dataset based on the Wikitable dataset (Pasupat and Liang, 2015) for training and testing, and our model finally achieves a 74.7% training accuracy and 71.0% dev accuracy on this dataset, which gives a new benchmark in research related to Wikitable.

2 Dataset

In this work, we use an open source dataset Wikitable (Pasupat and Liang, 2015) dataset for the training and the evaluation of our model, and study on the synthetic dataset from Neural Enquirer \cite{YinLLK15} to formulate intermediate SQL-like logical forms. 
The synthetic dataset, which uses specific-grammar-based generated questions and logical forms, consists of a training data with 100,000 [Query, Answer, Table] triples, and a test data with 20000 [Query, Answer, Table] triples. This dataset contains four fixed logical forms, and has been tested using our model at the first place.  However, since this dataset is generated based a small vocabulary (~300) and a very limited number of synthetic grammars for each logical form, it is not diverse enough for general Question-Answering purpose, where the complexity of natural language query and continuous evolution of vocabulary is difficult to be included into this few categories.
The Wikitable dataset (Pasupat and Liang, 2015) contains 22,033 examples on 2,108 tables. Since the dataset itself does not contain SQL-like logical forms, we extended the logical-form rules of the synthetic dataset, and composed and annotated nearly 200 queries and their corresponding SQL-like logical forms based on selected Wikitable (Pasupat and Liang, 2015) queries. In our augmented dataset, we generate close to 4000 questions and their corresponding logical forms. The dataset then is randomly shuffled and divided into training data and development data.

3 Using the code

After downloading to local, please make a new directory './train/'. Both the train data and dev data are in the directory './data/'. My tagging function and related rules for the SQL-like logical form I use are in the directory './data_prep/'. In order to visualize the last encoder hidden state, I output to directory './PCA-visual'.

1. Training a 2-layer 1024 hidden size seq2seq-tagger model:

python translate_tag.py

2. Visualize the last encoder hidden state:

cd PCA-visual
python tsne.py

3. Evaluating the train accuracy and dev accuracy is to copy the training data (all with .qu, .ta, .lo files) to './evaluation/' and rename to 'test', then go back to './'

python translate_tag.py --decode
python ./evaluation/verifyLogical.py

Currently the 2-layer 1024-unit is the best tuned model, with 0.747 train acc and 0.710 dev acc on our augmented dataset base on Wikitable.

