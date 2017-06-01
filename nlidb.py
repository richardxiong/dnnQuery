# Tensorflow Model modified by Hongyu Xiong: Deep neural parsing for database query
# specific model: (1) embedding_attention_seq2seq_pretrain
# (2) embedding_attention_seq2seq_pretrain2_tag
# (3) model_with_buckets_tag
# (4) functions are also outputting last encoder hidden state for PCA visual
#
# ==============================================================================

"""Binary for training translation models and decoding from them.
Running this program without --decode will download the corpus into
the directory specified as --data_dir and tokenize it in a very basic way,
and then start training a model saving checkpoints to --train_dir.
Running with --decode starts an interactive loop so you can see how
the current checkpoint translates natual language queries into SQL-like logical forms.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import logging

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import data_utils_tag
import scratch
import seq2seq_model_tag
#import tagger as tgr

#==================================================================================
'''
Things that might need to be changed

conventions:
1) the train files need to be called "train.qu", "train.lo"
2) the dev files need to be called "dev.qu", "dev.lo"
3) the test file needs to be called "test.qu"
4) The test file and the test table needs to be put in "test_dir"
5) It will be best the make all the _dir's equal, that means all the files are in the same folder
6) If enable_table_test is True, makes sure the test table named "test.json"
end of it
'''
#==================================================================================

tf.app.flags.DEFINE_float("learning_rate", 0.05 * 0.01, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.96,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 128,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 1024, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("from_vocab_size", 1000, "English vocabulary size.")
tf.app.flags.DEFINE_integer("to_vocab_size", 100, "French vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "./data", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "./train_3_1024", "Training directory.")
tf.app.flags.DEFINE_string("from_train_data", None, "Training data.")
tf.app.flags.DEFINE_string("to_train_data", None, "Training data.")
tf.app.flags.DEFINE_string("from_dev_data", None, "Training data.")
tf.app.flags.DEFINE_string("to_dev_data", None, "Training data.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 100,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_boolean("use_fp16", False,
                            "Train using fp16 instead of fp32.")

# added by Kaifeng, can be changed
tf.app.flags.DEFINE_string("test_dir", "./evaluation", "Test directory")
#tf.app.flags.DEFINE_integer('max_num_steps', 10000, 'the maximum number of steps.')
# no need to change this line if not using real table to test
tf.app.flags.DEFINE_boolean("enable_table_test", False, "Whether use a true table to test")
# directory of GloVe. Here by default we use GloVe 6B 100
#tf.app.flags.DEFINE_string('GloVe_dir', "glove.6B", "GloVe directory. ")
'''
'''

FLAGS = tf.app.flags.FLAGS

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
### _buckets = [(11, 8), (15, 12), (20, 16), (24, 21)]
# 1. no field / tagging model
#_buckets = [(10, 8), (15, 12), (19, 16), (23, 21)] #less pad then previous

_buckets = [(11, 8), (15, 12), (19, 14)]  # new logical forms

def read_data(source_path, target_path, tag_path, max_size=None):
  """Read data from source and target files and put into buckets.
  Args:
    source_path: path to the files with token-ids for the source language.
    target_path: path to the file with token-ids for the target language;
      it must be aligned with the source file: n-th line contains the desired
      output for n-th line from the source_path.
    max_size: maximum number of lines to read, all other will be ignored;
      if 0 or None, data files will be read completely (no limit).
  Returns:
    data_set: a list of length len(_buckets); data_set[n] contains a list of
      (source, target) pairs read from the provided data files that fit
      into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
      len(target) < _buckets[n][1]; source and target are lists of token-ids.
  """
  data_set = [[] for _ in _buckets]
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    with tf.gfile.GFile(target_path, mode="r") as target_file:
      with tf.gfile.GFile(tag_path, mode="r") as tag_file:
        source, target, tag = source_file.readline(), target_file.readline(), tag_file.readline()
        counter = 0
        while source and target and tag and (not max_size or counter < max_size):
          counter += 1
          if counter % 200 == 0:
            print("  reading data line %d" % counter)
            sys.stdout.flush()
          source_ids = [int(x) for x in source.split()]
          target_ids = [int(x) for x in target.split()]
          tag_ids = [int(x) for x in tag.split()]
          target_ids.append(data_utils_tag.EOS_ID)
          for bucket_id, (source_size, target_size) in enumerate(_buckets):
            if len(source_ids) < source_size and len(target_ids) < target_size:
              data_set[bucket_id].append([source_ids, tag_ids, target_ids])
              break
          source, target, tag = source_file.readline(), target_file.readline(), tag_file.readline()
  return data_set


def create_model(session, forward_only):
  """Create translation model and initialize or load parameters in session."""
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  model = seq2seq_model_tag.Seq2SeqModel(
      FLAGS.from_vocab_size,
      FLAGS.to_vocab_size,
      _buckets,
      FLAGS.size,
      FLAGS.num_layers,
      FLAGS.max_gradient_norm,
      FLAGS.batch_size,
      FLAGS.learning_rate,
      FLAGS.learning_rate_decay_factor,
      forward_only=forward_only,
      dtype=dtype)
  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
  if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.global_variables_initializer())
  return model


def train():
  """Train a en->fr translation model using WMT data."""
  from_train = None
  to_train = None
  tag_train = None
  from_dev = None
  to_dev = None
  tag_dev = None
  if FLAGS.from_train_data and FLAGS.to_train_data:
    from_train_data = FLAGS.from_train_data
    to_train_data = FLAGS.to_train_data
    tag_train_data = FLAGS.tag_train_data
    from_dev_data = from_train_data
    to_dev_data = to_train_data
    tag_dev_data = tag_train_data
    if FLAGS.from_dev_data and FLAGS.to_dev_data:
      from_dev_data = FLAGS.from_dev_data
      to_dev_data = FLAGS.to_dev_data
      tag_dev_data = FLAGS.tag_dev_data
    from_train, to_train, tag_train, from_dev, to_dev, tag_dev, _, _ = data_utils_tag.prepare_data(
        FLAGS.data_dir,
        from_train_data,
        to_train_data,
        tag_train_data,
        from_dev_data,
        to_dev_data,
        tag_dev_data,
        FLAGS.from_vocab_size,
        FLAGS.to_vocab_size)
  else:
      # Prepare WMT data.
      print("Preparing WMT data in %s" % FLAGS.data_dir)
      from_train, to_train, tag_train, from_dev, to_dev, tag_dev, _, _ = data_utils_tag.prepare_wmt_data(
          FLAGS.data_dir, FLAGS.from_vocab_size, FLAGS.to_vocab_size)

  with tf.Session() as sess:
    # Create model.
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
    model = create_model(sess, False)

    # Read data into buckets and compute their sizes.
    print ("Reading development and training data (limit: %d)."
           % FLAGS.max_train_data_size)
    dev_set = read_data(from_dev, to_dev, tag_dev)
    train_set = read_data(from_train, to_train, tag_train, FLAGS.max_train_data_size)
    train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
    train_total_size = float(sum(train_bucket_sizes))

    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in xrange(len(train_bucket_sizes))]

    # This is the training loop.
    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []
    while True:
      # Choose a bucket according to data distribution. We pick a random number
      # in [0, 1] and use the corresponding interval in train_buckets_scale.
      random_number_01 = np.random.random_sample()
      bucket_id = min([i for i in xrange(len(train_buckets_scale))
                       if train_buckets_scale[i] > random_number_01])

      # Get a batch and make a step.
      start_time = time.time()
      encoder_inputs, tag_inputs, decoder_inputs, target_weights = model.get_batch(
          train_set, bucket_id)
      _, step_loss, _, _ = model.step(sess, encoder_inputs, tag_inputs, decoder_inputs, ###, _
                                   target_weights, bucket_id, False)
      step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
      loss += step_loss / FLAGS.steps_per_checkpoint
      current_step += 1

      # Once in a while, we save checkpoint, print statistics, and run evals.
      if current_step % FLAGS.steps_per_checkpoint == 0:
        # Print statistics for the previous epoch.
        perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
        print ("global step %d learning rate %.4f step-time %.2f perplexity "
               "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                         step_time, perplexity))
        # Decrease learning rate if no improvement was seen over last 3 times.
        if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
          sess.run(model.learning_rate_decay_op)
        previous_losses.append(loss)
        # Save checkpoint and zero timer and loss.
        checkpoint_path = os.path.join(FLAGS.train_dir, "translate.ckpt")
        if not os.path.isabs(checkpoint_path):
          checkpoint_path = os.path.abspath(os.path.join(os.getcwd(), checkpoint_path))
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)
        step_time, loss = 0.0, 0.0
        # Run evals on development set and print their perplexity.
        ### Open a txt file recording the last hidden state
        # filename = "last_hidden_state-"+str(model.global_step.eval())+".txt"
        # lasthidden_path = os.path.join("./PCA-visual/", filename)
        # f_lh = open(lasthidden_path, 'w')
        for bucket_id in xrange(len(_buckets)):
          if len(dev_set[bucket_id]) == 0:
            print("  eval: empty bucket %d" % (bucket_id))
            continue
          encoder_inputs, tag_inputs, decoder_inputs, target_weights = model.get_batch(
              dev_set, bucket_id)
          # _, eval_loss, _, eval_lasthidden = model.step(sess, encoder_inputs, tag_inputs, decoder_inputs, ###
          #                              target_weights, bucket_id, True)
          _, eval_loss, _, _ = model.step(sess, encoder_inputs, tag_inputs, decoder_inputs, ###
                                       target_weights, bucket_id, True)
          eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300 else float(
              "inf")
          print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
          ### Print hidden state to file
          #print("   bucket %d has %d samples" % (bucket_id, len(eval_lasthidden)))
        #   for j in range(FLAGS.batch_size):
        #     for i in range(FLAGS.num_layers): # all the layer of encoders are outputed
        #       words = [str(num) for num in eval_lasthidden[i][j]]
        #       f_lh.write(','.join(words))
        #     f_lh.write('\n')
        # f_lh.close()
        sys.stdout.flush()


# modified by Kaifeng
import json
from logicalParser import logicalParser
def decode():
  with tf.Session() as sess:
    # Create model and load parameters.
    model = create_model(sess, True)
    model.batch_size = 1  # We decode one sentence at a time.

    # Load vocabularies.
    en_vocab_path = os.path.join(FLAGS.data_dir,
                                 "vocab%d.from" % FLAGS.from_vocab_size)
    fr_vocab_path = os.path.join(FLAGS.data_dir,
                                 "vocab%d.to" % FLAGS.to_vocab_size)
    en_vocab, _ = data_utils_tag.initialize_vocabulary(en_vocab_path)
    fr_vocab, rev_fr_vocab = data_utils_tag.initialize_vocabulary(fr_vocab_path)

    # Decode from standard input.
    # changed by Kaifeng, for test
    offset = 0; # the test data is the last 20000 items in the table
    testTableFile = FLAGS.test_dir +'/test.json'
    if FLAGS.enable_table_test:
        print('loading database table')
        with open(testTableFile) as testTables:
            tables = json.load(testTables)
        answerOutput = open(FLAGS.test_dir + '/answer.out', 'w')

    trainQuestionFile = FLAGS.data_dir + '/rand_train.qu'
    trainTagFile = FLAGS.data_dir + '/rand_train.ta'   # For tagging model, Hongyu
    devQuestionFile = FLAGS.data_dir + '/rand_dev.qu'
    devTagFile = FLAGS.data_dir + '/rand_dev.ta'   # For tagging model, Hongyu
    testQuestionFile = FLAGS.data_dir + '/rand_test.qu'
    testTagFile = FLAGS.data_dir + '/rand_test.ta'   # For tagging model, Hongyu

    #0530 newly added
    geoQuestionFile = FLAGS.data_dir + '/GeoQuery/geo880.qu'
    geoTagFile = FLAGS.data_dir + '/GeoQuery/geo880.ta'   # For tagging model, Hongyu
    logicalTemp_geo = open(FLAGS.test_dir + '/logicalTemp_geo.out', 'w')
    
    logicalTemp_train = open(FLAGS.test_dir + '/logicalTemp_train.out', 'w')
    logicalTemp_dev = open(FLAGS.test_dir + '/logicalTemp_dev.out', 'w')
    logicalTemp_test = open(FLAGS.test_dir + '/logicalTemp_test.out', 'w')

    ### evaluating tagging model, Hongyu
    
    print('======= start testing =======')
    # print('=== train dataset ===')
    # with open(trainQuestionFile,'r') as trainQuestions:
    #   with open(trainTagFile, 'r') as trainTags: 
    #     q_index = 0
    #     sentence, tag_sen = trainQuestions.readline(), trainTags.readline()
    #     while sentence and tag_sen:
    #         if q_index % 200 == 0:
    #           print("  reading data line %d" % q_index)
    #           sys.stdout.flush()
    #         qid = 'qID_' + str(q_index)
    #         print('training question: ', qid)
    #         # Get token-ids for the input sentence.
    #         token_ids = data_utils_tag.sentence_to_token_ids(tf.compat.as_bytes(sentence), en_vocab)
    #         tag_ids = data_utils_tag.sentence_to_token_ids(tf.compat.as_bytes(tag_sen), fr_vocab)
    #         # Which bucket does it belong to?
    #         bucket_id = len(_buckets) - 1
    #         for i, bucket in enumerate(_buckets):
    #           if bucket[0] >= len(token_ids):
    #             bucket_id = i
    #             break
    #         else:
    #           logging.warning("Sentence truncated: %s", sentence)

    #         # Get a 1-element batch to feed the sentence to the model.
    #         encoder_inputs, tag_inputs, decoder_inputs, target_weights = model.get_batch(
    #             {bucket_id: [(token_ids, tag_ids, [])]}, bucket_id)
    #         # Get output logits for the sentence.
    #         _, _, output_logits, _ = model.step(sess, encoder_inputs, tag_inputs, decoder_inputs,
    #                                          target_weights, bucket_id, True)
    #         # This is a greedy decoder - outputs are just argmaxes of output_logits.
    #         outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
    #         # If there is an EOS symbol in outputs, cut them at that point.
    #         if data_utils_tag.EOS_ID in outputs:
    #           outputs = outputs[:outputs.index(data_utils_tag.EOS_ID)]
    #         # Print out French sentence corresponding to outputs.
    #         resultLogical = " ".join([tf.compat.as_str(rev_fr_vocab[output]) for output in outputs])
    #         if FLAGS.enable_table_test:
    #             resultAnswer = logicalParser(tables[qid], resultLogical)
    #             answerOutput.write(str(resultAnswer) + '\n')

    #         logicalTemp_train.write(str(resultLogical) + '\n')
    #         q_index += 1
    #         sentence, tag_sen = trainQuestions.readline(), trainTags.readline()
    
    # print('=== dev dataset ===')
    # with open(devQuestionFile,'r') as devQuestions:
    #   with open(devTagFile, 'r') as devTags: 
    #     q_index = 0
    #     sentence, tag_sen = devQuestions.readline(), devTags.readline()
    #     while sentence and tag_sen:
    #         if q_index % 200 == 0:
    #           print("  reading data line %d" % q_index)
    #           sys.stdout.flush()
    #         qid = 'qID_' + str(q_index)
    #         print('deving question: ', qid)
    #         # Get token-ids for the input sentence.
    #         token_ids = data_utils_tag.sentence_to_token_ids(tf.compat.as_bytes(sentence), en_vocab)
    #         tag_ids = data_utils_tag.sentence_to_token_ids(tf.compat.as_bytes(tag_sen), fr_vocab)
    #         # Which bucket does it belong to?
    #         bucket_id = len(_buckets) - 1
    #         for i, bucket in enumerate(_buckets):
    #           if bucket[0] >= len(token_ids):
    #             bucket_id = i
    #             break
    #         else:
    #           logging.warning("Sentence truncated: %s", sentence)

    #         # Get a 1-element batch to feed the sentence to the model.
    #         encoder_inputs, tag_inputs, decoder_inputs, target_weights = model.get_batch(
    #             {bucket_id: [(token_ids, tag_ids, [])]}, bucket_id)
    #         # Get output logits for the sentence.
    #         _, _, output_logits, _ = model.step(sess, encoder_inputs, tag_inputs, decoder_inputs,
    #                                          target_weights, bucket_id, True)
    #         # This is a greedy decoder - outputs are just argmaxes of output_logits.
    #         outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
    #         # If there is an EOS symbol in outputs, cut them at that point.
    #         if data_utils_tag.EOS_ID in outputs:
    #           outputs = outputs[:outputs.index(data_utils_tag.EOS_ID)]
    #         # Print out French sentence corresponding to outputs.
    #         resultLogical = " ".join([tf.compat.as_str(rev_fr_vocab[output]) for output in outputs])
    #         if FLAGS.enable_table_test:
    #             resultAnswer = logicalParser(tables[qid], resultLogical)
    #             answerOutput.write(str(resultAnswer) + '\n')

    #         logicalTemp_dev.write(str(resultLogical) + '\n')
    #         q_index += 1
    #         sentence, tag_sen = devQuestions.readline(), devTags.readline()
    
    print('=== test dataset ===')        
    with open(testQuestionFile,'r') as testQuestions:
      with open(testTagFile, 'r') as testTags: 
        q_index = 0
        sentence, tag_sen = testQuestions.readline(), testTags.readline()
        while sentence and tag_sen:
            if q_index % 200 == 0:
              print("  reading data line %d" % q_index)
              sys.stdout.flush()
            qid = 'qID_' + str(q_index)
            print('testing question: ', qid)
            # Get token-ids for the input sentence.
            token_ids = data_utils_tag.sentence_to_token_ids(tf.compat.as_bytes(sentence), en_vocab)
            tag_ids = data_utils_tag.sentence_to_token_ids(tf.compat.as_bytes(tag_sen), fr_vocab)
            # Which bucket does it belong to?
            bucket_id = len(_buckets) - 1
            for i, bucket in enumerate(_buckets):
              if bucket[0] >= len(token_ids):
                bucket_id = i
                break
            else:
              logging.warning("Sentence truncated: %s", sentence)

            # Get a 1-element batch to feed the sentence to the model.
            encoder_inputs, tag_inputs, decoder_inputs, target_weights = model.get_batch(
                {bucket_id: [(token_ids, tag_ids, [])]}, bucket_id)
            
            # Get output logits for the sentence and CONFUSION matrix. # 0531 newly added
            filename = "confusion_matrix.txt"
            confusion_path = os.path.join("./PCA-visual/", filename)
            f_con = open(confusion_path, 'a+')
            _, _, output_logits, confusion_matrix = model.step(sess, encoder_inputs, tag_inputs, decoder_inputs,
                                             target_weights, bucket_id, True)
            f_con.write('*** example: '+str(q_index)+' ***\n')
            for i in range(confusion_matrix.shape[1]):
              words = [str(x) for x in confusion_matrix[0][i]]
              f_con.write(','.join(words) + '\n')
            f_con.close()
        
            # This is a greedy decoder - outputs are just argmaxes of output_logits.
            outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
            # If there is an EOS symbol in outputs, cut them at that point.
            if data_utils_tag.EOS_ID in outputs:
              outputs = outputs[:outputs.index(data_utils_tag.EOS_ID)]
            # Print out French sentence corresponding to outputs.
            resultLogical = " ".join([tf.compat.as_str(rev_fr_vocab[output]) for output in outputs])
            if FLAGS.enable_table_test:
                resultAnswer = logicalParser(tables[qid], resultLogical)
                answerOutput.write(str(resultAnswer) + '\n')

            logicalTemp_test.write(str(resultLogical) + '\n')
            q_index += 1
            sentence, tag_sen = testQuestions.readline(), testTags.readline()

    # print('=== geo dataset ===')
    # with open(geoQuestionFile,'r') as geoQuestions:
    #   with open(geoTagFile, 'r') as geoTags: 
    #     q_index = 0
    #     sentence, tag_sen = geoQuestions.readline(), geoTags.readline()
    #     while sentence and tag_sen:
    #         if q_index % 200 == 0:
    #           print("  reading data line %d" % q_index)
    #           sys.stdout.flush()
    #         qid = 'qID_' + str(q_index)
    #         print('geoing question: ', qid)
    #         # Get token-ids for the input sentence.
    #         token_ids = data_utils_tag.sentence_to_token_ids(tf.compat.as_bytes(sentence), en_vocab)
    #         tag_ids = data_utils_tag.sentence_to_token_ids(tf.compat.as_bytes(tag_sen), fr_vocab)
    #         # Which bucket does it belong to?
    #         bucket_id = len(_buckets) - 1
    #         for i, bucket in enumerate(_buckets):
    #           if bucket[0] >= len(token_ids):
    #             bucket_id = i
    #             break
    #         else:
    #           logging.warning("Sentence truncated: %s", sentence)

    #         # Get a 1-element batch to feed the sentence to the model.
    #         encoder_inputs, tag_inputs, decoder_inputs, target_weights = model.get_batch(
    #             {bucket_id: [(token_ids, tag_ids, [])]}, bucket_id)
    #         # Get output logits for the sentence.
    #         _, _, output_logits, _ = model.step(sess, encoder_inputs, tag_inputs, decoder_inputs,
    #                                          target_weights, bucket_id, True)
    #         # This is a greedy decoder - outputs are just argmaxes of output_logits.
    #         outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
    #         # If there is an EOS symbol in outputs, cut them at that point.
    #         if data_utils_tag.EOS_ID in outputs:
    #           outputs = outputs[:outputs.index(data_utils_tag.EOS_ID)]
    #         # Print out French sentence corresponding to outputs.
    #         resultLogical = " ".join([tf.compat.as_str(rev_fr_vocab[output]) for output in outputs])
    #         if FLAGS.enable_table_test:
    #             resultAnswer = logicalParser(tables[qid], resultLogical)
    #             answerOutput.write(str(resultAnswer) + '\n')

    #         logicalTemp_geo.write(str(resultLogical) + '\n')
    #         q_index += 1
    #         sentence, tag_sen = geoQuestions.readline(), geoTags.readline()
    # logicalTemp_geo.close()
    # logicalTemp_train.close()
    # logicalTemp_dev.close()
    logicalTemp_test.close()
    if FLAGS.enable_table_test:
        answerOutput.close()


def self_test():
  """Test the translation model."""
  with tf.Session() as sess:
    print("Self-test for neural translation model.")
    # Create model with vocabularies of 10, 2 small buckets, 2 layers of 32.
    model = seq2seq_model.Seq2SeqModel(10, 10, [(3, 3), (6, 6)], 32, 2,
                                       5.0, 32, 0.3, 0.99, num_samples=8)
    sess.run(tf.global_variables_initializer())

    # Fake data set for both the (3, 3) and (6, 6) bucket.
    data_set = ([([1, 1], [2, 2]), ([3, 3], [4]), ([5], [6])],
                [([1, 1, 1, 1, 1], [2, 2, 2, 2, 2]), ([3, 3, 3], [5, 6])])
    for _ in xrange(5):  # Train the fake model for 5 steps.
      bucket_id = random.choice([0, 1])
      encoder_inputs, decoder_inputs, target_weights, _ = model.get_batch(
          data_set, bucket_id)
      model.step(sess, encoder_inputs, decoder_inputs, target_weights,
                 bucket_id, False)


def main(_):
  if FLAGS.self_test:
    self_test()
  elif FLAGS.decode:
    decode()
  else:
    train()

if __name__ == "__main__":
  tf.app.run()