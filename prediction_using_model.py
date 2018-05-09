
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import sys
#tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
"""
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")
"""
"""
# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 220, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 100, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 20, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")



FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)
"""

# Load data
print("Loading data...")
x_text, y = data_helpers.load_data_and_labels("h3.neg","h3.neg")
# Build vocabulary
max_document_length = 220
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x_train = np.array(list(vocab_processor.fit_transform(x_text)))
#x_train=x
#y_train=y

def predict_step(x_batch, y_batch):
    """
    predict on a test set
    """
    feed_dict = {
      cnn.input_x: x_batch,
      cnn.input_y: y_batch,
      cnn.dropout_keep_prob: 1.0
    }
    y_pred = sess.run(
        [cnn.predictions], feed_dict)
    return y_pred


with tf.Session() as sess:    
    saver = tf.train.import_meta_graph('/home/saurabh/Downloads/cs298_my_code/cnn-text-classification-tf-master/cs298_code/runs/1524650575/checkpoints/model-400.meta')
    saver.restore(sess,tf.train.latest_checkpoint('/home/saurabh/Downloads/cs298_my_code/cnn-text-classification-tf-master/cs298_code/runs/1524650575/checkpoints'))
    
    #predictions = sess.run(y, feed_dict={x: vocab_processor.fit_transform(x_text)})
    #input_x = tf.Variable(tf.int32, name='input_x')
    graph = tf.get_default_graph()
    input_x = graph.get_tensor_by_name("input_x:0")
    predictions=graph.get_tensor_by_name("output/predictions:0")
    
    #print graph.get_operations()
    feed_dict = {input_x: x_train}
    #classification = sess.run(scores,feed_dict)
    #print classification
    #prediction=tf.argmax(y,1)
    
    #print predictions.eval(feed_dict)
    """
    print result
    print len(result)
    print result[1]
    print result[6]
    """
    print sess.run(predictions,feed_dict)
   
    

