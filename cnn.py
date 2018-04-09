# Africa Soil Property Prediction 20185160 Songyi Lee (hakbun) (irum)

# 	Test
# input		: PIDN(unique identifier), mid-infra. measurement, Depth, ...
# output	: PIDN(unique identifier), SOC, pH, Ca, P, Sand

import argparse
import tensorflow as tf
import numpy as np
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--index', default=100, help='index', type=int)
parser.add_argument('--option', default='cnn', help='options', type=str)
args = parser.parse_args()

# Data Processing
N_train = 1158
category_info = []

data_input = []
data_output = []
with open('/home/students/cs20175052/SJ_Version/Africa_Soil_Property_Prediction/data/training.csv', 'r') as f:
	reader = csv.reader(f, delimiter=',')
	category_info = next(reader)	
	N_category = len(category_info) # 3600
        #print(N_category)
	data_input = [[] for k in range(N_train-1)]
	for i, row in enumerate(reader):
		for j in range(1, N_category-5): # Remove ID for Soil
			data_input[i].append(row[j])
		data_output.append(row[N_category-5:])

# Training
X = tf.placeholder(tf.float32, shape=[None, N_category-6])
Y = tf.placeholder(tf.float32, shape=[None, 5])

learning_rate = 0.001
num_steps = 2000
batch_size = 128
dropout = 0.25
num_input = 3594
num_classes = 5

def conv_net(x_dict, n_classes, dropout, reuse, is_training):
    with tf.variable_scope('ConvNet', reuse=reuse):
        x = x_dict['soil']
        x = tf.reshape(x, shape=[-1, 6, 599, 1])
        conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

        conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

        fc1 = tf.contrib.layers.flatten(conv2)
        fc1 = tf.layers.dense(fc1, 1024)
        fc1 = tf.layers.dropout(fc1,rate=dropout,training=is_training)

        out = tf.layers.dense(fc1, n_classes)

    return out

def model_fn(features, labels, mode):
    logits_train = conv_net(features, num_classes, dropout, reuse=False, is_training=True)
    logits_test = conv_net(features, num_classes, dropout, reuse=True, is_training=False)

    pred_classes = tf.argmax(logits_test, axis = 1)
    pred_probas = tf.nn.softmax(logits_text)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_train, labels=tf.cast(labels, dtype=tf.float32)))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op, global_steps=tf.train.get_global_step())

    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

    estim_specs = tf.estimator.EstimatorSpec(mode=mode, predictions=pred_classes, loss=loss_op, train_op=train_op, eval_metric_ops={'accuracy': acc_op})

    return estim_specs

model = tf.estimator.Estimator(model_fn)
input_fn = tf.estimator.inputs.numpy_input_fn(x={'soil':np.asarray(data_input)}, y=np.asarray(data_output), batch_size=batch_size, num_epochs=None, shuffle=True)
model.train(input_fn, steps=num_steps)

# Test data set
N_test = 727
test_set = []
with open('/home/students/cs20175052/SJ_Version/Africa_Soil_Property_Prediction/data/sorted_test.csv', 'r') as f:
	reader = csv.reader(f, delimiter=',')
	test_category = next(reader)	
	N_test_category = len(test_category) #3565 
        #print(N_test_category)
	for i, row in enumerate(reader):
		test_set.append(row[1:])

input_fn = tf.estimator.inputs.numpy_input_fn(x={'soil':np.asarray(test_set)}, y=None, batch_size=batch_size, shuffle=False)

e = model.evaluate(input_fn)

print("Testing Accuracy:", e['accuracy'])

output=e['predictions']

outputFileName = '/home/students/cs20175052/SJ_Version/Africa_Soil_Property_Prediction/result/output_' + args.option + '_' + str(args.index) + ".csv"
outputFile = open(outputFileName,'w')
for line in output[0]:
    text=""
    for i in range(0,4):
        text = text+str(line[i])+','
    text += str(line[4])
    text += '\n'
    outputFile.write(text)

outputFile.close()
