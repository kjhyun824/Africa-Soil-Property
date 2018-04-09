# Africa Soil Property Prediction 20185160 Songyi Lee (hakbun) (irum)

# 	Test
# input		: PIDN(unique identifier), mid-infra. measurement, Depth, ...
# output	: PIDN(unique identifier), SOC, pH, Ca, P, Sand

import argparse
import tensorflow as tf
import numpy as np
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--index', default = 1, help='index', type=int)
parser.add_argument('--option', default = 'nn', help='options', type=str)
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
	data_input = [[] for k in range(N_train-1)]
	for i, row in enumerate(reader):
		for j in range(1, N_category-5): # Remove ID for Soil
			data_input[i].append(row[j])
		data_output.append(row[N_category-5:])

# Data Normalization : 0.0 ~ 1.0
mininput = []
maxinput = []
minoutput = []
maxoutput = []

for i in range(0,len(data_input[0])-1):
    mininput.append(1000000.0)
    maxinput.append(-100000.0)
    for j in range(0,len(data_input)-1):
        data_input[j][i] = float(data_input[j][i])

for i in range(0,len(data_output[0])-1):
    minoutput.append(1000000.0)
    maxoutput.append(-100000.0)
    for j in range(0,len(data_output)-1):
        data_output[j][i] = float(data_output[j][i])

for i in range(0,len(data_input[0])-1): # column index 
    for j in range(0,len(data_input)-1): # data index
        if mininput[i] > data_input[j][i]:
            mininput[i] = data_input[j][i]
        if maxinput[i] < data_input[j][i]:
            maxinput[i] = data_input[j][i]

for i in range(0,len(data_output[0])-1):
    for j in range(0,len(data_output)-1):
        if minoutput[i] > data_output[j][i]:
            minoutput[i] = data_output[j][i]
        if maxoutput[i] < data_input[j][i]:
            maxoutput[i] = data_output[j][i]

for i in range(0,len(data_input[0])-1): # column index 
    for j in range(0,len(data_input)-1): # data index
        data_input[j][i] = (data_input[j][i] - mininput[i]) / (maxinput[i] - mininput[i])

for i in range(0,len(data_output[0])-1):
    for j in range(0,len(data_output)-1):
        data_output[j][i] = (data_output[j][i] - minoutput[i]) / (maxoutput[i] - minoutput[i])

N_test = 727
test_set = []
with open('/home/students/cs20175052/SJ_Version/Africa_Soil_Property_Prediction/data/sorted_test.csv', 'r') as f:
	reader = csv.reader(f, delimiter=',')
	test_category = next(reader)	
	N_test_category = len(test_category) #3565 
	for i, row in enumerate(reader):
		test_set.append(row[1:])

n_hidden_1 = 256
n_hidden_2 = 256
num_steps = 2000
#num_steps = 2
learning_rate = 0.1
batch_size = 128

def neural_net(x_dict):
    x = x_dict['soil']
    layer1 = tf.layers.dense(x, n_hidden_1)
    layer2 = tf.layers.dense(layer1, n_hidden_2)
    out_layer = tf.layers.dense(layer2,5)

    return out_layer


def model_fn(features, labels, mode):
    #logits = neural_net(features)
    logits = tf.nn.softmax(neural_net(features))
    pred_classes = tf.argmax(logits,axis=1)
    pred_probas = tf.nn.softmax(logits)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

    loss_op = tf.reduce_mean(tf.square(labels - logits))
    #loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))#tf.cast(labels, dtype=tf.float32)))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())

    correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    acc_op = tf.reduce_mean(tf.cast(correct, tf.float32))
    #acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)
    
    estim_specs = tf.estimator.EstimatorSpec(mode=mode, predictions=pred_classes, loss=loss_op, train_op=train_op, )

    return estim_specs

model = tf.estimator.Estimator(model_fn)

data_input = np.asarray(data_input)
data_input = data_input.astype(np.float)
data_output = np.asarray(data_output)
data_output = data_output.astype(np.float)
test_set = np.asarray(test_set)
test_set = test_set.astype(np.float)

input_fn = tf.estimator.inputs.numpy_input_fn(x={'soil': data_input}, y=data_output, batch_size=batch_size, num_epochs=None, shuffle=True)

model.train(input_fn, steps=num_steps)

input_fn = tf.estimator.inputs.numpy_input_fn(x={'soil': test_set}, num_epochs=1, shuffle=False)

output = list(model.predict(input_fn))

print output
print len(output)

'''
for i in range(0,len(output[0])-1):
    for j in range(0,len(output)-1):
        output[j][i] = output[j][i] * (maxoutput[i] - minoutput[i]) + minoutput[i]

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
'''
