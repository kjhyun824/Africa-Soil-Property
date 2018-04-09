# Africa Soil Property Prediction 20185160 Songyi Lee (hakbun) (irum)

# 	Test
# input		: PIDN(unique identifier), mid-infra. measurement, Depth, ...
# output	: PIDN(unique identifier), SOC, pH, Ca, P, Sand

import argparse
import tensorflow as tf
import numpy as np
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--index', default = 30000, help='index', type=int)
parser.add_argument('--option', default = 'softmax', help='options', type=str)
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


# Training
X = tf.placeholder(tf.float32, shape=[None, N_category-6])
Y = tf.placeholder(tf.float32, shape=[None, 5])

w = tf.Variable(tf.zeros([N_category-6,5]), name='weight')
b = tf.Variable(tf.zeros([5]), name = 'bias')

hypothesis = tf.sigmoid(tf.matmul(X,w) + b)

cost = tf.reduce_mean(tf.square(Y - hypothesis))

learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
#optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(cost)
#optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

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

init = tf.global_variables_initializer()

correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
	sess.run(init)
        step = 0
        while True:
            _ , loss, acc = sess.run([optimizer, cost, accuracy], feed_dict={X: data_input, Y: data_output})
            step += 1
	    if step % 10 == 0:
                print '%d,%f,%f' % (step, loss, acc)
            if step == 3000:
                break
        output = sess.run([hypothesis], feed_dict={X: test_set}) 

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
