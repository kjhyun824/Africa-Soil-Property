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
parser.add_argument('--option', default='multlayer', help='options', type=str)
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

weights = {
        'h1' : tf.Variable(tf.random_normal([N_category-6, 256])),
        'h2' : tf.Variable(tf.random_normal([256, 128])),
        'h3' : tf.Variable(tf.random_normal([128, 64])),
        'h4' : tf.Variable(tf.random_normal([64, 32])),
        'out' : tf.Variable(tf.random_normal([32, 5]))
}

biases = {
        'b1' : tf.Variable(tf.random_normal([256])),
        'b2' : tf.Variable(tf.random_normal([128])),
        'b3' : tf.Variable(tf.random_normal([64])),
        'b4' : tf.Variable(tf.random_normal([32])),
        'out' : tf.Variable(tf.random_normal([5]))
}

def multilayer_perceptron(x):
    layer1 = tf.nn.softmax(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
    layer2 = tf.nn.softmax(tf.add(tf.matmul(layer1, weights['h2']), biases['b2']))
    layer3 = tf.nn.softmax(tf.add(tf.matmul(layer2, weights['h3']), biases['b3']))
    layer4 = tf.nn.softmax(tf.add(tf.matmul(layer3, weights['h4']), biases['b4']))
    outlayer = tf.nn.softmax(tf.add(tf.matmul(layer4, weights['out']), biases['out']))

    return outlayer

logits = multilayer_perceptron(X)

#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
cost = tf.reduce_mean(tf.sqrt(tf.reduce_mean((Y - logits) * (Y - logits))))

learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

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

print("start")
init = tf.global_variables_initializer()

threshold = 0.00001
lossBefore = 100

correct = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.Session() as sess:
	sess.run(init)
        step = 0
        while True:
                _ , loss, acc = sess.run([optimizer, cost, accuracy], feed_dict={X: data_input, Y: data_output})
                step += 1
		if step % 100 == 0:
                    print(step, "Cost : ", loss, "Accuracy : ", acc)
                    lossDiff = lossBefore - loss
                    if lossDiff > 0 and lossDiff < threshold:
                        break
                    lossBefore = loss
                if step == 3000:
                    break

        output = sess.run([pred], feed_dict={X: test_set}) 

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
