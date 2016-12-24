import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns

def add_layer(input, input_size, output_size, activation_func=None):
    w = tf.Variable(tf.random_normal([input_size, output_size]))
    b = tf.Variable(tf.random_normal([1, output_size]))
    u = tf.matmul(input, w) + b
    return activation_func(u) if activation_func else
def main():
	fopen = open('train.csv', 'r')
	train_df = pd.read_csv(fopen)
	fopen.close()
	fopen = open('test.csv', 'r')
	test_df = pd.read_csv(fopen)
	fopen.close()

	train_label = train_df.label 
	train_df.drop(['label'], inplace=True, axis=1)

	train_arr = train_df.values
	test_arr = test_df.values

	from sklearn.preprocessing import StandardScaler
	from sklearn.cross_validation import train_test_split
	from sklearn.metrics import classification_report
	X_train, X_test, y_train, y_test = train_test_split(train_arr, train_label, test_size=0.05, random_state=42)



	xs = tf.placeholder(tf.float32, [None, 784])
	ys = tf.placeholder(tf.float32, [None, 10])

	hidden_layer = add_layer(xs, 784, 500, activation_func=tf.nn.sigmoid)
	output_layer = add_layer(hidden_layer, 500, 10)
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output_layer, ys))
	opt = tf.train.AdamOptimizer(0.1).minimize(loss)

	with tf.Session() as sess:
	    sess.run(tf.initialize_all_variables())
	    for step in xrange(500):
	        sess.run(opt, feed_dict={ xs : X_train, ys : pd.get_dummies(y_train).values })
	        if step % 10 == 0:
	            print (step, sess.run(loss, feed_dict={ xs : X_train, ys : pd.get_dummies(y_train).values }))
	    predict = sess.run(tf.arg_max(output_layer, 1), feed_dict={ xs : X_test })

	print classification_report(y_test, predict)

if __name__ == '__main__':
	main()