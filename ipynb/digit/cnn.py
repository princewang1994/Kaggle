import tensorflow as tf
import numpy as np
import pandas as pd


fopen = open('../dataset/digit/train.csv', 'r')
train_df = pd.read_csv(fopen)
fopen.close()
fopen = open('../dataset/digit/test.csv', 'r')
test_df = pd.read_csv(fopen)
fopen.close() 

train_label = train_df.label.values
train_df.drop(['label'], inplace=True, axis=1)

train_arr = train_df.values.astype(np.float32)
train_arr /= 255

test_arr = test_df.values.astype(np.float32) 
test_arr /= 255.0


from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_arr, train_label, test_size=0.1, random_state=42)
n_sample = X_train.shape[0]
y_train = label_binarize(y_train, range(10))
y_test = label_binarize(y_test, range(10)) 


def add_fc_layer(input, input_size, output_size, activation=None):
    input = tf.reshape(input, [-1, input_size]) 
    w = tf.Variable(tf.truncated_normal([input_size, output_size], stddev=0.1, dtype=tf.float32))
    b = tf.Variable(tf.random_normal([1, output_size], dtype=tf.float32))
    out = tf.matmul(input, w) + b
    return activation(out) if activation else out 


def add_conv_layer(x, filter_size, activation=None):
    w = tf.Variable(tf.truncated_normal(filter_size, stddev=0.1), dtype=tf.float32)
    conv = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='VALID')
    b = tf.Variable(tf.random_normal([1, filter_size[-1]], dtype=tf.float32))
    out = conv + b
    return activation(out) if activation else out

def add_pool_layer(x, activation=None):
    pool = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    return activation(pool) if activation else pool 

def acc(pred, target):
    return (np.argmax(pred, axis=1) == np.argmax(target, axis=1)).sum() / float(target.shape[0]) 

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
reshape = tf.reshape(x, [-1, 28, 28, 1])
keep_prob = tf.placeholder(tf.float32) 

l1 = add_conv_layer(reshape, [5, 5, 1, 40], tf.nn.relu)
l2 = add_pool_layer(l1)

l3 = add_conv_layer(l2, [5, 5, 40, 80], tf.nn.relu)
l4 = add_pool_layer(l3)

l5 = add_fc_layer(l4, 4*4*80, 1024, tf.nn.relu)
drop = tf.nn.dropout(l5, keep_prob=keep_prob)
l6 = add_fc_layer(drop, 1024, 10)

soft_max = tf.nn.softmax(l6) 
loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(l6, y)) 
opt = tf.train.AdamOptimizer(1e-3).minimize(loss) 

ss = tf.InteractiveSession()
ss.run(tf.initialize_all_variables())
for step in xrange(2000):
    start = step * 200 % X_train.shape[0]
    end = (step * 200 + 200) % X_train.shape[0]
    xs, ys = X_train[start:end], y_train[start:end]
    ss.run(opt, feed_dict={x : xs, y : ys, keep_prob : 0.4})
    if step % 50 == 0:
        train_result = ss.run(soft_max, feed_dict={x : xs, keep_prob : 1})
        test_result = ss.run(soft_max, feed_dict={x : X_test, keep_prob : 1})
        ls = ss.run(loss, feed_dict={x : xs, y : ys, keep_prob : 1})
        print "train_acc={0}, loss={1}, valid_acc={2}".format(acc(train_result, ys), ls, acc(test_result, y_test))
     

def output_csv(predict, file_name):
    import csv
    output = zip(range(1, test_arr.shape[0] + 1), predict)
    output_file = open(file_name, 'w')
    writer = csv.writer(output_file)
    writer.writerow(['ImageId', 'Label'])
    writer.writerows(output)
    output_file.close()

print 'start prediction'
predict = np.array([])
for i in xrange(test_arr.shape[0] / 1000):
    res = ss.run(soft_max, feed_dict={x : test_arr[i * 1000: i * 1000 + 1000], keep_prob : 1})
    predict = np.append(predict, np.argmax(res, axis=1))
predict = predict.astype(np.int32)
print 'prediction completed'

print 'outputing to digit-cnn.csv'
output_csv(predict, 'digit-cnn.csv')
print 'success!'
