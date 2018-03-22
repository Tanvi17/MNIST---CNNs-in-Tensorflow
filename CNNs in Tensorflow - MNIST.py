'''

Model architecture: mix of cnn, max pooling and fully-connected layers

Dataset: MNIST
'''

import tensorflow as tf

## Dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(".", one_hot=True, reshape=False)

#parameters
learning_rate = 0.001
epochs = 14
batch_size = 128
n_classes = 10
dropout = 0.75
test_valid_size = 256

## Weights and biases
weights = {
	'wc1' : tf.Variable(tf.truncated_normal([5,6,1,32]))
	'wc2' : tf.Variable(tf.truncated_normal([5,6,32,64]))
	'wcd1' : tf.Variable(tf.truncated_normal([7*7*64, 1024]))
	'out' : tf.Variable(tf.truncated_normal([1024, n_classes]))
}

biases = {
	'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

## Layers
def conv2d(x, W, b, strides=1):
	x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME') + b
	return tf.nn.relu(x)

def maxpool2d(x, k=2):
	m = tf.nn.max_pool(x, ksize=[1,k,k,1], strides=[1,k,k,1], padding='SAME')
	return m

def conv_net(x, weights, biases, dropout):
	# Layer 1 : 28*28*1 -> 14*14*32
	conv1 = conv2d(x, weights['wc1'], biases['bc1'])
	conv1 = maxpool2d(conv1, k=2)

	# Layer 2 : 14*14*32 -> 7*7*64
	conv2 = conv2d(x, weights['wc2'], biases['bc2'])
	conv2 = maxpool2d(conv2, k=2)

	# Layer 3 : 7*7*64 -> 1024
	fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
	fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout) 

	# Output layer : 1024 -> 10 
	out = tf.add(tf.matmul(fc1, weights['out'], biases['out']))
	return out

## Session
x = tf.placeholder(tf.float32, [None,28,28,1])
y = tf.placeholder(tf.float32, [None, n_classes])
prob = tf.placeholder(tf.float32)

logits = conv_net(x, weights, biases, prob)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

prediction = tf.equal(tf.argmax(logits,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

init = tf.global_variables_initializer()

with tf.Session as sess:
	sess.run(init)

	for e in range(epochs):
		for batch in range(mnist.train.num_examples//batch_size):
			batch_x, batch_y = mnist.train.next_batch(batch_size)
			sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, prob: dropout})
			loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y, prob: 1.})
			valid_acc = sess.run(accuracy, feed_dict={x: mnist.validation.images[:test_valid_size], y: mnist.validation.labels[:test_valid_size], keep_prob: 1.})

			print ('Epoch {:>2}, Batch {:>3} -'
                  'Loss: {:>10.4f} Validation Accuracy: {:.6f}'.format( epoch + 1, batch + 1, loss, valid_acc))

    # Calculate Test Accuracy
    test_acc = sess.run(accuracy, feed_dict={ x: mnist.test.images[:test_valid_size], y: mnist.test.labels[:test_valid_size], prob: 1.})
    print('Testing Accuracy: {}'.format(test_acc))
