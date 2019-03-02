# Tested with Python 3.5.2 with tensorflow and matplotlib installed.
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot = True)
def gen_image(arr):
    two_d = (np.reshape(arr, (28, 28)) * 255).astype(np.uint8)
    # plt.imshow(two_d, interpolation='nearest')
    # return plt
    return two_d



# Get a batch of two random images and show in a pop-up window.
batch_xs, batch_ys = mnist.test.next_batch(20)
# gen_image(batch_xs[0]).show()
# gen_image(batch_xs[1]).show()


import tensorflow as tf

n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)

# Features and Labels
features = tf.placeholder(tf.float32, [None, n_input])
labels = tf.placeholder(tf.float32, [None, n_classes])

n_hidden_layer = 256

# Store layers weight & bias
weights = {
    'hidden_layer': tf.Variable(tf.random_normal([n_input, n_hidden_layer])),
    'out': tf.Variable(tf.random_normal([n_hidden_layer, n_classes]))
}
biases = {
    'hidden_layer': tf.Variable(tf.random_normal([n_hidden_layer])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Logits - xW + b
layer_1 = tf.add(tf.matmul(features, weights['hidden_layer']), biases['hidden_layer'])

layer_1 = tf.nn.relu(layer_1)

logits = tf.add(tf.matmul(layer_1, weights['out']), biases['out'])

y = tf.nn.softmax(tf.matmul(layer_1, weights['out']) + biases['out'])


# import math

save_file = './train_model.ckpt'
batch_size = 128
n_epochs = 100

saver = tf.train.Saver()


# Launch the graph
with tf.Session() as sess:
    saver.restore(sess, save_file)

    prediction = sess.run(y,        
        feed_dict={features: batch_xs})

# for pred in prediction:
#     print(np.argmax(pred))

ax = []
columns = 5
rows = 4
fig = plt.figure()
for i in range(1, columns*rows +1):
    img = gen_image(batch_xs[i-1])
    ax.append(fig.add_subplot(rows, columns, i))
    ax[-1].set_title(np.argmax(prediction[i-1]))
    plt.imshow(img)
plt.show()


