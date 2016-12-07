from keras.datasets import cifar10

from sklearn.model_selection import train_test_split

import numpy as np
import cv2
import tensorflow as tf
from random import shuffle
import time
from matplotlib import pyplot as plt

# Parameters
learning_rate = 0.001
iterations = 1000
train_batch_size = 50
test_batch_size = 100
display_step = 10

(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

train = {}
train['features'] = X_train
train['labels'] = Y_train
test = {}
test['features'] = X_test
test['labels'] = Y_test

### Preprocess the data here.
### Feel free to use as many code cells as needed.

## Helper function: convert an np.ndarray image from RGB to grayscale using OpenCV
def rgb_to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# # Color to grey
grey_train = {}
grey_train['features'] = []
for i in range(len(X_train)):
    grey_train['features'].append(rgb_to_gray(X_train[i]))
train['features'] = grey_train['features']

grey_test = {}
grey_test['features'] = []
for i in range(len(X_test)):
    grey_test['features'].append(rgb_to_gray(X_test[i]))
test['features'] = grey_test['features']


## Normalize: scale from 0 to 1
def normalize(image_list):
    normalized_train = {}
    normalized_train['features'] = []
    for i in range(len(image_list)):
        normalized_train['features'].append((image_list[i]) / 256)
    return normalized_train['features']

normalized_train = {}
normalized_train['features'] = normalize(train['features'])
train['features'] = normalized_train['features']

normalized_test = {}
normalized_test['features'] = normalize(test['features'])
test['features'] = normalized_test['features']


### To start off let's do a basic data summary.
n_train = len(train['features'])
print("n_train: {}".format(n_train))
n_test = len(test['features'])
print("n_test: {}".format(n_test))
image_shape = train['features'][0].shape
n_classes = 10


# Stolen helper function for one_hot encoding
# lives in tensorflow.contrib.learn.python.learn.datasets.mnist
def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

one_hot_train_labels = dense_to_one_hot(train['labels'], n_classes)
one_hot_test_labels = dense_to_one_hot(test['labels'], n_classes)

# Shuffle training set
index = list(range(0, len(train['features']) - 1))
shuffle(index)

shuffled_features = []
shuffled_labels = []
for i in index:
    shuffled_features.append(train['features'][i])
    shuffled_labels.append(one_hot_train_labels[i])

# Create batches
total_batch = int(n_train / train_batch_size) + 1
feature_batches = np.array_split(shuffled_features, total_batch)
label_batches = np.array_split(shuffled_labels, total_batch)

n_input = image_shape[0] * image_shape[1]  # traffic sign input (img shape: 32*32)


# The actual CNN itself

# Define some helper functions for creating new variables
def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def new_biases(length):
    return tf.Variable(tf.constant(0.0, shape=[length]))


def new_conv_layer(input,               # Previous layer
                   num_input_channels,  # Num channels from previous layer
                   filter_size,         # Width x height of each filter, if 5x5 enter 5 here
                   num_filters,         # Number of filters
                   use_pooling=True,    # Use 2x2 max pooling or not
                   use_dropout=True):   # Use dropout?

    # shape determined by Tensorflow API
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create some new weights for the shape above and initialize them randomly
    weights = new_weights(shape=shape)

    # Create one bias for each filter
    biases = new_biases(length=num_filters)

    # Create Tensorflow convolution operation
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],  # first and last stride must always be 1
                         padding='SAME')        # padding: what to do at edge of image

    # Add biases to the reuslts of convolution:
    layer += biases

    # Use pooling if indicated:
    if use_pooling:
        layer = tf.nn.max_pool(value=layer,
                              ksize=[1, 2, 2, 1],
                              strides=[1,2,2,1],
                              padding='SAME')

    # Then use a RELU to introduce some non-linearity
    layer = tf.nn.relu(layer)

    # ReLU is normally executed before pooling
    # but relu(max_pool(x)) == max_pool(relu(x))
    # So would rather run ReLU on a smaller piece (1x1 as opposed to 2x2)

    # Use Dropout?
    if use_dropout:
        layer = tf.nn.dropout(layer, keep_prob)

    # return both layer and filter weights for later use when running the session
    return layer, weights


# Helper function to flatten a layer, i.e. when feeding form a conv layer into a fully connected
def flatten_layer(layer):
    # Get shape of input
    input_shape = layer.get_shape()

    # format of shape should be [num_images, img_height, img_width, num_channels]
    # total # of features is therefore img_height * img_width * num_channels; grab this
    num_features = input_shape[1:4].num_elements()

    # flatten to 2D, leaving the first dimension open
    layer_flat = tf.reshape(layer, [-1, num_features])

    return layer_flat, num_features


# Helper function to make a fully connected layer
def new_fc_layer(input,             # previous layer
                 num_inputs,        # number of inputs from previous layer
                 num_outputs,       # of outputs
                 use_relu=True,     # Use ReLU?
                 use_dropout=True): # Use dropout?

    # Create new weights and biases
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Layer is matrix mult of inputs by weights, plus bias
    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    # Use Dropout?
    if use_dropout:
        layer = tf.nn.dropout(layer, keep_prob)

    return layer

# Layer configurations
# -------------------------
# Conv layer 1
filter_size1 = 3
num_filters1 = 36

# Conv layer 2
filter_size2 = 3
num_filters2 = 30

# Conv layer 3
filter_size3 = 3
num_filters3 = 24

# Fully connected layer 1
fc_size1 = 128

# Fully connected layer 2
fc_size2 = 64

# ------------------
# Get some image dimensions
img_size = image_shape[0]
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)
num_channels = 1        # Grayscale so only one channel
num_classes = n_classes

# tf Graph input
x = tf.placeholder(tf.float32, shape=[None, img_size, img_size], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, n_classes], name='y_true')
# convert from one-hot to the class number
y_true_cls = tf.argmax(y_true, dimension=1)
# for dropout
keep_prob = tf.placeholder("float")

# Make conv layer 1, takes in image
layer_conv1, weights_conv1 = new_conv_layer(input=x_image,
                                            num_input_channels = num_channels,
                                            filter_size=filter_size1,
                                            num_filters=num_filters1,
                                            use_pooling=True,
                                            use_dropout=False)

# Make conv layer 2, takes in output of conv layer 1
layer_conv2, weights_conv2 = new_conv_layer(input=layer_conv1,
                                            num_input_channels = num_filters1,
                                            filter_size=filter_size2,
                                            num_filters=num_filters2,
                                            use_pooling=False,
                                            use_dropout=False)

# Make conv layer 3, takes in output of conv layer 2
layer_conv3, weights_conv3 = new_conv_layer(input=layer_conv2,
                                            num_input_channels = num_filters2,
                                            filter_size=filter_size3,
                                            num_filters=num_filters3,
                                            use_pooling=False,
                                            use_dropout=False)


# Take in output of conv 4, make the flat layer
layer_flat, num_features = flatten_layer(layer_conv3)

# Make fully connected layer 1, takes in output of flat layer, output is # of neurons
layer_fc1 = new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=fc_size1,
                         use_relu = True,
                         use_dropout = False)

# Make fully connected layer 2, which takes in fc_size1 inputs and outputs fc_size2
layer_fc2 = new_fc_layer(input = layer_fc1,
                         num_inputs = fc_size1,
                         num_outputs = fc_size2,
                         use_relu=True,
                         use_dropout=True)

# Make last layer, which takes in fc_size2 inputs and outputs a vector of logits
logits = new_fc_layer(input = layer_fc2,
                         num_inputs = fc_size2,
                         num_outputs = num_classes,
                         use_relu=False,  # Don't use ReLU on final layer; pass to a softmax
                      use_dropout = False) # No dropout on final layer


# pass logits into softmax, get predictions out in the form of probabilities
y_pred = tf.nn.softmax(logits)  # DON'T FEED THIS INTO tf.nn.softmax_cross_entropy_with_logits()

top_k_softmax = tf.nn.top_k(y_pred, k=5, sorted=True)

y_pred_cls = tf.argmax(y_pred, dimension=1)

# # Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y_true))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# DEFINE SOME PERFORMANCE MEASURES
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
# this creates a vector of trues and falses
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# turn previous vector into a % value; this returns TRAINING accuracy

def print_test_accuracy(session, X_test, Y_test, onehot):
    # Number of images in the test-set.
    num_test = len(X_test)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    # The starting index for the next batch is denoted i.
    print("Calculating test accuracy...")
    i = 0

    while i < num_test:
        # The ending index for the next batch is denoted j.
        j = min(i + test_batch_size, num_test)

        # Get the images from the test-set between index i and j.
        images = X_test[i:j]

        # Get the associated labels.
        labels = onehot[i:j]

        # Create a feed-dict with these images and labels.
        feed_dict = {x: images,
                     y_true: labels,
                     keep_prob: 1} # No dropout when calculating accuracy

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Convenience variable for the true class-numbers of the test-set.
    cls_true = Y_test
    # print(len(Y_test))

    # reshape on cls_true to make it match cls_pred
    cls_true = np.reshape(cls_true, cls_pred.shape)

    # Create a boolean array whether each image is correctly classified.
    # some debug steps
    correct = (cls_true == cls_pred)
    # print("cls_true type: {}".format(type(cls_true)))
    # print("cls_true shape: {}".format(cls_true.shape))
    # print("cls_true: {}".format(cls_true))
    # print("cls_pred type: {}".format(type(cls_pred)))
    # print("cls_pred shape: {}".format(cls_pred.shape))
    # print("cls_pred: {}".format(cls_pred))
    # print(correct)

    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / num_test

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))


# initialize variables, create the session
init = tf.initialize_all_variables()
sess = tf.Session()

# # Launch the graph
start_time = time.time()
sess.run(init)


# Training
def train(sess, iterations):
    for i in range(0, iterations):
        # grab a batch
        batch_x = feature_batches[i % total_batch]
        batch_y = label_batches[i % total_batch] # hacky solution for making sure we always have a batch
        # Run optimization op (backprop) and cost op (to get loss value)
        sess.run(optimizer, feed_dict={x: batch_x,
                                       y_true: batch_y,
                                       keep_prob: 0.5})
        # Display logs per epoch step
        if i % display_step == 0:
            c = sess.run(cost, feed_dict={x: batch_x,
                                          y_true: batch_y,
                                          keep_prob: 1}) # no dropout when calculating cost or acc
            acc = sess.run(accuracy, feed_dict={x: batch_x,
                                                y_true: batch_y,
                                                keep_prob: 1}) # no dropout when calculating cost or acc
            print("Iteration {}:\tCost={:.5},\tTraining accuracy={:.1%}".format(i, c, acc))
            # sys.stdout.write("\033[F")  # uncomment this to not print out massive list while training
    print("Optimization Finished!")
    print("--- %s seconds ---" % (time.time() - start_time))
    print_test_accuracy(sess, test['features'], Y_test, one_hot_test_labels)


# place one .jpg or png image inside the Tensorflow session after training
# classify it, output the softmax probabilities
def classify(session, image_name, image, true_dense_label, top_k=False):
    # label must be a numpy array
    # create feed_dict of one image
    img_dict = {x: image,
                 y_true: dense_to_one_hot(true_dense_label, n_classes),
                 keep_prob: 1.0}
    y_pred_img = session.run(y_pred, feed_dict=img_dict)
    if top_k:
        softmax_values, softmax_indices = session.run(top_k_softmax, feed_dict=img_dict)
        print('-' * 5)
        print(image_name)
        print('-'*5)
        for i in range(0, len(softmax_indices[0])):
            print("Rank: {}\t Softmax: {:.5f}\t Sign: {}".format(i+1, softmax_values[0][i],
                                                                 softmax_indices[0][i]))
        print("")
        return y_pred_img, softmax_indices[0]
    else:
        return y_pred_img, None


# Test on my own image
def predict(sess, image, top_k=False):
    # returns label in dense format
    img = cv2.imread(image)
    gray_img = rgb_to_gray(img)
    normal_img = normalize([gray_img])
    label = [0] ## meaningless dummy label to feed into dict
    # Must feed in numpy arrays of lists, even if only single items
    softmax_img, top_k_labels = classify(sess, image, np.asarray(normal_img), np.asarray(label), top_k=top_k)
    dense_label = np.argmax(softmax_img)
    text_label = np.argmax(softmax_img)
    softmax_prob = softmax_img[0][np.argmax(softmax_img)]
    plt.imshow(gray_img, cmap="gray")
    plt.title("Predicted label: {} ({}) prob: {}".format(text_label, dense_label, softmax_prob))
    plt.show()
    if top_k == True:
        return top_k_labels
    else:
        return dense_label


# notes to self of stuff to do:
# 0) FILL IN PYTHON NOTEBOOK
# 1) k top predictions, probabilities
# 2) upsampling to make categories balanced, + jitter
# 3) Add a final test accuracy


# Run stuff here
train(sess, iterations)