from __future__ import print_function

import argparse
import gzip
import json
import os
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def one_hot(labels):
    """this creates a one hot encoding from a flat vector:
    i.e. given y = [0,2,1]
     it creates y_one_hot = [[1,0,0], [0,0,1], [0,1,0]]
    """
    classes = np.unique(labels)
    n_classes = classes.size
    one_hot_labels = np.zeros(labels.shape + (n_classes,))
    for c in classes:
        one_hot_labels[labels == c, c] = 1
    return one_hot_labels


def mnist(datasets_dir='./data'):
    if not os.path.exists(datasets_dir):
        os.mkdir(datasets_dir)
    data_file = os.path.join(datasets_dir, 'mnist.pkl.gz')
    if not os.path.exists(data_file):
        print('... downloading MNIST from the web')
        try:
            import urllib
            urllib.urlretrieve('http://google.com')
        except AttributeError:
            import urllib.request as urllib
        url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        urllib.urlretrieve(url, data_file)

    print('... loading data')
    # Load the dataset
    f = gzip.open(data_file, 'rb')
    try:
        train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
    except TypeError:
        train_set, valid_set, test_set = pickle.load(f)
    f.close()

    test_x, test_y = test_set
    test_x = test_x.astype('float32')
    test_x = test_x.astype('float32').reshape(test_x.shape[0], 28, 28, 1)
    test_y = test_y.astype('int32')
    valid_x, valid_y = valid_set
    valid_x = valid_x.astype('float32')
    valid_x = valid_x.astype('float32').reshape(valid_x.shape[0], 28, 28, 1)
    valid_y = valid_y.astype('int32')
    train_x, train_y = train_set
    train_x = train_x.astype('float32').reshape(train_x.shape[0], 28, 28, 1)
    train_y = train_y.astype('int32')
    print('... done loading data')
    return train_x, one_hot(train_y), valid_x, one_hot(valid_y), test_x, one_hot(test_y)

# Create the neural network
def conv_net(x, filter_size):
    # TF Estimator input is a dict, in case of multiple inputs
    # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    input_layer = tf.reshape(x, shape=[-1, 28, 28, 1])
    # Convolution Layer with 32 filters and a kernel size of 5
    conv1 = tf.layers.conv2d(inputs = input_layer,
                             filters = 16,
                             kernel_size = filter_size,
                             padding = "same",
                             activation=tf.nn.relu)
    # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
    pool1 = tf.layers.max_pooling2d(inputs = conv1,
                                    pool_size = 2,
                                    strides = 2)
    # Convolution Layer with 64 filters and a kernel size of 3
    conv2 = tf.layers.conv2d(inputs = pool1,
                             filters = 16,
                             kernel_size = filter_size,
                             padding = "same",
                             activation=tf.nn.relu)
    # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
    pool2 = tf.layers.max_pooling2d(inputs = conv2,
                                    pool_size = 2,
                                    strides = 2)
    # Flatten the data to a 1-D vector for the fully connected layer
    flat = tf.contrib.layers.flatten(pool2)
    # Fully connected layer (in tf contrib folder for now)
    dense_layer = tf.layers.dense(inputs = flat,
                          units = 128)
    # Output layer, class prediction
    pred = tf.layers.dense(inputs = dense_layer,
                           units = 10)
    return pred


x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')
learning_rate = tf.placeholder('float')

def train_and_validate(x_train, y_train, x_valid, y_valid, num_epochs, lr, num_filters, batch_size, filter_size):
    
    
    filter_size = 3
    batch_size = 128
    num_filters = 16
    lr = [0.1,0.01,0.001,0.0001]
    
    # TODO: train and validate your convolutional neural networks with the provided data and hyperparameters
    
    prediction = conv_net(x, filter_size)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y))
    
    #cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(tf.argmax(prediction, 1), tf.argmax(y, 1)))
    
    #optimizer =  tf.train.GradientDescentOptimizer(lr).minimize(cost)
    logits_train = tf.argmax(prediction, 1)
    labels = tf.argmax(y, 1)
    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    #cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=tf.cast(labels, dtype=tf.int32)))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    #optimizer = tf.train.AdamOptimizer().minimize(cost)
    optimizer =  tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    hm_epochs = 20
    
    val_hist = []
    train_hist = []

    for l in range(len(lr)): 
        val_loss_history = np.zeros(hm_epochs)
        train_loss_history= np.zeros(hm_epochs)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            step = 0
            for step in range(hm_epochs):
                for i in range(int(len(x_train)/ batch_size)):
                    rand_index = np.random.choice(len(x_train), batch_size)
                    epoch_x = x_train[rand_index].reshape([-1,784])
                    epoch_y = y_train[rand_index]
                    sess.run(optimizer, feed_dict = {x: epoch_x , y: epoch_y, learning_rate: lr[l]})
                #if step % 10 == 0:
                # Calculate batch accuracy
                acc = sess.run(accuracy, feed_dict={x: epoch_x, y: epoch_y})
                # Calculate batch loss
                loss = sess.run(cost, feed_dict={x: epoch_x, y: epoch_y})
                
                # Once in every 1000 *train* steps we evaluate the validation set (called test here)
                #if step % 100 == 0:
                val_step = 1
                val_acc = 0
                val_loss = 0
                while val_step * batch_size < len(x_valid):
                    epoch_x = x_valid[val_step*batch_size: ((val_step+1)*batch_size)].reshape([-1,784])
                    epoch_y = y_valid[val_step*batch_size: ((val_step+1)*batch_size)]
                    step_acc, step_loss = sess.run([accuracy, cost], feed_dict={x: epoch_x, y: epoch_y})
                    val_acc += step_acc
                    val_loss += step_loss
                    val_step += 1
                
                val_acc = val_acc / val_step
                val_loss = val_loss / val_step
                #val_acc = sess.run(accuracy, feed_dict={x: epoch_x, y: epoch_y})
                #print("Validation set accuracy: %s" % val_acc)

                val_loss_history[step] = val_loss
                train_loss_history[step] = loss
            print('Learning rate is:', lr[l])
            print('Filter Size is: ', filter_size)
            print("Optimization Finished!")
            print("Testing Accuracy:", \
            sess.run(accuracy, feed_dict={x: x_test.reshape([-1,784]), y: y_test}))
            
            val_hist.append(val_loss_history)
            train_hist.append(train_loss_history)

            print("Val. Loss History is: ", val_loss_history)
            print("Train Loss History is: ", train_loss_history)

                
            epoch_range= np.linspace(0, hm_epochs-1, hm_epochs)
        
            '''plt.plot(epoch_range, np.asarray(val_loss_history), color = 'r', label ="validation loss")
            plt.plot(epoch_range, np.asarray(train_loss_history), color = 'b', label = "training loss")
            plt.xlabel('number of epochs')
            plt.ylabel('learning curve (loss)')
            plt.legend(loc='upper right')
            plt.savefig('valid.png')'''
            print("done!")   
    return val_hist, train_hist  # TODO: Return the validation error after each epoch (i.e learning curve) and your model


def plot_lc (num_epochs, val_hist):
    epoch_range= np.linspace(0, num_epochs-1, num_epochs)
    for i in range(len(val_hist)):
        plt.plot(epoch_range, np.asarray(val_hist[i]))#, label ="validation loss")
        #plt.xlabel('number of epochs')
        #plt.ylabel('learning curve (loss)')
        #plt.legend(loc='upper right')
    plt.savefig('valid_lr.png')


def test(x_test, y_test, model):
    # TODO: test your network here by evaluating it on the test data
    return test_error


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", default="./", type=str, nargs="?",
                        help="Path where the results will be stored")
    parser.add_argument("--input_path", default="./", type=str, nargs="?",
                        help="Path where the data is located. If the data is not available it will be downloaded first")
    parser.add_argument("--learning_rate", default=1e-3, type=float, nargs="?", help="Learning rate for SGD")
    parser.add_argument("--num_filters", default=32, type=int, nargs="?",
                        help="The number of filters for each convolution layer")
    parser.add_argument("--batch_size", default=128, type=int, nargs="?", help="Batch size for SGD")
    parser.add_argument("--epochs", default=12, type=int, nargs="?",
                        help="Determines how many epochs the network will be trained")
    parser.add_argument("--run_id", default=0, type=int, nargs="?",
                        help="Helps to identify different runs of an experiments")
    parser.add_argument("--filter_size", default=3, type=int, nargs="?",
                        help="Filter width and height")
    args = parser.parse_args()

    # hyperparameters
    lr = args.learning_rate
    num_filters = args.num_filters
    batch_size = args.batch_size
    epochs = args.epochs
    filter_size = args.filter_size

    # train and test convolutional neural network
    x_train, y_train, x_valid, y_valid, x_test, y_test = mnist(args.input_path)

    val_hist, train_hist = train_and_validate(x_train, y_train, x_valid, y_valid, epochs, lr, num_filters, batch_size, filter_size)
    print("val_hist_length is:", len(val_hist))
    plot_lc(20, val_hist)
    
    #test_error = test(x_test, y_test, model)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["lr"] = lr
    results["num_filters"] = num_filters
    results["batch_size"] = batch_size
    results["filter_size"] = filter_size
    results["learning_curve"] = learning_curve
    results["test_error"] = test_error

    path = os.path.join(args.output_path, "results")
    os.makedirs(path, exist_ok=True)

    fname = os.path.join(path, "results_run_%d.json" % args.run_id)

    fh = open(fname, "w")
    json.dump(results, fh)
    fh.close()