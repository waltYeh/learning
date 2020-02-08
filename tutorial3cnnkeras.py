import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Input
from tensorflow.python.keras.layers import Reshape, MaxPooling2D
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten
# Config of NN
# Convolutional Layer 1.
filter_size1 = 5          # Convolution filters are 5 x 5 pixels.
num_filters1 = 16         # There are 16 of these filters.

# Convolutional Layer 2.
filter_size2 = 5          # Convolution filters are 5 x 5 pixels.
num_filters2 = 36         # There are 36 of these filters.

# Fully-connected layer.
fc_size = 128             # Number of neurons in fully-connected layer.

img_size=28
img_size_flat=img_size*img_size
img_shape=(img_size, img_size)
img_shape_full=(img_size, img_size,1)
num_classes=10
batch_size = 100
# Number of colour channels for the images: 1 channel for gray-scale.
num_channels = 1
# from mnist import MNIST
# data = MNIST(data_dir="data/MNIST/")
# If your computer crashes or becomes very slow 
# because you run out of RAM, then you may try and lower
# this number, but you may then need to perform more optimization iterations
train_batch_size = 64
# Split the test-set into smaller batches of this size.
test_batch_size = 256
total_iterations = 0
path_model = 'model_functional.keras'
using_seq_model = False
using_fun_model = False
reload_model = True
def plot_image(image):
    plt.imshow(image.reshape(img_shape),
               interpolation='nearest',
               cmap='binary')

    plt.show()
def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9
    
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()
def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))
def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))
def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters,        # Number of filters.
                   use_pooling=True):  # Use 2x2 max-pooling.

    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape)

    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)

    # Create the TensorFlow operation for convolution.
    # Note the strides are set to 1 in all dimensions.
    # The first and last stride must always be 1,
    # because the first is for the image-number and
    # the last is for the input-channel.
    # But e.g. strides=[1, 2, 2, 1] would mean that the filter
    # is moved 2 pixels across the x- and y-axis of the image.
    # The padding is set to 'SAME' which means the input image
    # is padded with zeroes so the size of the output is the same.
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    # Add the biases to the results of the convolution.
    # A bias-value is added to each filter-channel.
    layer += biases

    # Use pooling to down-sample the image resolution?
    if use_pooling:
        # This is 2x2 max-pooling, which means that we
        # consider 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    # Rectified Linear Unit (ReLU).
    # It calculates max(x, 0) for each input pixel x.
    # This adds some non-linearity to the formula and allows us
    # to learn more complicated functions.
    layer = tf.nn.relu(layer)

    # Note that ReLU is normally executed before the pooling,
    # but since relu(max_pool(x)) == max_pool(relu(x)) we can
    # save 75% of the relu-operations by max-pooling first.

    # We return both the resulting layer and the filter-weights
    # because we will plot the weights later.
    return layer, weights
def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features = layer_shape[1:4].num_elements()
    
    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.
    return layer_flat, num_features
def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=True): # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer
def optimize(optimizer, data, session, x, y_true, accuracy, num_iterations):
    global total_iterations
    start_time = time.time()
    for i in range(total_iterations, total_iterations + num_iterations):
        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch = data.train.next_batch(batch_size=train_batch_size)
        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}
        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)
        # Print status every 100 iterations.
        if i % 100 == 0:
            # Calculate the accuracy on the training-set.
            acc = session.run(accuracy, feed_dict=feed_dict_train)
            # Message for printing.
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"
            # Print it.
            print(msg.format(i + 1, acc))
    # Update the total number of iterations performed.
    total_iterations += num_iterations
    # Ending time.
    end_time = time.time()
    # Difference between start and end-times.
    time_dif = end_time - start_time
    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
def plot_example_errors(data, cls_pred, correct):
    # This function is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # correct is a boolean array whether the predicted class
    # is equal to the true class for each image in the test-set.

    # Negate the boolean array.
    incorrect = (correct == False)
    
    # Get the images from the test-set that have been
    # incorrectly classified.
    images = data.test.images[incorrect]
    
    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = data.test.labels[incorrect]
    # print(len(images))
    # print(len(cls_true))
    # print(len(cls_pred))
    # print(correct)
    # print(incorrect)
    # Plot the first 9 images.
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])

def plot_confusion_matrix(data, cls_pred):
    # This is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Get the true classifications for the test-set.
    cls_true = data.test.cls
    
    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)

    # Print the confusion matrix as text.
    print(cm)

    # Plot the confusion matrix as an image.
    plt.matshow(cm)

    # Make various adjustments to the plot.
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()
def print_test_accuracy(data, session, x, y_true, y_pred_cls,
                        show_example_errors=False,
                        show_confusion_matrix=False):

    # Number of images in the test-set.
    num_test = len(data.test.labels)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_test:
        # The ending index for the next batch is denoted j.
        j = min(i + test_batch_size, num_test)

        # Get the images from the test-set between index i and j.
        images = data.test.images[i:j, :]

        # Get the associated labels.
        labels = data.test.labels[i:j, :]

        # Create a feed-dict with these images and labels.
        feed_dict = {x: images,
                     y_true: labels}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Convenience variable for the true class-numbers of the test-set.
    cls_true = data.test.cls

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / num_test

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example eRrORS:")
        plot_example_errors(data, cls_pred=cls_pred, correct=correct)

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(data, cls_pred=cls_pred)
def plot_conv_weights(weights, input_channel=0):
    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = np.min(weights)
    w_max = np.max(weights)

    # Number of filters used in the conv. layer.
    num_filters = weights.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = int(math.ceil(math.sqrt(num_filters)))
    
    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot all the filter-weights.
    for i, ax in enumerate(axes.flat):
        # Only plot the valid filter-weights.
        if i<num_filters:
            # Get the weights for the i'th filter of the input channel.
            # See new_conv_layer() for details on the format
            # of this 4-dim tensor.
            img = weights[:, :, input_channel, i]

            # Plot image.
            ax.imshow(img, vmin=w_min, vmax=w_max,
                      interpolation='nearest', cmap='seismic')
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()
def plot_conv_output(values):
    # Number of filters used in the conv. layer.
    num_filters = values.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = int(math.ceil(math.sqrt(num_filters)))
    
    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot the output images of all the filters.
    for i, ax in enumerate(axes.flat):
        # Only plot the images for valid filters.
        if i<num_filters:
            # Get the output image of using the i'th filter.
            img = values[0, :, :, i]

            # Plot image.
            ax.imshow(img, interpolation='nearest', cmap='binary')
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()
def main():
    from tensorflow.examples.tutorials.mnist import input_data
    data=input_data.read_data_sets("data/MNIST/", one_hot=True)
    print("Size of:")
    print("- Training-set:\t\t{}".format(len(data.train.labels)))
    print("- Test-set:\t\t{}".format(data.test.labels))
    # Get the first images from the test-set.
    data.test.cls = np.array([label.argmax() for label in data.test.labels])

#   images = data.x_test[0:9]
    images = data.test.images[0:9]
    #Get the true classes 
#   cls_true = data.y_test_cls[0:9]
    cls_true = data.test.cls[0:9]
    # Plot the images and labels using our helper-function above.
#    plot_images(images=images, cls_true=cls_true)

    if using_seq_model:

        model = Sequential()
        # Add an input layer which is similar to a feed_dict in TensorFlow.
        # Note that the input-shape must be a tuple containing the image-size.
        model.add(InputLayer(input_shape=(img_size_flat,)))
        # The input is a flattened array with 784 elements,
        # but the convolutional layers expect images with shape (28, 28, 1)
        model.add(Reshape(img_shape_full))
        # x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
        # x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
        # y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
        # y_true_cls = tf.argmax(y_true, axis=1)


        # First convolutional layer with ReLU-activation and max-pooling.
        model.add(Conv2D(kernel_size=5, strides=1, filters=16, padding='same',
                     activation='relu', name='layer_conv1'))
        model.add(MaxPooling2D(pool_size=2, strides=2))
        # layer_conv1, weights_conv1 = new_conv_layer(input=x_image,
        #                num_input_channels=num_channels,
        #                filter_size=filter_size1,
        #                num_filters=num_filters1,
        #                use_pooling=True)
        # print (layer_conv1)

        # Second convolutional layer with ReLU-activation and max-pooling.
        model.add(Conv2D(kernel_size=5, strides=1, filters=36, padding='same',
                     activation='relu', name='layer_conv2'))
        model.add(MaxPooling2D(pool_size=2, strides=2))

        # layer_conv2, weights_conv2 = new_conv_layer(input=layer_conv1,
        #                num_input_channels=num_filters1,
        #                filter_size=filter_size2,
        #                num_filters=num_filters2,
        #                use_pooling=True)
        # print (layer_conv2)

        # Flatten the 4-rank output of the convolutional layers
        # to 2-rank that can be input to a fully-connected / dense layer.
        model.add(Flatten())

        # layer_flat, num_features = flatten_layer(layer_conv2)
        # print (layer_flat)
        # print (num_features)

        # First fully-connected / dense layer with ReLU-activation.
        model.add(Dense(128, activation='relu'))
        # layer_fc1 = new_fc_layer(input=layer_flat,
        #                      num_inputs=num_features,
        #                      num_outputs=fc_size,
        #                      use_relu=True)
        # print (layer_fc1)

        # Last fully-connected / dense layer with softmax-activation
        # for use in classification.
        model.add(Dense(num_classes, activation='softmax'))
        # layer_fc2 = new_fc_layer(input=layer_fc1,
        #                      num_inputs=fc_size,
        #                      num_outputs=num_classes,
        #                      use_relu=False)
        # print(layer_fc2)
        # y_pred = tf.nn.softmax(layer_fc2)
        # y_pred_cls = tf.argmax(y_pred, axis=1)
        # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
        #                                                     labels=y_true)
        # cost = tf.reduce_mean(cross_entropy)
        from tensorflow.python.keras.optimizers import Adam

        optimizer = Adam(lr=1e-3)
        model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

        # optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
        # correct_prediction = tf.equal(y_pred_cls, y_true_cls)
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # session = tf.Session()
        # session.run(tf.global_variables_initializer())
        model.fit(x=data.train.images,
              y=data.train.labels,
              epochs=1, batch_size=128)
        result = model.evaluate(x=data.test.images,
                            y=data.test.labels)
        print('')
        for name, value in zip(model.metrics_names, result):
            print(name, value)
            print("{0}: {1:.2%}".format(model.metrics_names[1], result[1]))
        

        # `save_model` requires h5py
        model.save(path_model)


        del model
    if using_fun_model:
        # Create an input layer which is similar to a feed_dict in TensorFlow.
        # Note that the input-shape must be a tuple containing the image-size.
        inputs = Input(shape=(img_size_flat,))

        # Variable used for building the Neural Network.
        net = inputs

        # The input is an image as a flattened array with 784 elements.
        # But the convolutional layers expect images with shape (28, 28, 1)
        net = Reshape(img_shape_full)(net)

        # First convolutional layer with ReLU-activation and max-pooling.
        net = Conv2D(kernel_size=5, strides=1, filters=16, padding='same',
                     activation='relu', name='layer_conv1')(net)
        net = MaxPooling2D(pool_size=2, strides=2)(net)

        # Second convolutional layer with ReLU-activation and max-pooling.
        net = Conv2D(kernel_size=5, strides=1, filters=36, padding='same',
                     activation='relu', name='layer_conv2')(net)
        net = MaxPooling2D(pool_size=2, strides=2)(net)

        # Flatten the output of the conv-layer from 4-dim to 2-dim.
        net = Flatten()(net)

        # First fully-connected / dense layer with ReLU-activation.
        net = Dense(128, activation='relu')(net)

        # Last fully-connected / dense layer with softmax-activation
        # so it can be used for classification.
        net = Dense(num_classes, activation='softmax')(net)

        # Output of the Neural Network.
        outputs = net

        from tensorflow.python.keras.models import Model
        model2 = Model(inputs=inputs, outputs=outputs)
        model2.compile(optimizer='rmsprop',
               loss='categorical_crossentropy',
               metrics=['accuracy'])
        model2.fit(x=data.train.images,
           y=data.train.labels,
           epochs=1, batch_size=128)
        result = model2.evaluate(x=data.test.images,
                         y=data.test.labels)
        print('')
        for name, value in zip(model2.metrics_names, result):
            print(name, value)
            print("{0}: {1:.2%}".format(model2.metrics_names[1], result[1]))
        

        # `save_model` requires h5py
        model2.save(path_model)

    if reload_model:

        from tensorflow.python.keras.models import load_model
        model3 = load_model(path_model)




        #images = data.x_test[0:9]
        images = data.test.images[0:9]
        #cls_true = data.y_test_cls[0:9]
        cls_true = data.test.labels[0:9]
        y_pred = model3.predict(x=images)
        cls_pred = np.argmax(y_pred, axis=1)
        plot_images(images=images,
                cls_true=cls_true,
                cls_pred=cls_pred)

        y_pred = model3.predict(x=data.test.images)
        cls_pred = np.argmax(y_pred, axis=1)
        cls_true = data.test.cls
        correct = (cls_true == cls_pred)
        plot_example_errors(data, cls_pred=cls_pred, correct=correct)

        model3.summary()
        layer_input = model3.layers[0]
        layer_conv1 = model3.layers[2]
        print(layer_conv1)
        layer_conv2 = model3.layers[4]
        weights_conv1 = layer_conv1.get_weights()[0]
        print(weights_conv1.shape)
        plot_conv_weights(weights=weights_conv1, input_channel=0)
        weights_conv2 = layer_conv2.get_weights()[0]
        plot_conv_weights(weights=weights_conv2, input_channel=0)
        image1 = data.test.images[0]
        plot_image(image1)

        from tensorflow.python.keras import backend as K
        output_conv1 = K.function(inputs=[layer_input.input],
                          outputs=[layer_conv1.output])
        layer_output1 = output_conv1([[image1]])[0]
        print(layer_output1.shape)
        plot_conv_output(values=layer_output1)

        from tensorflow.python.keras.models import Model
        output_conv2 = Model(inputs=layer_input.input,
                     outputs=layer_conv2.output)
        layer_output2 = output_conv2.predict(np.array([image1]))
        layer_output2.shape
        plot_conv_output(values=layer_output2)
    # global total_iterations
    # total_iterations = 0

    # print_test_accuracy(data, session, x, y_true, y_pred_cls, False, False)
    # optimize(optimizer, data, session, x, y_true, accuracy, num_iterations=1)
    # print_test_accuracy(data, session, x, y_true, y_pred_cls, False, False)
    # optimize(optimizer, data, session, x, y_true, accuracy, num_iterations=99)
    # print_test_accuracy(data, session, x, y_true, y_pred_cls, False, True)
    # optimize(optimizer, data, session, x, y_true, accuracy, num_iterations=900)
    # print_test_accuracy(data, session, x, y_true, y_pred_cls, False, True)
    # session.close()
main()