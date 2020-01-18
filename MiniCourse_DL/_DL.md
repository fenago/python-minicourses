Last Updated on December 11, 2019

Deep learning is a fascinating field of study and the techniques are
achieving world class results in a range of challenging machine learning
problems.

It can be hard to get started in deep learning.

Which library should you use and which techniques should you focus on?

In this post you will discover a 14-part crash course into deep learning
in Python with the easy to use and powerful Keras library.

This mini-course is intended for python machine learning practitioners
that are already comfortable with scikit-learn on the SciPy ecosystem
for machine learning.

Discover how to develop deep learning models for a range of predictive
modeling problems with just a few lines of code [in my new
book](https://machinelearningmastery.com/deep-learning-with-python/),
with 18 step-by-step tutorials and 9 projects.

Let’s get started.

(**Tip**: *you might want to print or bookmark this page so that you can
refer back to it later.*)

-   **Update Mar/2018**: Added alternate link to download the dataset as
    the original appears to have been taken down.

-   **Update Oct/2019**: Updated for Keras 2.3.0.

![](../vero_iure_incidunt_html_bf3e62ac2050280a.jpg)

Applied Deep Learning in Python Mini-Course

Photo by [darkday](https://www.flickr.com/photos/drainrat/15783392494/),
some rights reserved.

**Who Is This Mini-Course For?** {.western style="margin-top: 0in; margin-bottom: 0.13in; border: none; padding: 0in; line-height: 150%; page-break-inside: auto; background: #ffffff; page-break-after: auto"}
--------------------------------

Before we get started, let’s make sure you are in the right place. The
list below provides some general guidelines as to who this course was
designed for.

Don’t panic if you don’t match these points exactly, you might just need
to brush up in one area or another to keep up.

-   **Developers that know how to write a little code**. This means that
    it is not a big deal for you to get things done with Python and know
    how to setup the SciPy ecosystem on your workstation (a
    prerequisite). It does not mean your a wizard coder, but it does
    mean you’re not afraid to install packages and write scripts.

-   **Developers that know a little machine learning**. This means you
    know about the basics of machine learning like cross validation,
    some algorithms and the bias-variance trade-off. It does not mean
    that you are a machine learning PhD, just that you know the
    landmarks or know where to look them up.

This mini-course is not a textbook on Deep Learning.

It will take you from a developer that knows a little machine learning
in Python to a developer who can get results and bring the power of Deep
Learning to your own projects.

**Mini-Course Overview (what to expect)** {.western style="margin-top: 0in; margin-bottom: 0.13in; border: none; padding: 0in; line-height: 150%; page-break-inside: auto; background: #ffffff; page-break-after: auto"}
-----------------------------------------

This mini-course is divided into 14 parts.

Each lesson was designed to take the average developer about 30 minutes.
You might finish some much sooner and other you may choose to go deeper
and spend more time.

You can can complete each part as quickly or as slowly as you like. A
comfortable schedule may be to complete one lesson per day over a two
week period. Highly recommended.

The topics you will cover over the next 14 lessons are as follows:

-   **Lesson 01**: Introduction to Theano.

-   **Lesson 02**: Introduction to TensorFlow.

-   **Lesson 03**: Introduction to Keras.

-   **Lesson 04**: Crash Course in Multi-Layer Perceptrons.

-   **Lesson 05**: Develop Your First Neural Network in Keras.

-   **Lesson 06**: Use Keras Models With Scikit-Learn.

-   **Lesson 07**: Plot Model Training History.

-   **Lesson 08**: Save Your Best Model During Training With
    Checkpointing.

-   **Lesson 09**: Reduce Overfitting With Dropout Regularization.

-   **Lesson 10**: Lift Performance With Learning Rate Schedules.

-   **Lesson 11**: Crash Course in Convolutional Neural Networks.

-   **Lesson 12**: Handwritten Digit Recognition.

-   **Lesson 13**: Object Recognition in Small Photographs.

-   **Lesson 14**: Improve Generalization With Data Augmentation.

This is going to be a lot of fun.

You’re going to have to do some work though, a little reading, a little
research and a little programming. You want to learn deep learning
right?

(**Tip**: *All of the answers these lessons can be found on this blog,
use the search feature*)

Any questions at all, please post in the comments below.

Share your results in the comments.

Hang in there, don’t give up!

### **Need help with Deep Learning in Python?** {.western style="margin-top: 0in; margin-bottom: 0.1in; border: none; padding: 0in; line-height: 150%; page-break-inside: auto; page-break-after: auto"}

Take my free 2-week email course and discover MLPs, CNNs and LSTMs (with
code).

Click to sign-up now and also get a free PDF Ebook version of the
course.

[**Start Your FREE Mini-Course
Now!**](https://machinelearningmastery.leadpages.co/leadbox/142d6e873f72a2%3A164f8be4f346dc/5657382461898752/)

**Lesson 01: Introduction to Theano** {.western style="margin-top: 0in; margin-bottom: 0.13in; border: none; padding: 0in; line-height: 150%; page-break-inside: auto; background: #ffffff; page-break-after: auto"}
-------------------------------------

Theano is a Python library for fast numerical computation to aid in the
development of deep learning models.

At it’s heart Theano is a compiler for mathematical expressions in
Python. It knows how to take your structures and turn them into very
efficient code that uses NumPy and efficient native libraries to run as
fast as possible on CPUs or GPUs.

The actual syntax of Theano expressions is symbolic, which can be
off-putting to beginners used to normal software development.
Specifically, expression are defined in the abstract sense, compiled and
later actually used to make calculations.

In this lesson your goal is to install Theano and write a small example
that demonstrates the symbolic nature of Theano programs.

For example, you can install Theano using pip as follows:

  --- -------------------------
  1   sudo pip install Theano
  --- -------------------------

A small example of a Theano program that you can use as a starting point
is listed below:

  ---- -------------------------------------------------------------------
  1    import theano
       
  2    from theano import tensor
       
  3    \# declare two symbolic floating-point scalars
       
  4    a = tensor.dscalar()
       
  5    b = tensor.dscalar()
       
  6    \# create a simple expression
       
  7    c = a + b
       
  8    \# convert the expression into a callable object that takes (a,b)
       
  9    \# values as input and computes a value for c
       
  10   f = theano.function([a,b], c)
       
  11   \# bind 1.5 to 'a', 2.5 to 'b', and evaluate 'c'
       
  12   result = f(1.5, 2.5)
       
  13   print(result)
  ---- -------------------------------------------------------------------

Learn more about Theano on the [Theano
homepage](http://deeplearning.net/software/theano/).

**Lesson 02: Introduction to TensorFlow** {.western style="margin-top: 0in; margin-bottom: 0.13in; border: none; padding: 0in; line-height: 150%; page-break-inside: auto; background: #ffffff; page-break-after: auto"}
-----------------------------------------

TensorFlow is a Python library for fast numerical computing created and
released by Google. Like Theano, TensorFlow is intended to be used to
develop deep learning models.

With the backing of Google, perhaps used in some of it’s production
systems and used by the Google DeepMind research group, it is a platform
that we cannot ignore.

Unlike Theano, TensorFlow does have more of a production focus with a
capability to run on CPUs, GPUs and even very large clusters.

In this lesson your goal is to install TensorFlow become familiar with
the syntax of the symbolic expressions used in TensorFlow programs.

For example, you can install TensorFlow using pip:

  --- -----------------------------
  1   sudo pip install TensorFlow
  --- -----------------------------

A small example of a TensorFlow program that you can use as a starting
point is listed below:

  ---- ---------------------------------------------------------------
  1    \# Example of TensorFlow library
       
  2    import tensorflow as tf
       
  3    import tensorflow.compat.v1 as tf
       
  4    tf.disable\_v2\_behavior()
       
  5    \# declare two symbolic floating-point scalars
       
  6    a = tf.placeholder(tf.float32)
       
  7    b = tf.placeholder(tf.float32)
       
  8    \# create a simple symbolic expression using the add function
       
  9    add = tf.add(a, b)
       
  10   \# bind 1.5 to 'a', 2.5 to 'b', and evaluate 'c'
       
  11   sess = tf.Session()
       
  12   binding = {a: 1.5, b: 2.5}
       
  13   c = sess.run(add, feed\_dict=binding)
       
  14   print(c)
  ---- ---------------------------------------------------------------

Learn more about TensorFlow on the [TensorFlow
homepage](https://www.tensorflow.org/).

**Lesson 03: Introduction to Keras** {.western style="margin-top: 0in; margin-bottom: 0.13in; border: none; padding: 0in; line-height: 150%; page-break-inside: auto; background: #ffffff; page-break-after: auto"}
------------------------------------

A difficulty of both Theano and TensorFlow is that it can take a lot of
code to create even very simple neural network models.

These libraries were designed primarily as a platform for research and
development more than for the practical concerns of applied deep
learning.

The Keras library addresses these concerns by providing a wrapper for
both Theano and TensorFlow. It provides a clean and simple API that
allows you to define and evaluate deep learning models in just a few
lines of code.

Because of the ease of use and because it leverages the power of Theano
and TensorFlow, Keras is quickly becoming the go-to library for applied
deep learning.

The focus of Keras is the concept of a model. The life-cycle of a model
can be summarized as follows:

1.  Define your model. Create a Sequential model and add configured
    layers.

2.  Compile your model. Specify loss function and optimizers and call
    the compile()\
     function on the model.

3.  Fit your model. Train the model on a sample of data by calling the
    fit() function on\
     the model.

4.  Make predictions. Use the model to generate predictions on new data
    by calling functions such as evaluate() or predict() on the model.

Your goal for this lesson is to install Keras.

For example, you can install Keras using pip:

  --- ------------------------
  1   sudo pip install keras
  --- ------------------------

Start to familiarize yourself with the Keras library ready for the
upcoming lessons where we will implement our first model.

You can learn more about the Keras library on the [Keras
homepage](http://keras.io/).

**Lesson 04: Crash Course in Multi-Layer Perceptrons** {.western style="margin-top: 0in; margin-bottom: 0.13in; border: none; padding: 0in; line-height: 150%; page-break-inside: auto; background: #ffffff; page-break-after: auto"}
------------------------------------------------------

Artificial neural networks are a fascinating area of study, although
they can be intimidating

when just getting started.

The field of artificial neural networks is often just called neural
networks or multi-layer

Perceptrons after perhaps the most useful type of neural network.

The building block for neural networks are artificial neurons. These are
simple computational

units that have weighted input signals and produce an output signal
using an activation function.

Neurons are arranged into networks of neurons. A row of neurons is
called a layer and one

network can have multiple layers. The architecture of the neurons in the
network is often called the network topology.

Once configured, the neural network needs to be trained on your dataset.
The classical and still preferred training algorithm for neural networks
is called stochastic

gradient descent.

![](../vero_iure_incidunt_html_5d664c63585d17f0.png)

Model of a Simple Neuron

Your goal for this lesson is to become familiar with neural network
terminology.

Dig a little deeper into terms like neuron, weights, activation
function, learning rate and more.

**Lesson 05: Develop Your First Neural Network in Keras** {.western style="margin-top: 0in; margin-bottom: 0.13in; border: none; padding: 0in; line-height: 150%; page-break-inside: auto; background: #ffffff; page-break-after: auto"}
---------------------------------------------------------

Keras allows you to develop and evaluate deep learning models in very
few lines of code.

In this lesson your goal is to develop your first neural network using
the Keras library.

Use a standard binary (two-class) classification dataset from the UCI
Machine Learning Repository, like the Pima Indians onset of diabetes or
the [ionosphere
datasets](https://archive.ics.uci.edu/ml/datasets/Ionosphere).

Piece together code to achieve the following:

1.  Load your dataset using NumPy or Pandas.

2.  Define your neural network model and compile it.

3.  Fit your model to the dataset.

4.  Estimate the performance of your model on unseen data.

To give you a massive kick start, below is a complete working example
that you can use as a starting point.

Download the dataset and place it in your current working directory.

-   [Dataset
    File](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv).

-   [Dataset
    Details](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.names).

  ---- -------------------------------------------------------------------------------------
  1    from keras.models import Sequential
       
  2    from keras.layers import Dense
       
  3    \# Load the dataset
       
  4    dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
       
  5    X = dataset[:,0:8]
       
  6    Y = dataset[:,8]
       
  7    \# Define and Compile
       
  8    model = Sequential()
       
  9    model.add(Dense(12, input\_dim=8, activation='relu'))
       
  10   model.add(Dense(8, activation='relu'))
       
  11   model.add(Dense(1, activation='sigmoid'))
       
  12   model.compile(loss='binary\_crossentropy' , optimizer='adam', metrics=['accuracy'])
       
  13   \# Fit the model
       
  14   model.fit(X, Y, epochs=150, batch\_size=10)
       
  15   \# Evaluate the model
       
  16   scores = model.evaluate(X, Y)
       
  17   print("%s: %.2f%%" % (model.metrics\_names[1], scores[1]\*100))
  ---- -------------------------------------------------------------------------------------

Now develop your own model on a different dataset, or adapt this
example.

Learn more about the [Keras API for simple model
development](http://keras.io/models/sequential/).

**Lesson 06: Use Keras Models With Scikit-Learn** {.western style="margin-top: 0in; margin-bottom: 0.13in; border: none; padding: 0in; line-height: 150%; page-break-inside: auto; background: #ffffff; page-break-after: auto"}
-------------------------------------------------

The scikit-learn library is a general purpose machine learning framework
in Python built on top of SciPy.

Scikit-learn excels at tasks such as evaluating model performance and
optimizing model hyperparameters in just a few lines of code.

Keras provides a wrapper class that allows you to use your deep learning
models with scikit-learn. For example, an instance of KerasClassifier
class in Keras can wrap your deep learning model and be used as an
Estimator in scikit-learn.

When using the KerasClassifier class, you must specify the name of a
function that the class can use to define and compile your model. You
can also pass additional parameters to the constructor of the
KerasClassifier class that will be passed to the *model.fit()* call
later, like the number of epochs and batch size.

In this lesson your goal is to develop a deep learning model and
evaluate it using k-fold cross validation.

For example, you can define an instance of the KerasClassifier and the
custom function to create your model as follows:

  ---- ---------------------------------------------------------------------------------
  1    \# Function to create model, required for KerasClassifier
       
  2    def create\_model():
       
  3    \# Create model
       
  4    model = Sequential()
       
  5    ...
       
  6    \# Compile model
       
  7    model.compile(...)
       
  8    return model
       
  9    \
        \
  10   
       \# create classifier for use in scikit-learn
  11   
       model = KerasClassifier(build\_fn=create\_model, nb\_epoch=150, batch\_size=10)
  12   
       \# evaluate model using 10-fold cross validation in scikit-learn
  13   
       kfold = StratifiedKFold(n\_splits=10, shuffle=True, random\_state=seed)
  14   
       results = cross\_val\_score(model, X, Y, cv=kfold)
  ---- ---------------------------------------------------------------------------------

Learn more about using your Keras deep learning models with scikit-learn
on the [Wrappers for the Sciki-Learn API
webpage](http://keras.io/scikit-learn-api/).

**Lesson 07: Plot Model Training History** {.western style="margin-top: 0in; margin-bottom: 0.13in; border: none; padding: 0in; line-height: 150%; page-break-inside: auto; background: #ffffff; page-break-after: auto"}
------------------------------------------

You can learn a lot about neural networks and deep learning models by
observing their performance over time during training.

Keras provides the capability to register callbacks when training a deep
learning model.

One of the default callbacks that is registered when training all deep
learning models is the History callback. It records training metrics for
each epoch. This includes the loss and the accuracy (for classification
problems) as well as the loss and accuracy for the validation dataset,
if one is set.

The history object is returned from calls to the fit() function used to
train the model. Metrics are stored in a dictionary in the history
member of the object returned.

Your goal for this lesson is to investigate the history object and
create plots of model performance during training.

For example, you can print the list of metrics collected by your history
object as follows:

  --- -------------------------------
  1   \# list all data in history
      
  2   history = model.fit(...)
      
  3   print(history.history.keys())
  --- -------------------------------

You can learn more about the [History object and the callback API in
Keras](http://keras.io/callbacks/#history).

**Lesson 08: Save Your Best Model During Training With Checkpointing** {.western style="margin-top: 0in; margin-bottom: 0.13in; border: none; padding: 0in; line-height: 150%; page-break-inside: auto; background: #ffffff; page-break-after: auto"}
----------------------------------------------------------------------

Application checkpointing is a fault tolerance technique for long
running processes.

The Keras library provides a checkpointing capability by a callback API.
The ModelCheckpoint

callback class allows you to define where to checkpoint the model
weights, how the file should

be named and under what circumstances to make a checkpoint of the model.

Checkpointing can be useful to keep track of the model weights in case
your training run is stopped prematurely. It is also useful to keep
track of the best model observed during training.

In this lesson, your goal is to use the ModelCheckpoint callback in
Keras to keep track of the best model observed during training.

You could define a ModelCheckpoint that saves network weights to the
same file each time an improvement is observed. For example:

  --- ---------------------------------------------------------------------------------------------------------------
  1   from keras.callbacks import ModelCheckpoint
      
  2   ...
      
  3   checkpoint = ModelCheckpoint('weights.best.hdf5', monitor='val\_accuracy', save\_best\_only=True, mode='max')
      
  4   callbacks\_list = [checkpoint]
      
  5   \# Fit the model
      
  6   model.fit(..., callbacks=callbacks\_list)
  --- ---------------------------------------------------------------------------------------------------------------

Learn more about using the [ModelCheckpoint callback in
Keras](http://keras.io/callbacks/#modelcheckpoint).

**Lesson 09: Reduce Overfitting With Dropout Regularization** {.western style="margin-top: 0in; margin-bottom: 0.13in; border: none; padding: 0in; line-height: 150%; page-break-inside: auto; background: #ffffff; page-break-after: auto"}
-------------------------------------------------------------

A big problem with neural networks is that they can overlearn your
training dataset.

Dropout is a simple yet very effective technique for reducing dropout
and has proven useful in large deep learning models.

Dropout is a technique where randomly selected neurons are ignored
during training. They are *dropped-out* randomly. This means that their
contribution to the activation of downstream neurons is temporally
removed on the forward pass and any weight updates are not applied to
the neuron on the backward pass.

You can add a dropout layer to your deep learning model using the
Dropout layer class.

In this lesson your goal is to experiment with adding dropout at
different points in your neural network and set to different probability
of dropout values.

For example, you can create a dropout layer with the probability of 20%
and add it to your model as follows:

  --- ----------------------------------
  1   from keras.layers import Dropout
      
  2   ...
      
  3   model.add(Dropout(0.2))
  --- ----------------------------------

You can learn more [about dropout in
Keras](http://keras.io/layers/core/#dropout).

**Lesson 10: Lift Performance With Learning Rate Schedules** {.western style="margin-top: 0in; margin-bottom: 0.13in; border: none; padding: 0in; line-height: 150%; page-break-inside: auto; background: #ffffff; page-break-after: auto"}
------------------------------------------------------------

You can often get a boost in the performance of your model by using a
learning rate schedule.

Often called an adaptive learning rate or an annealed learning rate,
this is a technique where the learning rate used by stochastic gradient
descent changes while training your model.

Keras has a time-based learning rate schedule built into the
implementation of the stochastic gradient descent algorithm in the SGD
class.

When constructing the class, you can specify the decay which is the
amount that your learning rate (also specified) will decrease each
epoch. When using learning rate decay you should bump up your initial
learning rate and consider adding a large momentum value such as 0.8 or
0.9.

Your goal in this lesson is to experiment with the time-based learning
rate schedule built into Keras.

For example, you can specify a learning rate schedule that starts at 0.1
and drops by 0.0001 each epoch as follows:

  --- ---------------------------------------------------------------
  1   from keras.optimizers import SGD
      
  2   ...
      
  3   sgd = SGD(lr=0.1, momentum=0.9, decay=0.0001, nesterov=False)
      
  4   model.compile(..., optimizer=sgd)
  --- ---------------------------------------------------------------

You can learn more about the [SGD class in Keras
here](http://keras.io/optimizers/#sgd).

**Lesson 11: Crash Course in Convolutional Neural Networks** {.western style="margin-top: 0in; margin-bottom: 0.13in; border: none; padding: 0in; line-height: 150%; page-break-inside: auto; background: #ffffff; page-break-after: auto"}
------------------------------------------------------------

Convolutional Neural Networks are a powerful artificial neural network
technique.

They expect and preserve the spatial relationship between pixels in
images by learning internal feature representations using small squares
of input data.

Feature are learned and used across the whole image, allowing for the
objects in your images to be shifted or translated in the scene and
still detectable by the network. It is this reason why this type of
network is so useful for object recognition in photographs, picking out
digits, faces, objects and so on with varying orientation.

There are three types of layers in a Convolutional Neural Network:

1.  **Convolutional Layers** comprised of filters and feature maps.

2.  **Pooling Layers** that down sample the activations from feature
    maps.

3.  **Fully-Connected Layers** that plug on the end of the model and can
    be used to make predictions.

In this lesson you are to familiarize yourself with the terminology used
when describing convolutional neural networks.

This may require a little research on your behalf.

Don’t worry too much about how they work just yet, just learn the
terminology and configuration of the various layers used in this type of
network.

**Lesson 12: Handwritten Digit Recognition** {.western style="margin-top: 0in; margin-bottom: 0.13in; border: none; padding: 0in; line-height: 150%; page-break-inside: auto; background: #ffffff; page-break-after: auto"}
--------------------------------------------

Handwriting digit recognition is a difficult computer vision
classification problem.

The [MNIST
dataset](https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/)
is a standard problem for evaluating algorithms on the problem of
handwriting digit recognition. It contains 60,000 images of digits that
can be used to train a model, and 10,000 images that can be used to
evaluate its performance.

![](../vero_iure_incidunt_html_191e4bbf3ddb4301.png)

Example MNIST images

State of the art results can be achieved on the MNIST problem using
convolutional neural networks. Keras makes loading the MNIST dataset
dead easy.

In this lesson, your goal is to develop a very simple convolutional
neural network for the MNIST problem comprised of one convolutional
layer, one max pooling layer and one dense layer to make predictions.

For example, you can load the MNIST dataset in Keras as follows:

  --- ---------------------------------------------------------------
  1   from keras.datasets import mnist
      
  2   ...
      
  3   (X\_train, y\_train), (X\_test, y\_test) = mnist.load\_data()
  --- ---------------------------------------------------------------

It may take a moment to download the files to your computer.

As a tip, the Keras [Conv2D](http://keras.io/layers/convolutional/)
layer that you will use as your first hidden layer expects image data in
the format width x height x channels, where the MNIST data has 1 channel
because the images are gray scale and a width and height of 28 pixels.
You can easily reshape the MNIST dataset as follows:

  --- -------------------------------------------------------------
  1   X\_train = X\_train.reshape((X\_train.shape[0], 28, 28, 1))
      
  2   X\_test = X\_test.reshape((X\_test.shape[0], 28, 28, 1))
  --- -------------------------------------------------------------

You will also need to one-hot encode the output class value, that Keras
also provides a handy helper function to achieve:

  --- ------------------------------------------------
  1   from keras.utils import np\_utils
      
  2   ...
      
  3   y\_train = np\_utils.to\_categorical(y\_train)
      
  4   y\_test = np\_utils.to\_categorical(y\_test)
  --- ------------------------------------------------

As a final tip, here is a model definition that you can use as a
starting point:

  --- -----------------------------------------------------------------------------------------
  1   model = Sequential()
      
  2   model.add(Conv2D(32, (3, 3), padding='valid', input\_shape=(28, 28, 1),
      
  3   activation='relu'))
      
  4   model.add(MaxPooling2D())
      
  5   model.add(Flatten())
      
  6   model.add(Dense(128, activation='relu'))
      
  7   model.add(Dense(num\_classes, activation='softmax'))
      
  8   model.compile(loss='categorical\_crossentropy', optimizer='adam', metrics=['accuracy'])
  --- -----------------------------------------------------------------------------------------

**Lesson 13: Object Recognition in Small Photographs** {.western style="margin-top: 0in; margin-bottom: 0.13in; border: none; padding: 0in; line-height: 150%; page-break-inside: auto; background: #ffffff; page-break-after: auto"}
------------------------------------------------------

Object recognition is a problem where your model must indicate what is
in a photograph.

Deep learning models achieve state of the art results in this problem
using deep convolutional neural networks.

A popular standard dataset for evaluating models on this type of problem
is called
[CIFAR-10](https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/).
It contains 60,000 small photographs, each of one of 10 objects, like a
cat, ship or airplane.

![](../vero_iure_incidunt_html_29ac3e37bcf24715.png)

Small Sample of CIFAR-10 Images

As with the MNIST dataset, Keras provides a convenient function that you
can use to load the dataset, and it will download it to your computer
the first time you try to load it. The dataset is a 163 MB so it may
take a few minutes to download.

Your goal in this lesson is to develop a deep convolutional neural
network for the CIFAR-10 dataset. I would recommend a repeated pattern
of convolution and pooling layers. Consider experimenting with drop-out
and long training times.

For example, you can load the CIFAR-10 dataset in Keras and prepare it
for use with a convolutional neural network as follows:

  ---- ---------------------------------------------------------------------------
  1    from keras.datasets import cifar10
       
  2    from keras.utils import np\_utils
       
  3    \# load data
       
  4    (X\_train, y\_train), (X\_test, y\_test) = cifar10.load\_data()
       
  5    \# normalize inputs from 0-255 to 0.0-1.0
       
  6    X\_train = X\_train.astype('float32') X\_test = X\_test.astype('float32')
       
  7    X\_train = X\_train / 255.0
       
  8    X\_test = X\_test / 255.0
       
  9    \# one hot encode outputs
       
  10   y\_train = np\_utils.to\_categorical(y\_train)
       
  11   y\_test = np\_utils.to\_categorical(y\_test)
  ---- ---------------------------------------------------------------------------

**Lesson 14: Improve Generalization With Data Augmentation** {.western style="margin-top: 0in; margin-bottom: 0.13in; border: none; padding: 0in; line-height: 150%; page-break-inside: auto; background: #ffffff; page-break-after: auto"}
------------------------------------------------------------

Data preparation is required when working with neural network and deep
learning models.

Increasingly [data augmentation is also
required](https://machinelearningmastery.com/how-to-configure-image-data-augmentation-when-training-deep-learning-neural-networks/)
on more complex object recognition tasks. This is where images in your
dataset are modified with random flips and shifts. This in essence makes
your training dataset larger and helps your model to generalize the
position and orientation of objects in images.

Keras provides an image augmentation API that will create modified
versions of images in your dataset just-in-time. The
[ImageDataGenerator](http://keras.io/preprocessing/image/) class can be
used to define the image augmentation operations to perform which can be
fit to a dataset and then used in place of your dataset when training
your model.

Your goal with this lesson is to experiment with the Keras image
augmentation API using a dataset you are already familiar with from a
previous lesson like MNIST or CIFAR-10.

For example, the example below creates random rotations of up to 90
degrees of images in the MNIST dataset.

  ---- ----------------------------------------------------------------------------
  1    \# Random Rotations
       
  2    from keras.datasets import mnist
       
  3    from keras.preprocessing.image import ImageDataGenerator
       
  4    from matplotlib import pyplot
       
  5    \# load data
       
  6    (X\_train, y\_train), (X\_test, y\_test) = mnist.load\_data()
       
  7    \# reshape to be [samples][pixels][width][height]
       
  8    X\_train = X\_train.reshape((X\_train.shape[0], 28, 28, 1))
       
  9    X\_test = X\_test.reshape((X\_test.shape[0], 28, 28, 1))
       
  10   \# convert from int to float
       
  11   X\_train = X\_train.astype('float32')
       
  12   X\_test = X\_test.astype('float32')
       
  13   \# define data preparation
       
  14   datagen = ImageDataGenerator(rotation\_range=90)
       
  15   \# fit parameters from data
       
  16   datagen.fit(X\_train)
       
  17   \# configure batch size and retrieve one batch of images
       
  18   for X\_batch, y\_batch in datagen.flow(X\_train, y\_train, batch\_size=9):
       
  19   \# create a grid of 3 \* 3 images
       
  20   for i in range(0, 9):
       
  21   pyplot.subplot(330 + 1 + i)
       
  22   pyplot.imshow(X\_batch[i].reshape(28, 28), cmap=pyplot.get\_cmap('gray'))
       
  23   \# show the plot
       
  24   pyplot.show()
       
  25   break
  ---- ----------------------------------------------------------------------------

You can learn more about the [Keras image augmentation
API](http://keras.io/preprocessing/image/).

**Deep Learning Mini-Course Review** {.western style="margin-top: 0in; margin-bottom: 0.13in; border: none; padding: 0in; line-height: 150%; page-break-inside: auto; background: #ffffff; page-break-after: auto"}
------------------------------------

Congratulations, you made it. Well done!

Take a moment and look back at how far you have come:

-   You discovered deep learning libraries in python including the
    powerful numerical libraries Theano and TensorFlow and the easy to
    use Keras library for applied deep learning.

-   You built your first neural network using Keras and learned how to
    use your deep learning models with scikit-learn and how to retrieve
    and plot the training history for your models.

-   You learned about more advanced techniques such as dropout
    regularization and learning rate schedules and how you can use these
    techniques in Keras.

-   Finally, you took the next step and learned about and developed
    convolutional neural networks for complex computer vision tasks and
    learned about augmentation of image data.

Don’t make light of this, you have come a long way in a short amount of
time. This is just the beginning of your journey with deep learning in
python. Keep practicing and developing your skills.

Did you enjoy this mini-course? Do you have any questions or sticking
points?

Leave a comment and let me know.

\
