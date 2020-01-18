**Who Is This Crash-Course For?** {.western style="margin-top: 0in; margin-bottom: 0.13in; border: none; padding: 0in; line-height: 150%; page-break-inside: auto; background: #ffffff; page-break-after: auto"}
---------------------------------

Before we get started, let’s make sure you are in the right place.

The list below provides some general guidelines as to who this course
was designed for.

You need to know:

-   Your way around basic Python and NumPy.

-   The basics of Keras for deep learning.

You do NOT need to know:

-   How to be a math wiz!

-   How to be a deep learning expert!

This crash course will take you from a developer that knows a little
deep learning to a developer who can get better performance on your deep
learning project.

Note: This crash course assumes you have a working Python 2 or 3 SciPy
environment with at least NumPy and Keras 2 installed. If you need help
with your environment, you can follow the step-by-step tutorial here:

-   [How to Set Up a Python Environment for Machine Learning and Deep
    Learning With
    Anaconda](https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/)

### **Want Better Results with Deep Learning?** {.western style="margin-top: 0in; margin-bottom: 0.1in; border: none; padding: 0in; line-height: 150%; page-break-inside: auto; page-break-after: auto"}

Take my free 7-day email crash course now (with sample code).

Click to sign-up and also get a free PDF Ebook version of the course.

[**Download Your FREE
Mini-Course**](https://machinelearningmastery.lpages.co/leadbox/1433e7773f72a2%3A164f8be4f346dc/5764144745676800/)

**Crash-Course Overview** {.western style="margin-top: 0in; margin-bottom: 0.13in; border: none; padding: 0in; line-height: 150%; page-break-inside: auto; background: #ffffff; page-break-after: auto"}
-------------------------

This crash course is broken down into seven lessons.

You could complete one lesson per day (recommended) or complete all of
the lessons in one day (hardcore). It really depends on the time you
have available and your level of enthusiasm.

Below are seven lessons that will allow you to confidently improve the
performance of your deep learning model:

-   **Lesson 01**: Better Deep Learning Framework

-   **Lesson 02**: Batch Size

-   **Lesson 03**: Learning Rate Schedule

-   **Lesson 04**: Batch Normalization

-   **Lesson 05**: Weight Regularization

-   **Lesson 06**: Adding Noise

-   **Lesson 07**: Early Stopping

Each lesson could take you 60 seconds or up to 30 minutes. Take your
time and complete the lessons at your own pace. Ask questions and even
post results in the comments below.

The lessons expect you to go off and find out how to do things. I will
give you hints, but part of the point of each lesson is to force you to
learn where to go to look for help (hint, I have all of the answers
directly on this blog; use the search box).

I do provide more help in the form of links to related posts because I
want you to build up some confidence and inertia.

Post your results in the comments; I’ll cheer you on!

Hang in there; don’t give up.

**Note**: This is just a crash course. For a lot more detail and fleshed
out tutorials, see my book on the topic titled “[Better Deep
Learning](https://machinelearningmastery.com/better-deep-learning/).”

**Lesson 01: Better Deep Learning Framework** {.western style="margin-top: 0in; margin-bottom: 0.13in; border: none; padding: 0in; line-height: 150%; page-break-inside: auto; background: #ffffff; page-break-after: auto"}
---------------------------------------------

In this lesson, you will discover a framework that you can use to
systematically improve the performance of your deep learning model.

Modern deep learning libraries such as Keras allow you to define and
start fitting a wide range of neural network models in minutes with just
a few lines of code.

Nevertheless, it is still challenging to configure a neural network to
get good performance on a new predictive modeling problem.

There are three types of problems that are straightforward to diagnose
with regard to the poor performance of a deep learning neural network
model; they are:

-   **Problems with Learning**. Problems with learning manifest in a
    model that cannot effectively learn a training dataset or shows slow
    progress or bad performance when learning the training dataset.

-   **Problems with Generalization**. Problems with generalization
    manifest in a model that overfits the training dataset and makes
    poor performance on a holdout dataset.

-   **Problems with Predictions**. Problems with predictions manifest as
    the stochastic training algorithm having a strong influence on the
    final model, causing a high variance in behavior and performance.

The sequential relationship between the three areas in the proposed
breakdown allows the issue of deep learning model performance to be
first isolated, then targeted with a specific technique or methodology.

We can summarize techniques that assist with each of these problems as
follows:

-   **Better Learning**. Techniques that improve or accelerate the
    adaptation of neural network model weights in response to a training
    dataset.

-   **Better Generalization**. Techniques that improve the performance
    of a neural network model on a holdout dataset.

-   **Better Predictions**. Techniques that reduce the variance in the
    performance of a final model.

You can use this framework to first diagnose the type of problem that
you have and then identify a technique to evaluate to attempt to address
your problem.

### **Your Task** {.western style="margin-top: 0in; margin-bottom: 0.1in; border: none; padding: 0in; line-height: 150%; page-break-inside: auto; background: #ffffff; page-break-after: auto"}

For this lesson, you must list two techniques or areas of focus that
belong to each of the three areas of the framework.

Having trouble? Note that we will be looking some examples from two of
the three areas as part of this mini-course.

Post your answer in the comments below. I would love to see what you
discover.

### **Next** {.western style="margin-top: 0in; margin-bottom: 0.1in; border: none; padding: 0in; line-height: 150%; page-break-inside: auto; background: #ffffff; page-break-after: auto"}

In the next lesson, you will discover how to control the speed of
learning with the batch size.

**Lesson 02: Batch Size** {.western style="margin-top: 0in; margin-bottom: 0.13in; border: none; padding: 0in; line-height: 150%; page-break-inside: auto; background: #ffffff; page-break-after: auto"}
-------------------------

In this lesson, you will discover the importance of the [batch
size](https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/)
when training neural networks.

Neural networks are trained using gradient descent where the estimate of
the error used to update the weights is calculated based on a subset of
the training dataset.

The number of examples from the training dataset used in the estimate of
the error gradient is called the batch size and is an important
hyperparameter that influences the dynamics of the learning algorithm.

The choice of batch size controls how quickly the algorithm learns, for
example:

-   **Batch Gradient Descent**. Batch size is set to the number of
    examples in the training dataset, more accurate estimate of error
    but longer time between weight updates.

-   **Stochastic Gradient Descent**. Batch size is set to 1, noisy
    estimate of error but frequent updates to weights.

-   **Minibatch Gradient Descent**. Batch size is set to a value more
    than 1 and less than the number of training examples, trade-off
    between batch and stochastic gradient descent.

Keras allows you to configure the batch size via the *batch\_size*
argument to the *fit()* function, for example:

  --- ---------------------------------------------------------------------------
  1   \# fit model
      
  2   history = model.fit(trainX, trainy, epochs=1000, batch\_size=len(trainX))
  --- ---------------------------------------------------------------------------

The example below demonstrates a Multilayer Perceptron with batch
gradient descent on a binary classification problem.

  ---- -----------------------------------------------------------------------------------------------------------------------
  1    \# example of batch gradient descent
       
  2    from sklearn.datasets import make\_circles
       
  3    from keras.layers import Dense
       
  4    from keras.models import Sequential
       
  5    from keras.optimizers import SGD
       
  6    from matplotlib import pyplot
       
  7    \# generate dataset
       
  8    X, y = make\_circles(n\_samples=1000, noise=0.1, random\_state=1)
       
  9    \# split into train and test
       
  10   n\_train = 500
       
  11   trainX, testX = X[:n\_train, :], X[n\_train:, :]
       
  12   trainy, testy = y[:n\_train], y[n\_train:]
       
  13   \# define model
       
  14   model = Sequential()
       
  15   model.add(Dense(50, input\_dim=2, activation='relu'))
       
  16   model.add(Dense(1, activation='sigmoid'))
       
  17   \# compile model
       
  18   opt = SGD(lr=0.01, momentum=0.9)
       
  19   model.compile(loss='binary\_crossentropy', optimizer=opt, metrics=['accuracy'])
       
  20   \# fit model
       
  21   history = model.fit(trainX, trainy, validation\_data=(testX, testy), epochs=1000, batch\_size=len(trainX), verbose=0)
       
  22   \# evaluate the model
       
  23   \_, train\_acc = model.evaluate(trainX, trainy, verbose=0)
       
  24   \_, test\_acc = model.evaluate(testX, testy, verbose=0)
       
  25   print('Train: %.3f, Test: %.3f' % (train\_acc, test\_acc))
       
  26   \# plot loss learning curves
       
  27   pyplot.subplot(211)
       
  28   pyplot.title('Cross-Entropy Loss', pad=-40)
       
  29   pyplot.plot(history.history['loss'], label='train')
       
  30   pyplot.plot(history.history['val\_loss'], label='test')
       
  31   pyplot.legend()
       
  32   \# plot accuracy learning curves
       
  33   pyplot.subplot(212)
       
  34   pyplot.title('Accuracy', pad=-40)
       
  35   pyplot.plot(history.history['accuracy'], label='train')
       
  36   pyplot.plot(history.history['val\_accuracy'], label='test')
       
  37   pyplot.legend()
       
  38   pyplot.show()
  ---- -----------------------------------------------------------------------------------------------------------------------

### **Your Task** {.western style="margin-top: 0in; margin-bottom: 0.1in; border: none; padding: 0in; line-height: 150%; page-break-inside: auto; background: #ffffff; page-break-after: auto"}

For this lesson, you must run the code example with each type of
gradient descent (batch, minibatch, and stochastic) and describe the
effect that it has on the [learning
curves](https://machinelearningmastery.com/how-to-control-neural-network-model-capacity-with-nodes-and-layers/)
during training.

Post your answer in the comments below. I would love to see what you
discover.

### **Next** {.western style="margin-top: 0in; margin-bottom: 0.1in; border: none; padding: 0in; line-height: 150%; page-break-inside: auto; background: #ffffff; page-break-after: auto"}

In the next lesson, you will discover how to fine tune a model during
training with a learning rate schedule

**Lesson 03: Learning Rate Schedule** {.western style="margin-top: 0in; margin-bottom: 0.13in; border: none; padding: 0in; line-height: 150%; page-break-inside: auto; background: #ffffff; page-break-after: auto"}
-------------------------------------

In this lesson, you will discover how to configure an adaptive learning
rate schedule to fine tune the model during the training run.

The amount of change to the model during each step of this search
process, or the step size, is called the “*learning rate*” and provides
perhaps the most important hyperparameter to tune for your neural
network in order to achieve good performance on your problem.

Configuring a fixed learning rate is very challenging and requires
careful experimentation. An alternative to using a fixed learning rate
is to instead vary the learning rate over the training process.

Keras provides the *ReduceLROnPlateau* learning rate schedule that will
adjust the learning rate when a plateau in model performance is
detected, e.g. no change for a given number of training epochs. For
example:

  --- ---------------------------------------------------------------------------------------------------
  1   \# define learning rate schedule
      
  2   rlrp = ReduceLROnPlateau(monitor='val\_loss', factor=0.1, patience=5, min\_delta=1E-7, verbose=1)
  --- ---------------------------------------------------------------------------------------------------

This callback is designed to reduce the learning rate after the model
stops improving with the hope of fine-tuning model weights during
training.

The example below demonstrates a Multilayer Perceptron with a learning
rate schedule on a binary classification problem, where the learning
rate will be reduced by an order of magnitude if no change is detected
in validation loss over 5 training epochs.

  ---- ---------------------------------------------------------------------------------------------------------------
  1    \# example of a learning rate schedule
       
  2    from sklearn.datasets import make\_circles
       
  3    from keras.layers import Dense
       
  4    from keras.models import Sequential
       
  5    from keras.optimizers import SGD
       
  6    from keras.callbacks import ReduceLROnPlateau
       
  7    from matplotlib import pyplot
       
  8    \# generate dataset
       
  9    X, y = make\_circles(n\_samples=1000, noise=0.1, random\_state=1)
       
  10   \# split into train and test
       
  11   n\_train = 500
       
  12   trainX, testX = X[:n\_train, :], X[n\_train:, :]
       
  13   trainy, testy = y[:n\_train], y[n\_train:]
       
  14   \# define model
       
  15   model = Sequential()
       
  16   model.add(Dense(50, input\_dim=2, activation='relu'))
       
  17   model.add(Dense(1, activation='sigmoid'))
       
  18   \# compile model
       
  19   opt = SGD(lr=0.01, momentum=0.9)
       
  20   model.compile(loss='binary\_crossentropy', optimizer=opt, metrics=['accuracy'])
       
  21   \# define learning rate schedule
       
  22   rlrp = ReduceLROnPlateau(monitor='val\_loss', factor=0.1, patience=5, min\_delta=1E-7, verbose=1)
       
  23   \# fit model
       
  24   history = model.fit(trainX, trainy, validation\_data=(testX, testy), epochs=300, verbose=0, callbacks=[rlrp])
       
  25   \# evaluate the model
       
  26   \_, train\_acc = model.evaluate(trainX, trainy, verbose=0)
       
  27   \_, test\_acc = model.evaluate(testX, testy, verbose=0)
       
  28   print('Train: %.3f, Test: %.3f' % (train\_acc, test\_acc))
       
  29   \# plot loss learning curves
       
  30   pyplot.subplot(211)
       
  31   pyplot.title('Cross-Entropy Loss', pad=-40)
       
  32   pyplot.plot(history.history['loss'], label='train')
       
  33   pyplot.plot(history.history['val\_loss'], label='test')
       
  34   pyplot.legend()
       
  35   \# plot accuracy learning curves
       
  36   pyplot.subplot(212)
       
  37   pyplot.title('Accuracy', pad=-40)
       
  38   pyplot.plot(history.history['accuracy'], label='train')
       
  39   pyplot.plot(history.history['val\_accuracy'], label='test')
       
  40   pyplot.legend()
       
  41   pyplot.show()
  ---- ---------------------------------------------------------------------------------------------------------------

### **Your Task** {.western style="margin-top: 0in; margin-bottom: 0.1in; border: none; padding: 0in; line-height: 150%; page-break-inside: auto; background: #ffffff; page-break-after: auto"}

For this lesson, you must run the code example with and without the
learning rate schedule and describe the effect that the learning rate
schedule has on the learning curves during training.

Post your answer in the comments below. I would love to see what you
discover.

### **Next** {.western style="margin-top: 0in; margin-bottom: 0.1in; border: none; padding: 0in; line-height: 150%; page-break-inside: auto; background: #ffffff; page-break-after: auto"}

In the next lesson, you will discover how you can accelerate the
training process with batch normalization

**Lesson 04: Batch Normalization** {.western style="margin-top: 0in; margin-bottom: 0.13in; border: none; padding: 0in; line-height: 150%; page-break-inside: auto; background: #ffffff; page-break-after: auto"}
----------------------------------

In this lesson, you will discover how to accelerate the training process
of your deep learning neural network using batch normalization.

Batch normalization, or batchnorm for short, is proposed as a technique
to help coordinate the update of multiple layers in the model.

The authors of the paper introducing batch normalization refer to change
in the distribution of inputs during training as “*internal covariate
shift*“. Batch normalization was designed to counter the internal
covariate shift by scaling the output of the previous layer,
specifically by standardizing the activations of each input variable per
mini-batch, such as the activations of a node from the previous layer.

Keras supports Batch Normalization via a separate *BatchNormalization*
layer that can be added between the hidden layers of your model. For
example:

  --- ---------------------------------
  1   model.add(BatchNormalization())
  --- ---------------------------------

The example below demonstrates a Multilayer Perceptron model with batch
normalization on a binary classification problem.

  ---- ---------------------------------------------------------------------------------------------
  1    \# example of batch normalization
       
  2    from sklearn.datasets import make\_circles
       
  3    from keras.models import Sequential
       
  4    from keras.layers import Dense
       
  5    from keras.optimizers import SGD
       
  6    from keras.layers import BatchNormalization
       
  7    from matplotlib import pyplot
       
  8    \# generate dataset
       
  9    X, y = make\_circles(n\_samples=1000, noise=0.1, random\_state=1)
       
  10   \# split into train and test
       
  11   n\_train = 500
       
  12   trainX, testX = X[:n\_train, :], X[n\_train:, :]
       
  13   trainy, testy = y[:n\_train], y[n\_train:]
       
  14   \# define model
       
  15   model = Sequential()
       
  16   model.add(Dense(50, input\_dim=2, activation='relu'))
       
  17   model.add(BatchNormalization())
       
  18   model.add(Dense(1, activation='sigmoid'))
       
  19   \# compile model
       
  20   opt = SGD(lr=0.01, momentum=0.9)
       
  21   model.compile(loss='binary\_crossentropy', optimizer=opt, metrics=['accuracy'])
       
  22   \# fit model
       
  23   history = model.fit(trainX, trainy, validation\_data=(testX, testy), epochs=300, verbose=0)
       
  24   \# evaluate the model
       
  25   \_, train\_acc = model.evaluate(trainX, trainy, verbose=0)
       
  26   \_, test\_acc = model.evaluate(testX, testy, verbose=0)
       
  27   print('Train: %.3f, Test: %.3f' % (train\_acc, test\_acc))
       
  28   \# plot loss learning curves
       
  29   pyplot.subplot(211)
       
  30   pyplot.title('Cross-Entropy Loss', pad=-40)
       
  31   pyplot.plot(history.history['loss'], label='train')
       
  32   pyplot.plot(history.history['val\_loss'], label='test')
       
  33   pyplot.legend()
       
  34   \# plot accuracy learning curves
       
  35   pyplot.subplot(212)
       
  36   pyplot.title('Accuracy', pad=-40)
       
  37   pyplot.plot(history.history['accuracy'], label='train')
       
  38   pyplot.plot(history.history['val\_accuracy'], label='test')
       
  39   pyplot.legend()
       
  40   pyplot.show()
  ---- ---------------------------------------------------------------------------------------------

### **Your Task** {.western style="margin-top: 0in; margin-bottom: 0.1in; border: none; padding: 0in; line-height: 150%; page-break-inside: auto; background: #ffffff; page-break-after: auto"}

For this lesson, you must run the code example with and without batch
normalization and describe the effect that batch normalization has on
the learning curves during training.

Post your answer in the comments below. I would love to see what you
discover.

### **Next** {.western style="margin-top: 0in; margin-bottom: 0.1in; border: none; padding: 0in; line-height: 150%; page-break-inside: auto; background: #ffffff; page-break-after: auto"}

In the next lesson, you will discover how to reduce overfitting using
weight regularization.

**Lesson 05: Weight Regularization** {.western style="margin-top: 0in; margin-bottom: 0.13in; border: none; padding: 0in; line-height: 150%; page-break-inside: auto; background: #ffffff; page-break-after: auto"}
------------------------------------

In this lesson, you will discover how to reduce overfitting of your deep
learning neural network using weight regularization.

A model with large weights is more complex than a model with smaller
weights. It is a sign of a network that may be overly specialized to
training data.

The learning algorithm can be updated to encourage the network toward
using small weights.

One way to do this is to change the calculation of loss used in the
optimization of the network to also consider the size of the weights.
This is called weight regularization or weight decay.

Keras supports weight regularization via the *kernel\_regularizer*
argument on a layer, which can be configured to use the [L1 or L2 vector
norm](https://machinelearningmastery.com/vector-norms-machine-learning/),
for example:

  --- --------------------------------------------------------------------------------------
  1   model.add(Dense(500, input\_dim=2, activation='relu', kernel\_regularizer=l2(0.01)))
  --- --------------------------------------------------------------------------------------

The example below demonstrates a Multilayer Perceptron model with weight
decay on a binary classification problem.

  ---- ----------------------------------------------------------------------------------------------
  1    \# example of weight decay
       
  2    from sklearn.datasets import make\_circles
       
  3    from keras.models import Sequential
       
  4    from keras.layers import Dense
       
  5    from keras.regularizers import l2
       
  6    from matplotlib import pyplot
       
  7    \# generate dataset
       
  8    X, y = make\_circles(n\_samples=100, noise=0.1, random\_state=1)
       
  9    \# split into train and test
       
  10   n\_train = 30
       
  11   trainX, testX = X[:n\_train, :], X[n\_train:, :]
       
  12   trainy, testy = y[:n\_train], y[n\_train:]
       
  13   \# define model
       
  14   model = Sequential()
       
  15   model.add(Dense(500, input\_dim=2, activation='relu', kernel\_regularizer=l2(0.01)))
       
  16   model.add(Dense(1, activation='sigmoid'))
       
  17   \# compile model
       
  18   model.compile(loss='binary\_crossentropy', optimizer='adam', metrics=['accuracy'])
       
  19   \# fit model
       
  20   history = model.fit(trainX, trainy, validation\_data=(testX, testy), epochs=4000, verbose=0)
       
  21   \# evaluate the model
       
  22   \_, train\_acc = model.evaluate(trainX, trainy, verbose=0)
       
  23   \_, test\_acc = model.evaluate(testX, testy, verbose=0)
       
  24   print('Train: %.3f, Test: %.3f' % (train\_acc, test\_acc))
       
  25   \# plot loss learning curves
       
  26   pyplot.subplot(211)
       
  27   pyplot.title('Cross-Entropy Loss', pad=-40)
       
  28   pyplot.plot(history.history['loss'], label='train')
       
  29   pyplot.plot(history.history['val\_loss'], label='test')
       
  30   pyplot.legend()
       
  31   \# plot accuracy learning curves
       
  32   pyplot.subplot(212)
       
  33   pyplot.title('Accuracy', pad=-40)
       
  34   pyplot.plot(history.history['accuracy'], label='train')
       
  35   pyplot.plot(history.history['val\_accuracy'], label='test')
       
  36   pyplot.legend()
       
  37   pyplot.show()
  ---- ----------------------------------------------------------------------------------------------

### **Your Task** {.western style="margin-top: 0in; margin-bottom: 0.1in; border: none; padding: 0in; line-height: 150%; page-break-inside: auto; background: #ffffff; page-break-after: auto"}

For this lesson, you must run the code example with and without weight
regularization and describe the effect that it has on the learning
curves during training.

Post your answer in the comments below. I would love to see what you
discover.

### **Next** {.western style="margin-top: 0in; margin-bottom: 0.1in; border: none; padding: 0in; line-height: 150%; page-break-inside: auto; background: #ffffff; page-break-after: auto"}

In the next lesson, you will discover how to reduce overfitting by
adding noise to your model

**Lesson 06: Adding Noise** {.western style="margin-top: 0in; margin-bottom: 0.13in; border: none; padding: 0in; line-height: 150%; page-break-inside: auto; background: #ffffff; page-break-after: auto"}
---------------------------

In this lesson, you will discover that adding noise to a neural network
during training can improve the robustness of the network, resulting in
better generalization and faster learning.

Training a neural network with a small dataset can cause the network to
memorize all training examples, in turn leading to poor performance on a
holdout dataset.

One approach to making the input space smoother and easier to learn is
to add noise to inputs during training.

The addition of noise during the training of a neural network model has
a regularization effect and, in turn, improves the robustness of the
model.

Noise can be added to your model in Keras via the *GaussianNoise* layer.
For example:

  --- -------------------------------
  1   model.add(GaussianNoise(0.1))
  --- -------------------------------

Noise can be added to a model at the input layer or between hidden
layers.

The example below demonstrates a Multilayer Perceptron model with added
noise between the hidden layers on a binary classification problem.

  ---- ----------------------------------------------------------------------------------------------
  1    \# example of adding noise
       
  2    from sklearn.datasets import make\_circles
       
  3    from keras.models import Sequential
       
  4    from keras.layers import Dense
       
  5    from keras.layers import GaussianNoise
       
  6    from matplotlib import pyplot
       
  7    \# generate dataset
       
  8    X, y = make\_circles(n\_samples=100, noise=0.1, random\_state=1)
       
  9    \# split into train and test
       
  10   n\_train = 30
       
  11   trainX, testX = X[:n\_train, :], X[n\_train:, :]
       
  12   trainy, testy = y[:n\_train], y[n\_train:]
       
  13   \# define model
       
  14   model = Sequential()
       
  15   model.add(Dense(500, input\_dim=2, activation='relu'))
       
  16   model.add(GaussianNoise(0.1))
       
  17   model.add(Dense(1, activation='sigmoid'))
       
  18   \# compile model
       
  19   model.compile(loss='binary\_crossentropy', optimizer='adam', metrics=['accuracy'])
       
  20   \# fit model
       
  21   history = model.fit(trainX, trainy, validation\_data=(testX, testy), epochs=4000, verbose=0)
       
  22   \# evaluate the model
       
  23   \_, train\_acc = model.evaluate(trainX, trainy, verbose=0)
       
  24   \_, test\_acc = model.evaluate(testX, testy, verbose=0)
       
  25   print('Train: %.3f, Test: %.3f' % (train\_acc, test\_acc))
       
  26   \# plot loss learning curves
       
  27   pyplot.subplot(211)
       
  28   pyplot.title('Cross-Entropy Loss', pad=-40)
       
  29   pyplot.plot(history.history['loss'], label='train')
       
  30   pyplot.plot(history.history['val\_loss'], label='test')
       
  31   pyplot.legend()
       
  32   \# plot accuracy learning curves
       
  33   pyplot.subplot(212)
       
  34   pyplot.title('Accuracy', pad=-40)
       
  35   pyplot.plot(history.history['accuracy'], label='train')
       
  36   pyplot.plot(history.history['val\_accuracy'], label='test')
       
  37   pyplot.legend()
       
  38   pyplot.show()
  ---- ----------------------------------------------------------------------------------------------

### **Your Task** {.western style="margin-top: 0in; margin-bottom: 0.1in; border: none; padding: 0in; line-height: 150%; page-break-inside: auto; background: #ffffff; page-break-after: auto"}

For this lesson, you must run the code example with and without the
addition of noise and describe the effect that it has on the learning
curves during training.

Post your answer in the comments below. I would love to see what you
discover.

### **Next** {.western style="margin-top: 0in; margin-bottom: 0.1in; border: none; padding: 0in; line-height: 150%; page-break-inside: auto; background: #ffffff; page-break-after: auto"}

In the next lesson, you will discover how to reduce overfitting using
early stopping.

**Lesson 07: Early Stopping** {.western style="margin-top: 0in; margin-bottom: 0.13in; border: none; padding: 0in; line-height: 150%; page-break-inside: auto; background: #ffffff; page-break-after: auto"}
-----------------------------

In this lesson, you will discover that stopping the training of a neural
network early before it has overfit the training dataset can reduce
overfitting and improve the generalization of deep neural networks.

A major challenge in training neural networks is how long to train them.

Too little training will mean that the model will underfit the train and
the test sets. Too much training will mean that the model will overfit
the training dataset and have poor performance on the test set.

A compromise is to train on the training dataset but to stop training at
the point when performance on a validation dataset starts to degrade.
This simple, effective, and widely used approach to training neural
networks is called early stopping.

Keras supports early stopping via the *EarlyStopping* callback that
allows you to specify the metric to monitor during training.

  --- ------------------------------------------------------------------------------
  1   \# patient early stopping
      
  2   es = EarlyStopping(monitor='val\_loss', mode='min', verbose=1, patience=200)
  --- ------------------------------------------------------------------------------

The example below demonstrates a Multilayer Perceptron with early
stopping on a binary classification problem that will stop when the
validation loss has not improved for 200 training epochs.

  ---- --------------------------------------------------------------------------------------------------------------
  1    \# example of early stopping
       
  2    from sklearn.datasets import make\_circles
       
  3    from keras.models import Sequential
       
  4    from keras.layers import Dense
       
  5    from keras.callbacks import EarlyStopping
       
  6    from matplotlib import pyplot
       
  7    \# generate dataset
       
  8    X, y = make\_circles(n\_samples=100, noise=0.1, random\_state=1)
       
  9    \# split into train and test
       
  10   n\_train = 30
       
  11   trainX, testX = X[:n\_train, :], X[n\_train:, :]
       
  12   trainy, testy = y[:n\_train], y[n\_train:]
       
  13   \# define model
       
  14   model = Sequential()
       
  15   model.add(Dense(500, input\_dim=2, activation='relu'))
       
  16   model.add(Dense(1, activation='sigmoid'))
       
  17   \# compile model
       
  18   model.compile(loss='binary\_crossentropy', optimizer='adam', metrics=['accuracy'])
       
  19   \# patient early stopping
       
  20   es = EarlyStopping(monitor='val\_loss', mode='min', verbose=1, patience=200)
       
  21   \# fit model
       
  22   history = model.fit(trainX, trainy, validation\_data=(testX, testy), epochs=4000, verbose=0, callbacks=[es])
       
  23   \# evaluate the model
       
  24   \_, train\_acc = model.evaluate(trainX, trainy, verbose=0)
       
  25   \_, test\_acc = model.evaluate(testX, testy, verbose=0)
       
  26   print('Train: %.3f, Test: %.3f' % (train\_acc, test\_acc))
       
  27   \# plot loss learning curves
       
  28   pyplot.subplot(211)
       
  29   pyplot.title('Cross-Entropy Loss', pad=-40)
       
  30   pyplot.plot(history.history['loss'], label='train')
       
  31   pyplot.plot(history.history['val\_loss'], label='test')
       
  32   pyplot.legend()
       
  33   \# plot accuracy learning curves
       
  34   pyplot.subplot(212)
       
  35   pyplot.title('Accuracy', pad=-40)
       
  36   pyplot.plot(history.history['accuracy'], label='train')
       
  37   pyplot.plot(history.history['val\_accuracy'], label='test')
       
  38   pyplot.legend()
       
  39   pyplot.show()
  ---- --------------------------------------------------------------------------------------------------------------

### **Your Task** {.western style="margin-top: 0in; margin-bottom: 0.1in; border: none; padding: 0in; line-height: 150%; page-break-inside: auto; background: #ffffff; page-break-after: auto"}

For this lesson, you must run the code example with and without early
stopping and describe the effect it has on the learning curves during
training.

Post your answer in the comments below. I would love to see what you
discover.

### **Next** {.western style="margin-top: 0in; margin-bottom: 0.1in; border: none; padding: 0in; line-height: 150%; page-break-inside: auto; background: #ffffff; page-break-after: auto"}

This was your final lesson.

**The End!** {.western style="margin-top: 0in; margin-bottom: 0.13in; border: none; padding: 0in; line-height: 150%; page-break-inside: auto; background: #ffffff; page-break-after: auto"}
------------

**(*****Look how far you have come!*****)** {.western style="margin-top: 0in; margin-bottom: 0.13in; border: none; padding: 0in; line-height: 150%; page-break-inside: auto; background: #ffffff; page-break-after: auto"}
-------------------------------------------

You made it. Well done!

Take a moment and look back at how far you have come.

You discovered:

-   A framework that you can use to systematically diagnose and improve
    the performance of your deep learning model.

-   Batch size can be used to control the precision of the estimated
    error and the speed of learning during training.

-   Learning rate schedule can be used to fine tune the model weights
    during training.

-   Batch normalization can be used to dramatically accelerate the
    training process of neural network models.

-   Weight regularization will penalize models based on the size of the
    weights and reduce overfitting.

-   Adding noise will make the model more robust to differences in input
    and reduce overfitting

-   Early stopping will halt the training process at the right time and
    reduce overfitting.

This is just the beginning of your journey with deep learning
performance improvement. Keep practicing and developing your skills.

Take the next step and check out [my book on getting better performance
with deep
learning](https://machinelearningmastery.com/better-deep-learning/).

**Summary** {.western style="margin-top: 0in; margin-bottom: 0.13in; border: none; padding: 0in; line-height: 150%; page-break-inside: auto; background: #ffffff; page-break-after: auto"}
-----------

How did you do with the mini-course?

Did you enjoy this crash course?

\

