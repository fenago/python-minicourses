### **Deep Learning for Time Series Forecasting Crash Course.** {.western align="center" style="margin-top: 0in; margin-bottom: 0.1in; border: none; padding: 0in; line-height: 150%; page-break-inside: auto; background: #ffffff; page-break-after: auto"}

#### **Bring Deep Learning methods to Your Time Series project in 7 Days.** {align="center" style="margin-top: 0in; margin-bottom: 0.08in; border: none; padding: 0in; line-height: 150%; page-break-inside: auto; background: #ffffff; page-break-after: auto"}

[Time series
forecasting](https://machinelearningmastery.com/time-series-forecasting/)
is challenging, especially when working with long sequences, noisy data,
multi-step forecasts and multiple input and output variables.

Deep learning methods offer a lot of promise for time series
forecasting, such as the automatic learning of temporal dependence and
the automatic handling of temporal structures like trends and
seasonality.

In this crash course, you will discover how you can get started and
confidently develop deep learning models for time series forecasting
problems using Python in 7 days.

This is a big and important post. You might want to bookmark it.

Discover how to build models for multivariate and multi-step time series
forecasting with LSTMs and more [in my new
book](https://machinelearningmastery.com/deep-learning-for-time-series-forecasting/),
with 25 step-by-step tutorials and full source code.

Let’s get started.

![](../velit_ullam_html_aeb8d25c741d5e6b.jpg)

How to Get Started with Deep Learning for Time Series Forecasting (7-Day
Mini-Course)

Photo by [Brian
Richardson](https://www.flickr.com/photos/seriousbri/3736154699/), some
rights reserved.

**Who Is This Crash-Course For?** {.western style="margin-top: 0in; margin-bottom: 0.13in; border: none; padding: 0in; line-height: 150%; page-break-inside: auto; background: #ffffff; page-break-after: auto"}
---------------------------------

Before we get started, let’s make sure you are in the right place.

The list below provides some general guidelines as to who this course
was designed for.

You need to know:

-   You need to know the basics of time series forecasting.

-   You need to know your way around basic Python, NumPy and Keras for
    deep learning.

You do NOT need to know:

-   You do not need to be a math wiz!

-   You do not need to be a deep learning expert!

-   You do not need to be a time series expert!

This crash course will take you from a developer that knows a little
machine learning to a developer who can bring deep learning methods to
your own time series forecasting project.

**Note**: This crash course assumes you have a working Python 2 or 3
SciPy environment with at least NumPy and Keras 2 installed. If you need
help with your environment, you can follow the step-by-step tutorial
here:

-   [How to Setup a Python Environment for Machine Learning and Deep
    Learning with
    Anaconda](https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/)

**Crash-Course Overview** {.western style="margin-top: 0in; margin-bottom: 0.13in; border: none; padding: 0in; line-height: 150%; page-break-inside: auto; background: #ffffff; page-break-after: auto"}
-------------------------

This crash course is broken down into 7 lessons.

You could complete one lesson per day (recommended) or complete all of
the lessons in one day (hardcore). It really depends on the time you
have available and your level of enthusiasm.

Below are 7 lessons that will get you started and productive with deep
learning for time series forecasting in Python:

-   **Lesson 01**: Promise of Deep Learning

-   **Lesson 02**: How to Transform Data for Time Series

-   **Lesson 03**: MLP for Time Series Forecasting

-   **Lesson 04**: CNN for Time Series Forecasting

-   **Lesson 05**: LSTM for Time Series Forecasting

-   **Lesson 06:** CNN-LSTM for Time Series Forecasting

-   **Lesson 07**: Encoder-Decoder LSTM Multi-step Forecasting

Each lesson could take you 60 seconds or up to 30 minutes. Take your
time and complete the lessons at your own pace. Ask questions and even
post results in the comments below.

The lessons expect you to go off and find out how to do things. I will
give you hints, but part of the point of each lesson is to force you to
learn where to go to look for help on and about the deep learning, time
series forecasting and the best-of-breed tools in Python (hint, *I have
all of the answers directly on this blog, use the search box*).

I do provide more help in the form of links to related posts because I
want you to build up some confidence and inertia.

Post your results in the comments, I’ll cheer you on!

Hang in there, don’t give up.

**Note**: This is just a crash course. For a lot more detail and 25
fleshed out tutorials, see my book on the topic titled “[Deep Learning
for Time Series
Forecasting](https://machinelearningmastery.com/deep-learning-for-time-series-forecasting/)“.

### **Need help with Deep Learning for Time Series?** {.western style="margin-top: 0in; margin-bottom: 0.1in; border: none; padding: 0in; line-height: 150%; page-break-inside: auto; page-break-after: auto"}

Take my free 7-day email crash course now (with sample code).

Click to sign-up and also get a free PDF Ebook version of the course.

[**Download Your FREE
Mini-Course**](https://machinelearningmastery.lpages.co/leadbox/14531ee73f72a2%3A164f8be4f346dc/5630742793027584/)

**Lesson 01: Promise of Deep Learning** {.western style="margin-top: 0in; margin-bottom: 0.13in; border: none; padding: 0in; line-height: 150%; page-break-inside: auto; background: #ffffff; page-break-after: auto"}
---------------------------------------

In this lesson, you will discover the promise of deep learning methods
for time series forecasting.

Generally, neural networks like Multilayer Perceptrons or MLPs provide
capabilities that are offered by few algorithms, such as:

-   **Robust to Noise**. Neural networks are robust to noise in input
    data and in the mapping function and can even support learning and
    prediction in the presence of missing values.

-   **Nonlinear**. Neural networks do not make strong assumptions about
    the mapping function and readily learn linear and nonlinear
    relationships.

-   **Multivariate Inputs**. An arbitrary number of input features can
    be specified, providing direct support for multivariate forecasting.

-   **Multi-step Forecasts**. An arbitrary number of output values can
    be specified, providing\
     direct support for multi-step and even multivariate forecasting.

For these capabilities alone, feedforward neural networks may be useful
for time series forecasting.

### **Your Task** {.western style="margin-top: 0in; margin-bottom: 0.1in; border: none; padding: 0in; line-height: 150%; page-break-inside: auto; background: #ffffff; page-break-after: auto"}

For this lesson you must suggest one capability from both Convolutional
Neural Networks and Recurrent Neural Networks that may be beneficial in
modeling time series forecasting problems.

Post your answer in the comments below. I would love to see what you
discover.

### **More Information** {.western style="margin-top: 0in; margin-bottom: 0.1in; border: none; padding: 0in; line-height: 150%; page-break-inside: auto; background: #ffffff; page-break-after: auto"}

-   [The Promise of Recurrent Neural Networks for Time Series
    Forecasting](https://machinelearningmastery.com/promise-recurrent-neural-networks-time-series-forecasting/)

In the next lesson, you will discover how to transform time series data
for time series forecasting.

**Lesson 02: How to Transform Data for Time Series** {.western style="margin-top: 0in; margin-bottom: 0.13in; border: none; padding: 0in; line-height: 150%; page-break-inside: auto; background: #ffffff; page-break-after: auto"}
----------------------------------------------------

In this lesson, you will discover how to transform your time series data
into a supervised learning format.

The majority of practical machine learning uses supervised learning.

Supervised learning is where you have input variables (X) and an output
variable (y) and you use an algorithm to learn the mapping function from
the input to the output. The goal is to approximate the real underlying
mapping so well that when you have new input data, you can predict the
output variables for that data.

Time series data can be phrased as supervised learning.

Given a sequence of numbers for a time series dataset, we can
restructure the data to look like a supervised learning problem. We can
do this by using previous time steps as input variables and use the next
time step as the output variable.

For example, the series:

  --- --------------------
  1   1, 2, 3, 4, 5, ...
  --- --------------------

Can be transformed into samples with input and output components that
can be used as part of a training set to train a supervised learning
model like a deep learning neural network.

  --- -------------
  1   X, y
      
  2   [1, 2, 3] 4
      
  3   [2, 3, 4] 5
      
  4   ...
  --- -------------

This is called a sliding window transformation as it is just like
sliding a window across prior observations that are used as inputs to
the model in order to predict the next value in the series. In this case
the window width is 3 time steps.

### **Your Task** {.western style="margin-top: 0in; margin-bottom: 0.1in; border: none; padding: 0in; line-height: 150%; page-break-inside: auto; background: #ffffff; page-break-after: auto"}

For this lesson you must develop Python code to transform the daily
female births dataset into a supervised learning format with some number
of inputs and one output.

You can download the dataset from here:
[daily-total-female-births.csv](https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-total-female-births.csv)

Post your answer in the comments below. I would love to see what you
discover.

### **More Information** {.western style="margin-top: 0in; margin-bottom: 0.1in; border: none; padding: 0in; line-height: 150%; page-break-inside: auto; background: #ffffff; page-break-after: auto"}

-   [Time Series Forecasting as Supervised
    Learning](https://machinelearningmastery.com/time-series-forecasting-supervised-learning/)

-   [How to Convert a Time Series to a Supervised Learning Problem in
    Python](https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/)

-   [How to Prepare Univariate Time Series Data for Long Short-Term
    Memory
    Networks](https://machinelearningmastery.com/prepare-univariate-time-series-data-long-short-term-memory-networks/)

In the next lesson, you will discover how to develop a Multilayer
Perceptron deep learning model for forecasting a univariate time series.

**Lesson 03: MLP for Time Series Forecasting** {.western style="margin-top: 0in; margin-bottom: 0.13in; border: none; padding: 0in; line-height: 150%; page-break-inside: auto; background: #ffffff; page-break-after: auto"}
----------------------------------------------

In this lesson, you will discover how to develop a Multilayer Perceptron
model or MLP for univariate time series forecasting.

We can define a simple univariate problem as a sequence of integers, fit
the model on this sequence and have the model predict the next value in
the sequence. We will frame the problem to have 3 inputs and 1 output,
for example: [10, 20, 30] as input and [40] as output.

First, we can define the model. We will define the number of input time
steps as 3 via the *input\_dim* argument on the first hidden layer. In
this case we will use the efficient Adam version of stochastic gradient
descent and optimizes the mean squared error (‘*mse*‘) loss function.

Once the model is defined, it can be fit on the training data and the
fit model can be used to make a prediction.

The complete example is listed below.

  ---- ---------------------------------------------------------------------
  1    \# univariate mlp example
       
  2    from numpy import array
       
  3    from keras.models import Sequential
       
  4    from keras.layers import Dense
       
  5    \# define dataset
       
  6    X = array([[10, 20, 30], [20, 30, 40], [30, 40, 50], [40, 50, 60]])
       
  7    y = array([40, 50, 60, 70])
       
  8    \# define model
       
  9    model = Sequential()
       
  10   model.add(Dense(100, activation='relu', input\_dim=3))
       
  11   model.add(Dense(1))
       
  12   model.compile(optimizer='adam', loss='mse')
       
  13   \# fit model
       
  14   model.fit(X, y, epochs=2000, verbose=0)
       
  15   \# demonstrate prediction
       
  16   x\_input = array([50, 60, 70])
       
  17   x\_input = x\_input.reshape((1, 3))
       
  18   yhat = model.predict(x\_input, verbose=0)
       
  19   print(yhat)
  ---- ---------------------------------------------------------------------

Running the example will fit the model on the data then predict the next
out-of-sample value.

Given [50, 60, 70] as input, the model correctly predicts 80 as the next
value in the sequence.

### **Your Task** {.western style="margin-top: 0in; margin-bottom: 0.1in; border: none; padding: 0in; line-height: 150%; page-break-inside: auto; background: #ffffff; page-break-after: auto"}

For this lesson you must download the daily female births dataset, split
it into train and test sets and develop a model that can make reasonably
accurate predictions on the test set.

You can download the dataset from here:
[daily-total-female-births.csv](https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-total-female-births.csv)

Post your answer in the comments below. I would love to see what you
discover.

### **More Information** {.western style="margin-top: 0in; margin-bottom: 0.1in; border: none; padding: 0in; line-height: 150%; page-break-inside: auto; background: #ffffff; page-break-after: auto"}

-   [Crash Course On Multi-Layer Perceptron Neural
    Networks](https://machinelearningmastery.com/neural-networks-crash-course/)

-   [Time Series Prediction With Deep Learning in
    Keras](https://machinelearningmastery.com/time-series-prediction-with-deep-learning-in-python-with-keras/)

-   [Exploratory Configuration of a Multilayer Perceptron Network for
    Time Series
    Forecasting](https://machinelearningmastery.com/exploratory-configuration-multilayer-perceptron-network-time-series-forecasting/)

In the next lesson, you will discover how to develop a Convolutional
Neural Network model for forecasting a univariate time series.

**Lesson 04: CNN for Time Series Forecasting** {.western style="margin-top: 0in; margin-bottom: 0.13in; border: none; padding: 0in; line-height: 150%; page-break-inside: auto; background: #ffffff; page-break-after: auto"}
----------------------------------------------

In this lesson, you will discover how to develop a Convolutional Neural
Network model or CNN for univariate time series forecasting.

We can define a simple univariate problem as a sequence of integers, fit
the model on this sequence and have the model predict the next value in
the sequence. We will frame the problem to have 3 inputs and 1 output,
for example: [10, 20, 30] as input and [40] as output.

An important difference from the MLP model is that the CNN model expects
three-dimensional input with the shape [*samples, timesteps, features*].
We will define the data in the form [*samples, timesteps*] and reshape
it accordingly.

We will define the number of input time steps as 3 and the number of
features as 1 via the *input\_shape* argument on the first hidden layer.

We will use one convolutional hidden layer followed by a max pooling
layer. The filter maps are then flattened before being interpreted by a
Dense layer and outputting a prediction. The model uses the efficient
Adam version of stochastic gradient descent and optimizes the mean
squared error (‘*mse*‘) loss function.

Once the model is defined, it can be fit on the training data and the
fit model can be used to make a prediction.

The complete example is listed below.

  ---- ---------------------------------------------------------------------------------------
  1    \# univariate cnn example
       
  2    from numpy import array
       
  3    from keras.models import Sequential
       
  4    from keras.layers import Dense
       
  5    from keras.layers import Flatten
       
  6    from keras.layers.convolutional import Conv1D
       
  7    from keras.layers.convolutional import MaxPooling1D
       
  8    \# define dataset
       
  9    X = array([[10, 20, 30], [20, 30, 40], [30, 40, 50], [40, 50, 60]])
       
  10   y = array([40, 50, 60, 70])
       
  11   \# reshape from [samples, timesteps] into [samples, timesteps, features]
       
  12   X = X.reshape((X.shape[0], X.shape[1], 1))
       
  13   \# define model
       
  14   model = Sequential()
       
  15   model.add(Conv1D(filters=64, kernel\_size=2, activation='relu', input\_shape=(3, 1)))
       
  16   model.add(MaxPooling1D(pool\_size=2))
       
  17   model.add(Flatten())
       
  18   model.add(Dense(50, activation='relu'))
       
  19   model.add(Dense(1))
       
  20   model.compile(optimizer='adam', loss='mse')
       
  21   \# fit model
       
  22   model.fit(X, y, epochs=1000, verbose=0)
       
  23   \# demonstrate prediction
       
  24   x\_input = array([50, 60, 70])
       
  25   x\_input = x\_input.reshape((1, 3, 1))
       
  26   yhat = model.predict(x\_input, verbose=0)
       
  27   print(yhat)
  ---- ---------------------------------------------------------------------------------------

Running the example will fit the model on the data then predict the next
out-of-sample value.

Given [50, 60, 70] as input, the model correctly predicts 80 as the next
value in the sequence.

### **Your Task** {.western style="margin-top: 0in; margin-bottom: 0.1in; border: none; padding: 0in; line-height: 150%; page-break-inside: auto; background: #ffffff; page-break-after: auto"}

For this lesson you must download the daily female births dataset, split
it into train and test sets and develop a model that can make reasonably
accurate predictions on the test set.

You can download the dataset from here:
[daily-total-female-births.csv](https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-total-female-births.csv)

Post your answer in the comments below. I would love to see what you
discover.

### **More Information** {.western style="margin-top: 0in; margin-bottom: 0.1in; border: none; padding: 0in; line-height: 150%; page-break-inside: auto; background: #ffffff; page-break-after: auto"}

-   [Crash Course in Convolutional Neural Networks for Machine
    Learning](https://machinelearningmastery.com/crash-course-convolutional-neural-networks/)

In the next lesson, you will discover how to develop a Long Short-Term
Memory network model for forecasting a univariate time series.

**Lesson 05: LSTM for Time Series Forecasting** {.western style="margin-top: 0in; margin-bottom: 0.13in; border: none; padding: 0in; line-height: 150%; page-break-inside: auto; background: #ffffff; page-break-after: auto"}
-----------------------------------------------

In this lesson, you will discover how to develop a Long Short-Term
Memory Neural Network model or LSTM for univariate time series
forecasting.

We can define a simple univariate problem as a sequence of integers, fit
the model on this sequence and have the model predict the next value in
the sequence. We will frame the problem to have 3 inputs and 1 output,
for example: [10, 20, 30] as input and [40] as output.

An important difference from the MLP model, and like the CNN model, is
that the LSTM model expects three-dimensional input with the shape
[*samples, timesteps, features*]. We will define the data in the form
[*samples, timesteps*] and reshape it accordingly.

We will define the number of input time steps as 3 and the number of
features as 1 via the *input\_shape* argument on the first hidden layer.

We will use one LSTM layer to process each input sub-sequence of 3 time
steps, followed by a Dense layer to interpret the summary of the input
sequence. The model uses the efficient Adam version of stochastic
gradient descent and optimizes the mean squared error (‘*mse*‘) loss
function.

Once the model is defined, it can be fit on the training data and the
fit model can be used to make a prediction.

The complete example is listed below.

  ---- --------------------------------------------------------------------------
  1    \# univariate lstm example
       
  2    from numpy import array
       
  3    from keras.models import Sequential
       
  4    from keras.layers import LSTM
       
  5    from keras.layers import Dense
       
  6    \# define dataset
       
  7    X = array([[10, 20, 30], [20, 30, 40], [30, 40, 50], [40, 50, 60]])
       
  8    y = array([40, 50, 60, 70])
       
  9    \# reshape from [samples, timesteps] into [samples, timesteps, features]
       
  10   X = X.reshape((X.shape[0], X.shape[1], 1))
       
  11   \# define model
       
  12   model = Sequential()
       
  13   model.add(LSTM(50, activation='relu', input\_shape=(3, 1)))
       
  14   model.add(Dense(1))
       
  15   model.compile(optimizer='adam', loss='mse')
       
  16   \# fit model
       
  17   model.fit(X, y, epochs=1000, verbose=0)
       
  18   \# demonstrate prediction
       
  19   x\_input = array([50, 60, 70])
       
  20   x\_input = x\_input.reshape((1, 3, 1))
       
  21   yhat = model.predict(x\_input, verbose=0)
       
  22   print(yhat)
  ---- --------------------------------------------------------------------------

Running the example will fit the model on the data then predict the next
out-of-sample value.

Given [50, 60, 70] as input, the model correctly predicts 80 as the next
value in the sequence.

### **Your Task** {.western style="margin-top: 0in; margin-bottom: 0.1in; border: none; padding: 0in; line-height: 150%; page-break-inside: auto; background: #ffffff; page-break-after: auto"}

For this lesson you must download the daily female births dataset, split
it into train and test sets and develop a model that can make reasonably
accurate predictions on the test set.

You can download the dataset from here:
[daily-total-female-births.csv](https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-total-female-births.csv)

Post your answer in the comments below. I would love to see what you
discover.

### **More Information** {.western style="margin-top: 0in; margin-bottom: 0.1in; border: none; padding: 0in; line-height: 150%; page-break-inside: auto; background: #ffffff; page-break-after: auto"}

-   [A Gentle Introduction to Long Short-Term Memory Networks by the
    Experts](https://machinelearningmastery.com/gentle-introduction-long-short-term-memory-networks-experts/)

-   [Crash Course in Recurrent Neural Networks for Deep
    Learning](https://machinelearningmastery.com/crash-course-recurrent-neural-networks-deep-learning/)

In the next lesson, you will discover how to develop a hybrid CNN-LSTM
model for a univariate time series forecasting problem.

**Lesson 06: CNN-LSTM for Time Series Forecasting** {.western style="margin-top: 0in; margin-bottom: 0.13in; border: none; padding: 0in; line-height: 150%; page-break-inside: auto; background: #ffffff; page-break-after: auto"}
---------------------------------------------------

In this lesson, you will discover how to develop a hybrid CNN-LSTM model
for univariate time series forecasting.

The benefit of this model is that the model can support very long input
sequences that can be read as blocks or subsequences by the CNN model,
then pieced together by the LSTM model.

We can define a simple univariate problem as a sequence of integers, fit
the model on this sequence and have the model predict the next value in
the sequence. We will frame the problem to have 4 inputs and 1 output,
for example: [10, 20, 30, 40] as input and [50] as output.

When using a hybrid CNN-LSTM model, we will further divide each sample
into further subsequences. The CNN model will interpret each
sub-sequence and the LSTM will piece together the interpretations from
the subsequences. As such, we will split each sample into 2 subsequences
of 2 times per subsequence.

The CNN will be defined to expect 2 time steps per subsequence with one
feature. The entire CNN model is then wrapped in TimeDistributed wrapper
layers so that it can be applied to each subsequence in the sample. The
results are then interpreted by the LSTM layer before the model outputs
a prediction.

The model uses the efficient Adam version of stochastic gradient descent
and optimizes the mean squared error (‘mse’) loss function.

Once the model is defined, it can be fit on the training data and the
fit model can be used to make a prediction.

The complete example is listed below.

  ---- --------------------------------------------------------------------------------------------------------------
  1    \# univariate cnn-lstm example
       
  2    from numpy import array
       
  3    from keras.models import Sequential
       
  4    from keras.layers import LSTM
       
  5    from keras.layers import Dense
       
  6    from keras.layers import Flatten
       
  7    from keras.layers import TimeDistributed
       
  8    from keras.layers.convolutional import Conv1D
       
  9    from keras.layers.convolutional import MaxPooling1D
       
  10   \# define dataset
       
  11   X = array([[10, 20, 30, 40], [20, 30, 40, 50], [30, 40, 50, 60], [40, 50, 60, 70]])
       
  12   y = array([50, 60, 70, 80])
       
  13   \# reshape from [samples, timesteps] into [samples, subsequences, timesteps, features]
       
  14   X = X.reshape((X.shape[0], 2, 2, 1))
       
  15   \# define model
       
  16   model = Sequential()
       
  17   model.add(TimeDistributed(Conv1D(filters=64, kernel\_size=1, activation='relu'), input\_shape=(None, 2, 1)))
       
  18   model.add(TimeDistributed(MaxPooling1D(pool\_size=2)))
       
  19   model.add(TimeDistributed(Flatten()))
       
  20   model.add(LSTM(50, activation='relu'))
       
  21   model.add(Dense(1))
       
  22   model.compile(optimizer='adam', loss='mse')
       
  23   \# fit model
       
  24   model.fit(X, y, epochs=500, verbose=0)
       
  25   \# demonstrate prediction
       
  26   x\_input = array([50, 60, 70, 80])
       
  27   x\_input = x\_input.reshape((1, 2, 2, 1))
       
  28   yhat = model.predict(x\_input, verbose=0)
       
  29   print(yhat)
  ---- --------------------------------------------------------------------------------------------------------------

Running the example will fit the model on the data then predict the next
out-of-sample value.

Given [50, 60, 70, 80] as input, the model correctly predicts 90 as the
next value in the sequence.

### **Your Task** {.western style="margin-top: 0in; margin-bottom: 0.1in; border: none; padding: 0in; line-height: 150%; page-break-inside: auto; background: #ffffff; page-break-after: auto"}

For this lesson you must download the daily female births dataset, split
it into train and test sets and develop a model that can make reasonably
accurate predictions on the test set.

You can download the dataset from here:
[daily-total-female-births.csv](https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-total-female-births.csv)

Post your answer in the comments below. I would love to see what you
discover.

### **More Information** {.western style="margin-top: 0in; margin-bottom: 0.1in; border: none; padding: 0in; line-height: 150%; page-break-inside: auto; background: #ffffff; page-break-after: auto"}

-   [CNN Long Short-Term Memory
    Networks](https://machinelearningmastery.com/cnn-long-short-term-memory-networks/)

-   [How to Use the TimeDistributed Layer for Long Short-Term Memory
    Networks in
    Python](https://machinelearningmastery.com/timedistributed-layer-for-long-short-term-memory-networks-in-python/)

In the next lesson, you will discover how to develop an Encoder-Decoder
LSTM network model for multi-step time series forecasting.

**Lesson 07: Encoder-Decoder LSTM Multi-step Forecasting** {.western style="margin-top: 0in; margin-bottom: 0.13in; border: none; padding: 0in; line-height: 150%; page-break-inside: auto; background: #ffffff; page-break-after: auto"}
----------------------------------------------------------

In this lesson, you will discover how to develop an Encoder-Decoder LSTM
Network model for multi-step time series forecasting.

We can define a simple univariate problem as a sequence of integers, fit
the model on this sequence and have the model predict the next two
values in the sequence. We will frame the problem to have 3 inputs and 2
outputs, for example: [10, 20, 30] as input and [40, 50] as output.

The LSTM model expects three-dimensional input with the shape [*samples,
timesteps, features*]. We will define the data in the form [*samples,
timesteps*] and reshape it accordingly. The output must also be shaped
this way when using the Encoder-Decoder model.

We will define the number of input time steps as 3 and the number of
features as 1 via the *input\_shape* argument on the first hidden layer.

We will define an LSTM encoder to read and encode the input sequences of
3 time steps. The encoded sequence will be repeated 2 times by the model
for the two output time steps required by the model using a RepeatVector
layer. These will be fed to a decoder LSTM layer before using a Dense
output layer wrapped in a TimeDistributed layer that will produce one
output for each step in the output sequence.

The model uses the efficient Adam version of stochastic gradient descent
and optimizes the mean squared error (‘*mse*‘) loss function.

Once the model is defined, it can be fit on the training data and the
fit model can be used to make a prediction.

The complete example is listed below.

  ---- --------------------------------------------------------------------------
  1    \# multi-step encoder-decoder lstm example
       
  2    from numpy import array
       
  3    from keras.models import Sequential
       
  4    from keras.layers import LSTM
       
  5    from keras.layers import Dense
       
  6    from keras.layers import RepeatVector
       
  7    from keras.layers import TimeDistributed
       
  8    \# define dataset
       
  9    X = array([[10, 20, 30], [20, 30, 40], [30, 40, 50], [40, 50, 60]])
       
  10   y = array([[40,50],[50,60],[60,70],[70,80]])
       
  11   \# reshape from [samples, timesteps] into [samples, timesteps, features]
       
  12   X = X.reshape((X.shape[0], X.shape[1], 1))
       
  13   y = y.reshape((y.shape[0], y.shape[1], 1))
       
  14   \# define model
       
  15   model = Sequential()
       
  16   model.add(LSTM(100, activation='relu', input\_shape=(3, 1)))
       
  17   model.add(RepeatVector(2))
       
  18   model.add(LSTM(100, activation='relu', return\_sequences=True))
       
  19   model.add(TimeDistributed(Dense(1)))
       
  20   model.compile(optimizer='adam', loss='mse')
       
  21   \# fit model
       
  22   model.fit(X, y, epochs=100, verbose=0)
       
  23   \# demonstrate prediction
       
  24   x\_input = array([50, 60, 70])
       
  25   x\_input = x\_input.reshape((1, 3, 1))
       
  26   yhat = model.predict(x\_input, verbose=0)
       
  27   print(yhat)
  ---- --------------------------------------------------------------------------

Running the example will fit the model on the data then predict the next
two out-of-sample values.

Given [50, 60, 70] as input, the model correctly predicts [80, 90] as
the next two values in the sequence.

### **Your Task** {.western style="margin-top: 0in; margin-bottom: 0.1in; border: none; padding: 0in; line-height: 150%; page-break-inside: auto; background: #ffffff; page-break-after: auto"}

For this lesson you must download the daily female births dataset, split
it into train and test sets and develop a model that can make reasonably
accurate predictions on the test set.

You can download the dataset from here:
[daily-total-female-births.csv](https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-total-female-births.csv)

Post your answer in the comments below. I would love to see what you
discover.

### **More Information** {.western style="margin-top: 0in; margin-bottom: 0.1in; border: none; padding: 0in; line-height: 150%; page-break-inside: auto; background: #ffffff; page-break-after: auto"}

-   [Encoder-Decoder Long Short-Term Memory
    Networks](https://machinelearningmastery.com/encoder-decoder-long-short-term-memory-networks/)

-   [4 Strategies for Multi-Step Time Series
    Forecasting](https://machinelearningmastery.com/multi-step-time-series-forecasting/)

-   [Multi-step Time Series Forecasting with Long Short-Term Memory
    Networks in
    Python](https://machinelearningmastery.com/multi-step-time-series-forecasting-long-short-term-memory-networks-python/)

**The End!** {.western style="margin-top: 0in; margin-bottom: 0.13in; border: none; padding: 0in; line-height: 150%; page-break-inside: auto; background: #ffffff; page-break-after: auto"}
------------

**(*****Look How Far You Have Come*****)** {.western style="margin-top: 0in; margin-bottom: 0.13in; border: none; padding: 0in; line-height: 150%; page-break-inside: auto; background: #ffffff; page-break-after: auto"}
------------------------------------------

You made it. Well done!

Take a moment and look back at how far you have come.

You discovered:

-   The promise of deep learning neural networks for time series
    forecasting problems.

-   How to transform a time series dataset into a supervised learning
    problem.

-   How to develop a Multilayer Perceptron model for a univariate time
    series forecasting problem.

-   How to develop a Convolutional Neural Network model for a univariate
    time series forecasting problem.

-   How to develop a Long Short-Term Memory network model for a
    univariate time series forecasting problem.

-   How to develop a Hybrid CNN-LSTM model for a univariate time series
    forecasting problem.

-   How to develop an Encoder-Decoder LSTM model for a multi-step time
    series forecasting problem.

This is just the beginning of your journey with deep learning for time
series forecasting. Keep practicing and developing your skills.

Take the next step and check out my book on [deep learning for time
series](https://machinelearningmastery.com/deep-learning-for-time-series-forecasting/).

\

