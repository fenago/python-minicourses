**Mini-Course on Long Short-Term Memory Recurrent Neural Networks with Keras** ust 14, 2019

Long Short-Term Memory (LSTM) recurrent neural networks are one of the
most interesting types of deep learning at the moment.
They have been used to demonstrate world-class results in complex
problem domains such as language translation, automatic image
captioning, and text generation.

LSTMs are different to multilayer Perceptrons and convolutional neural
networks in that they are designed specifically for sequence prediction
problems.
In this mini-course, you will discover how you can quickly bring LSTM
models to your own sequence forecasting problems.

After completing this mini-course, you will know:

-   What LSTMs are, how they are trained, and how to prepare data for
    training LSTM models.

-   How to develop a suite of LSTM models including stacked,
    bidirectional, and encoder-decoder models.

-   How you can get the most out of your models with hyperparameter
    optimization, updating, and finalizing models.

Discover how to develop LSTMs such as stacked, bidirectional, CNN-LSTM,
Encoder-Decoder seq2seq and more [in my new
book](https://machinelearningmastery.com/lstms-with-python/), with 14
step-by-step tutorials and full code.

Let’s get started.

**Note**: This is a big guide; you may want to bookmark it.

![](../earum_voluptatem_provident_html_3056e60733b87f25.jpg)

**Who Is This Mini-Course For?** 
--------------------------------

Before we get started, let’s make sure you are in the right place.

This course is for developers that know some applied machine learning
and need to get good at LSTMs fast.

Maybe you want or need to start using LSTMs on your project. This guide
was written to help you do that quickly and efficiently.

-   You know your way around Python.

-   You know your way around SciPy.

-   You know how to install software on your workstation.

-   You know how to wrangle your own data.

-   You know how to work through a predictive modeling problem with
    machine learning.

-   You may know a little bit of deep learning.

-   You may know a little bit of Keras.

You know how to set up your workstation to use Keras and scikit-learn;
if not, you can learn how to here:

-   [How to Setup a Python Environment for Machine Learning and Deep
    Learning with
    Anaconda](http://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/)

This guide was written in the top-down and results-first machine
learning style that you’re used to. It will teach you how to get
results, but it is not a panacea.

You will develop useful skills by working through this guide.

After completing this course, you will:

-   Know how LSTMs work.

-   Know how to prepare data for LSTMs.

-   Know how to apply a suite of types of LSTMs.

-   Know how to tune LSTMs to a problem.

-   Know how to save an LSTM model and use it to make predictions.

Next, let’s review the lessons.

**Mini-Course Overview** 
------------------------

This mini-course is broken down into 14 lessons.
You could complete one lesson per day (recommended) or complete all of
the lessons in one day (hardcore!).
It really depends on the time you have available and your level of
enthusiasm.

Below are 14 lessons that will get you started and productive with LSTMs
in Python. The lessons are divided into three main themes: foundations,
models, and advanced.

![](../earum_voluptatem_provident_html_ffac9678c6938ac4.png)

Overview of LSTM Mini-Course

### **Foundations** 

The focus of these lessons are the things that you need to know before
using LSTMs.

-   **Lesson 01**: What are LSTMs?

-   **Lesson 02**: How LSTMs are trained

-   **Lesson 03**: How to prepare data for LSTMs

-   **Lesson 04**: How to develop LSTMs in Keras

### **Models** 

-   **Lesson 05**: How to develop Vanilla LSTMs

-   **Lesson 06**: How to develop Stacked LSTMs

-   **Lesson 07**: How to develop CNN LSTMs

-   **Lesson 08**: How to develop Encoder-Decoder LSTMs

-   **Lesson 09**: How to develop Bi-directional LSTMs

-   **Lesson 10**: How to develop LSTMs with Attention

-   **Lesson 11**: How to develop Generative LSTMs

### **Advanced** 

-   **Lesson 12**: How to tune LSTM hyperparameters

-   **Lesson 13**: How to update LSTM models

-   **Lesson 14**: How to make predictions with LSTMs


The lessons expect you to go off and find out how to do things. I will
give you hints, but part of the point of each lesson is to force you to
learn where to go to look for help (hint, I have all of the answers on
this blog; use the search).
I do provide more help in the early lessons because I want you to build
up some confidence and inertia.

Hang in there; don’t give up!

**Foundations** 
---------------

The lessons in this section are designed to give you an understanding of
how LSTMs work and how to implement LSTM models using the Keras library.

**Lesson 1: What are LSTMs?** 
-----------------------------

### **Goal** 

The goal of this lesson is to understand LSTMs from a high-level
sufficiently so that you can explain what they are and how they work to
a colleague or manager.

### **Questions** 

-   What is sequence prediction and what are some general examples?

-   What are the limitations of traditional neural networks for sequence
    prediction?

-   What is the promise of RNNs for sequence prediction?

-   What is the LSTM and what are its constituent parts?

-   What are some prominent applications of LSTMs?

### **Further Reading** 

-   [Crash Course in Recurrent Neural Networks for Deep
    Learning](http://machinelearningmastery.com/crash-course-recurrent-neural-networks-deep-learning/)

-   [Gentle Introduction to Models for Sequence Prediction with
    Recurrent Neural
    Networks](http://machinelearningmastery.com/handle-long-sequences-long-short-term-memory-recurrent-neural-networks/)

-   [The Promise of Recurrent Neural Networks for Time Series
    Forecasting](http://machinelearningmastery.com/promise-recurrent-neural-networks-time-series-forecasting/)

-   [On the Suitability of Long Short-Term Memory Networks for Time
    Series
    Forecasting](http://machinelearningmastery.com/suitability-long-short-term-memory-networks-time-series-forecasting/)

-   [A Gentle Introduction to Long Short-Term Memory Networks by the
    Experts](http://machinelearningmastery.com/gentle-introduction-long-short-term-memory-networks-experts/)

-   [8 Inspirational Applications of Deep
    Learning](http://machinelearningmastery.com/inspirational-applications-deep-learning/)

**Lesson 2: How LSTMs are trained** 
-----------------------------------

### **Goal** 

The goal of this lesson is to understand how LSTM models are trained on
example sequences.

### **Questions** 

-   What common problems afflict the training of traditional RNNs?

-   How does the LSTM overcome these problems?

-   What algorithm is used to train LSTMs?

-   How does Backpropagation Through Time work?

-   What is truncated BPTT and what benefit does it offer?

-   How is BPTT implemented and configured in Keras?

### **Further Reading** 

-   [A Gentle Introduction to Backpropagation Through
    Time](http://machinelearningmastery.com/gentle-introduction-backpropagation-time/)

-   [How to Prepare Sequence Prediction for Truncated Backpropagation
    Through Time in
    Keras](http://machinelearningmastery.com/truncated-backpropagation-through-time-in-keras/)

**Lesson 3: How to prepare data for LSTMs** 
-------------------------------------------

### **Goal** 

The goal of this lesson is to understand how to prepare sequence
prediction data for use with LSTM models.

### **Questions** 

-   How do you prepare numeric data for use with LSTMs?

-   How do you prepare categorical data for use with LSTMs?

-   How do you handle missing values in sequences when using LSTMs?

-   How do you frame a sequence as a supervised learning problem?

-   How do you handle long sequences when working with LSTMs?

-   How do you handle input sequences with different lengths?

-   How do you reshape input data for LSTMs in Keras?

### **Experiment** 

Demonstrate how to transform a numerical input sequence into a form
suitable for training an LSTM.

### **Further Reading** 

-   [How to Scale Data for Long Short-Term Memory Networks in
    Python](http://machinelearningmastery.com/how-to-scale-data-for-long-short-term-memory-networks-in-python/)

-   [How to One Hot Encode Sequence Data in
    Python](http://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/)

-   [How to Handle Missing Timesteps in Sequence Prediction Problems
    with
    Python](http://machinelearningmastery.com/handle-missing-timesteps-sequence-prediction-problems-python/)

-   [How to Convert a Time Series to a Supervised Learning Problem in
    Python](http://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/)

-   [How to Handle Very Long Sequences with Long Short-Term Memory
    Recurrent Neural
    Networks](http://machinelearningmastery.com/handle-long-sequences-long-short-term-memory-recurrent-neural-networks/)

-   [How to Prepare Sequence Prediction for Truncated Backpropagation
    Through Time in
    Keras](http://machinelearningmastery.com/truncated-backpropagation-through-time-in-keras/)

-   [Data Preparation for Variable-Length Input
    Sequences](http://machinelearningmastery.com/data-preparation-variable-length-input-sequences-sequence-prediction/)

**Lesson 4: How to develop LSTMs in Keras** 
-------------------------------------------

### **Goal** 

The goal of this lesson is to understand how to define, fit, and
evaluate LSTM models using the Keras deep learning library in Python.

### **Questions** 

-   How do you define an LSTM Model?

-   How do you compile an LSTM Model?

-   How do you fit an LSTM Model?

-   How do you evaluate an LSTM Model?

-   How do you make predictions with an LSTM Model?

-   How can LSTMs be applied to different types of sequence prediction
    problems?

### **Experiment** 

Prepare an example that demonstrates the life-cycle of an LSTM model on
a sequence prediction problem.

### **Further Reading** 

-   [The 5 Step Life-Cycle for Long Short-Term Memory Models in
    Keras](http://machinelearningmastery.com/5-step-life-cycle-long-short-term-memory-models-keras/)

-   [Gentle Introduction to Models for Sequence Prediction with
    Recurrent Neural
    Networks](http://machinelearningmastery.com/handle-long-sequences-long-short-term-memory-recurrent-neural-networks/)

**Models** 
----------

The lessons in this section are designed to teach you how to get results
with LSTM models on sequence prediction problems.

**Lesson 5: How to develop Vanilla LSTMs** 
------------------------------------------

### **Goal** 

The goal of this lesson is to learn how to develop and evaluate vanilla
LSTM models.

-   What is the vanilla LSTM architecture?

-   What are some examples where the vanilla LSTM has been applied?

### **Experiment** 

Design and execute an experiment that demonstrates a vanilla LSTM on a
sequence prediction problem.

### **Further Reading** 

-   [Sequence Classification with LSTM Recurrent Neural Networks in
    Python with
    Keras](http://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/)

-   [Time Series Prediction with LSTM Recurrent Neural Networks in
    Python with
    Keras](http://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/)

-   [Time Series Forecasting with the Long Short-Term Memory Network in
    Python](http://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/)

**Lesson 6: How to develop Stacked LSTMs** 
------------------------------------------

### **Goal** 

The goal of this lesson is to learn how to develop and evaluate stacked
LSTM models.

### **Questions** 

-   What are the difficulties in using a vanilla LSTM on a sequence
    problem with hierarchical structure?

-   What are stacked LSTMs?

-   What are some examples of where the stacked LSTM has been applied?

-   What benefits do stacked LSTMs provide?

-   How can a stacked LSTM be implemented in Keras?

### **Experiment** 

Design and execute an experiment that demonstrates a stacked LSTM on a
sequence prediction problem with hierarchical input structure.

### **Further Reading** 

-   [Sequence Classification with LSTM Recurrent Neural Networks in
    Python with
    Keras](http://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/)

-   [Time Series Prediction with LSTM Recurrent Neural Networks in
    Python with
    Keras](http://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/)

**Lesson 7: How to develop CNN LSTMs** 
--------------------------------------

### **Goal** 

The goal of this lesson is to learn how to develop LSTM models that use
a Convolutional Neural Network on the front end.

### **Questions** 

-   What are the difficulties of using a vanilla LSTM with spatial input
    data?

-   What is the CNN LSTM architecture?

-   What are some examples of the CNN LSTM?

-   What benefits does the CNN LSTM provide?

-   How can the CNN LSTM architecture be implemented in Keras?

### **Experiment** 

Design and execute an experiment that demonstrates a CNN LSTM on a
sequence prediction problem with spatial input.

### **Further Reading** 

-   [Sequence Classification with LSTM Recurrent Neural Networks in
    Python with
    Keras](http://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/)

**Lesson 8: How to develop Encoder-Decoder LSTMs** 
--------------------------------------------------

### **Goal** 

The goal of this lesson is to learn how to develop encoder-decoder LSTM
models.

### **Questions** 

-   What are sequence-to-sequence (seq2seq) prediction problems?

-   What are the difficulties of using a vanilla LSTM on seq2seq
    problems?

-   What is the encoder-decoder LSTM architecture?

-   What are some examples of encoder-decoder LSTMs?

-   What are the benefits of encoder-decoder LSTMs?

-   How can encoder-decoder LSTMs be implemented in Keras?

### **Experiment** 

Design and execute an experiment that demonstrates an encoder-decoder
LSTM on a sequence-to-sequence prediction problem.

### **Further Reading** 

-   [How to Use the TimeDistributed Layer for Long Short-Term Memory
    Networks in
    Python](http://machinelearningmastery.com/timedistributed-layer-for-long-short-term-memory-networks-in-python/)

-   [How to Learn to Add Numbers with seq2seq Recurrent Neural
    Networks](http://machinelearningmastery.com/learn-add-numbers-seq2seq-recurrent-neural-networks/)

-   [How to use an Encoder-Decoder LSTM to Echo Sequences of Random
    Integers](http://machinelearningmastery.com/how-to-use-an-encoder-decoder-lstm-to-echo-sequences-of-random-integers/)

**Lesson 9: How to develop Bi-directional LSTMs** 
-------------------------------------------------

### **Goal** 

The goal of this lesson is to learn how to developer Bidirectional LSTM
models.

### **Questions** 

-   What is a bidirectional LSTM?

-   What are some examples where bidirectional LSTMs have been used?

-   What benefit does a bidirectional LSTM offer over a vanilla LSTM?

-   What concerns regarding time steps does a bidirectional architecture
    raise?

-   How can bidirectional LSTMs be implemented in Keras?

### **Experiment** 

Design and execute an experiment that compares forward, backward, and
bidirectional LSTM models on a sequence prediction problem.

### **Further Reading** 

-   [How to Develop a Bidirectional LSTM For Sequence Classification in
    Python with
    Keras](http://machinelearningmastery.com/develop-bidirectional-lstm-sequence-classification-python-keras/)

**Lesson 10: How to develop LSTMs with Attention** 
--------------------------------------------------

### **Goal** 

The goal of this lesson is to learn how to develop LSTM models with
attention.

### **Questions** 

-   What impact do long sequences with neutral information have on
    LSTMs?

-   What is attention in LSTM models?

-   What are some examples where attention has been used in LSTMs?

-   What benefit does attention provide to sequence prediction?

-   How can an attention architecture be implemented in Keras?

### **Experiment** 

Design and execute an experiment that applies attention to a sequence
prediction problem with long sequences of neutral information.

### **Further Reading** 

-   [Attention in Long Short-Term Memory Recurrent Neural
    Networks](http://machinelearningmastery.com/attention-long-short-term-memory-recurrent-neural-networks/)

**Lesson 11: How to develop Generative LSTMs** 
----------------------------------------------

### **Goal** 

The goal of this lesson is to learn how to develop LSTMs for use in
generative models.

-   What are generative models?

-   How can LSTMs be used as generative models?

-   What are some examples of LSTMs as generative models?

-   What benefits do LSTMs have as generative models?

### **Experiment** 

Design and execute an experiment to learn a corpus of text and generate
new samples of text with the same syntax, grammar, and style.

### **Further Reading** 

-   [Text Generation With LSTM Recurrent Neural Networks in Python with
    Keras](http://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/)

**Advanced** 
------------

The lessons in this section are designed to teach you how to get the
most from your LSTM models on your own sequence prediction problems.

**Lesson 12: How to tune LSTM hyperparameters** 
-----------------------------------------------

### **Goal** 

The goal of this lesson is to learn how to tune LSTM hyperparameters.

### **Questions** 

-   How can we diagnose over-learning or under-learning of an LSTM
    model?

-   What are two schemes for tuning model hyperparameters?

-   How can model skill be reliably estimated given LSTMs are stochastic
    algorithms?

-   List LSTM hyperparameters that can be tuned, with examples of values
    that could be evaluated for:

    -   Model initialization and behavior.

    -   Model architecture and structure.

    -   Learning behavior.

### **Experiment** 

Design and execute an experiment to tune one hyperparameter of an LSTM
and select the best configuration.

### **Further Reading** 

-   [How to Evaluate the Skill of Deep Learning
    Models](http://machinelearningmastery.com/evaluate-skill-deep-learning-models/)

-   [How to Tune LSTM Hyperparameters with Keras for Time Series
    Forecasting](http://machinelearningmastery.com/tune-lstm-hyperparameters-keras-time-series-forecasting/)

-   [How to Grid Search Hyperparameters for Deep Learning Models in
    Python With
    Keras](http://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/)

-   [How To Improve Deep Learning
    Performance](http://machinelearningmastery.com/improve-deep-learning-performance/)

**Lesson 13: How to update LSTM models** 
----------------------------------------

### **Goal** 

The goal of this lesson is to learn how to update LSTM models after new
data becomes available.

### **Questions** 

-   What are the benefits of updating LSTM models in response to new
    data?

-   What are some schemes for updating an LSTM model with new data?

### **Experiment** 

Design and execute an experiment to fit an LSTM model to a sequence
prediction problem that contrasts the effect on the model skill of
different model update schemes.

### **Further Reading** 

-   [How to Update LSTM Networks During Training for Time Series
    Forecasting](http://machinelearningmastery.com/update-lstm-networks-training-time-series-forecasting/)

**Lesson 14: How to make predictions with LSTMs** 
-------------------------------------------------

### **Goal** 

The goal of this lesson is to learn how to finalize an LSTM model and
use it to make predictions on new data.

### **Questions** 

-   How do you save model structure and weights in Keras?

-   How do you fit a final LSTM model?

-   How do you make a prediction with a finalized model?

### **Experiment** 

Design and execute an experiment to fit a final LSTM model, save it to
file, then later load it and make a prediction on a held back validation
dataset.

### **Further Reading** 

-   [Save and Load Your Keras Deep Learning
    Models](http://machinelearningmastery.com/save-load-keras-deep-learning-models/)

-   [How to Train a Final Machine Learning
    Model](http://machinelearningmastery.com/train-final-machine-learning-model/)

**The End!** 
------------

**(*****Look How Far You Have Come*****)** 
------------------------------------------

You made it. Well done!

Take a moment and look back at how far you have come. Here is what you
have learned:

1.  What LSTMs are and why they are the go-to deep learning technique
    for sequence prediction.

2.  That LSTMs are trained using the BPTT algorithm which also imposes a
    way of thinking about your sequence prediction problem.

3.  That data preparation for sequence prediction may involve Masking
    missing values and splitting, padding and truncating input
    sequences.

4.  That Keras provides a 5-step life-cycle for LSTM models, including
    define, compile, fit, evaluate, and predict.

5.  That the vanilla LSTM is comprised of an input layer, a hidden LSTM
    layer, and a dense output layer.

6.  That hidden LSTM layers can be stacked but must expose the output of
    the entire sequence from layer to layer.

7.  That CNNs can be used as input layers for LSTMs when working with
    image and video data.

8.  That the encoder-decoder architecture can be used when predicting
    variable length output sequences.

9.  That providing input sequences forward and backward in bidirectional
    LSTMs can lift the skill on some problems.

10. That attention can provide an optimization for long input sequences
    that contain neutral information.

11. That LSTMs can learn the structured relationship of input data which
    in turn can be used to generate new examples.

12. That the LSTMs hyperparameters of LSTMs can be tuned much like any
    other stochastic model.

13. That fit LSTM models can be updated when new data is made available.

14. That a final LSTM model can be saved to file and later loaded in
    order to make predictions on new data.

Don’t make light of this; you have come a long way in a short amount of
time.

This is just the beginning of your LSTM journey with Keras. Keep
practicing and developing your skills.

\

