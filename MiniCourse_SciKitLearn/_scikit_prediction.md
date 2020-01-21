
**How to Make Predictions with scikit-learn**

#### **How to predict classification or regression outcomes**

Once you choose and fit a final machine learning model in scikit-learn,
you can use it to make predictions on new data instances.
There is some confusion amongst beginners about how exactly to do this.
I often see questions such as:

How do I make predictions with my model in scikit-learn?

In this tutorial, you will discover exactly how you can make
classification and regression predictions with a finalized machine
learning model in the scikit-learn Python library.

After completing this tutorial, you will know:

-   How to finalize a model in order to make it ready for making
    predictions.

-   How to make class and probability predictions in scikit-learn.

-   How to make regression predictions in scikit-learn.

Discover how to prepare data with pandas, fit and evaluate models with
scikit-learn, and more [in my new
book](https://machinelearningmastery.com/machine-learning-with-python/),
with 16 step-by-step tutorials, 3 projects, and full python code.

Let’s get started.

![](../soluta_consectetur_iusto_html_13af4d9855a2406f.jpg)


**Tutorial Overview** 
---------------------

This tutorial is divided into 3 parts; they are:

1.  First Finalize Your Model

2.  How to Predict With Classification Models

3.  How to Predict With Regression Models

**1. First Finalize Your Model** 
--------------------------------

Before you can make predictions, you must train a final model.
You may have trained models using k-fold cross validation or train/test
splits of your data. This was done in order to give you an estimate of
the skill of the model on out-of-sample data, e.g. new data.

These models have served their purpose and can now be discarded.
You now must train a final model on all of your available data.
You can learn more about how to train a final model here:

-   [How to Train a Final Machine Learning
    Model](https://machinelearningmastery.com/train-final-machine-learning-model/)

**2. How to Predict With Classification Models** 
------------------------------------------------

Classification problems are those where the model learns a mapping
between input features and an output feature that is a label, such as
“*spam*” and “*not spam*.”

Below is sample code of a finalized
[LogisticRegression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
model for a simple binary classification problem.

Although we are using *LogisticRegression* in this tutorial, the same
functions are available on practically all classification algorithms in
scikit-learn.

```
# example of training a final classification model
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_blobs
# generate 2d classification dataset
X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)
# fit final model
model = LogisticRegression()
model.fit(X, y)
```

##### Run Notebook
Click notebook `13.ipynb` in jupterLab UI and run jupyter notebook.

After finalizing your model, you may want to save the model to file,
e.g. via pickle. Once saved, you can load the model any time and use it
to make predictions. For an example of this, see the post:

-   [Save and Load Machine Learning Models in Python with
    scikit-learn](https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/)

For simplicity, we will skip this step for the examples in this
tutorial.
There are two types of classification predictions we may wish to make
with our finalized model; they are class predictions and probability
predictions.

### **Class Predictions** 

A class prediction is: given the finalized model and one or more data
instances, predict the class for the data instances.
We do not know the outcome classes for the new data. That is why we need
the model in the first place.

We can predict the class for new data instances using our finalized
classification model in scikit-learn using the *predict()* function.
For example, we have one or more data instances in an array called
*Xnew*. This can be passed to the *predict()* function on our model in
order to predict the class values for each instance in the array.

```
Xnew = [[...], [...]]
ynew = model.predict(Xnew)
```

### **Multiple Class Predictions** 

Let’s make this concrete with an example of predicting multiple data
instances at once.

```
# example of training a final classification model
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_blobs
# generate 2d classification dataset
X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)
# fit final model
model = LogisticRegression()
model.fit(X, y)
# new instances where we do not know the answer
Xnew, _ = make_blobs(n_samples=3, centers=2, n_features=2, random_state=1)
# make a prediction
ynew = model.predict(Xnew)
# show the inputs and predicted outputs
for i in range(len(Xnew)):
	print("X=%s, Predicted=%s" % (Xnew[i], ynew[i]))
```


##### Run Notebook
Click notebook `14.ipynb` in jupterLab UI and run jupyter notebook.

Running the example predicts the class for the three new data instances,
then prints the data and the predictions together.

```
X=[-0.79415228  2.10495117], Predicted=0
X=[-8.25290074 -4.71455545], Predicted=1
X=[-2.18773166  3.33352125], Predicted=0
```

### **Single Class Prediction** 

If you had just one new data instance, you can provide this as instance
wrapped in an array to the *predict()* function; for example:

```
# example of making a single class prediction
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_blobs
# generate 2d classification dataset
X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)
# fit final model
model = LogisticRegression()
model.fit(X, y)
# define one new instance
Xnew = [[-0.79415228, 2.10495117]]
# make a prediction
ynew = model.predict(Xnew)
print("X=%s, Predicted=%s" % (Xnew[0], ynew[0]))
```

##### Run Notebook
Click notebook `15.ipynb` in jupterLab UI and run jupyter notebook.

Running the example prints the single instance and the predicted class.

```
X=[-0.79415228, 2.10495117], Predicted=0
```

### **A Note on Class Labels** 

When you prepared your data, you will have mapped the class values from
your domain (such as strings) to integer values. You may have used a
[LabelEncoder](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html#sklearn.preprocessing.LabelEncoder).

This *LabelEncoder* can be used to convert the integers back into string
values via the *inverse_transform()* function.
For this reason, you may want to save (pickle) the *LabelEncoder* used
to encode your y values when fitting your final model.

### **Probability Predictions** 

Another type of prediction you may wish to make is the probability of
the data instance belonging to each class.
This is called a probability prediction where given a new instance, the
model returns the probability for each outcome class as a value between
0 and 1.

You can make these types of predictions in scikit-learn by calling the
*predict_proba()* function, for example:

```
Xnew = [[...], [...]]
ynew = model.predict_proba(Xnew)
```

This function is only available on those classification models capable
of making a probability prediction, which is most, but not all, models.

The example below makes a probability prediction for each example in the
*Xnew* array of data instance.

```
# example of making multiple probability predictions
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_blobs
# generate 2d classification dataset
X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)
# fit final model
model = LogisticRegression()
model.fit(X, y)
# new instances where we do not know the answer
Xnew, _ = make_blobs(n_samples=3, centers=2, n_features=2, random_state=1)
# make a prediction
ynew = model.predict_proba(Xnew)
# show the inputs and predicted probabilities
for i in range(len(Xnew)):
	print("X=%s, Predicted=%s" % (Xnew[i], ynew[i]))
```

##### Run Notebook
Click notebook `16.ipynb` in jupterLab UI and run jupyter notebook.

Running the instance makes the probability predictions and then prints
the input data instance and the probability of each instance belonging
to the first and second classes (0 and 1).

```
X=[-0.79415228 2.10495117], Predicted=[0.94556472 0.05443528]
X=[-8.25290074 -4.71455545], Predicted=[3.60980873e-04 9.99639019e-01]
X=[-2.18773166 3.33352125], Predicted=[0.98437415 0.01562585]
```

This can be helpful in your application if you want to present the
probabilities to the user for expert interpretation.

**3. How to Predict With Regression Models** 
--------------------------------------------

Regression is a supervised learning problem where, given input examples,
the model learns a mapping to suitable output quantities, such as “0.1”
and “0.2”, etc.

Below is an example of a finalized *LinearRegression* model. Again, the
functions demonstrated for making regression predictions apply to all of
the regression models available in scikit-learn.

```
# example of training a final regression model
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
# generate regression dataset
X, y = make_regression(n_samples=100, n_features=2, noise=0.1, random_state=1)
# fit final model
model = LinearRegression()
model.fit(X, y)
```

##### Run Notebook
Click notebook `17.ipynb` in jupterLab UI and run jupyter notebook.

We can predict quantities with the finalized regression model by calling
the *predict()* function on the finalized model.
As with classification, the predict() function takes a list or array of
one or more data instances.

### **Multiple Regression Predictions** 

The example below demonstrates how to make regression predictions on
multiple data instances with an unknown expected outcome.

```
# example of training a final regression model
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
# generate regression dataset
X, y = make_regression(n_samples=100, n_features=2, noise=0.1)
# fit final model
model = LinearRegression()
model.fit(X, y)
# new instances where we do not know the answer
Xnew, _ = make_regression(n_samples=3, n_features=2, noise=0.1, random_state=1)
# make a prediction
ynew = model.predict(Xnew)
# show the inputs and predicted outputs
for i in range(len(Xnew)):
	print("X=%s, Predicted=%s" % (Xnew[i], ynew[i]))
```

##### Run Notebook
Click notebook `18.ipynb` in jupterLab UI and run jupyter notebook.

Running the example makes multiple predictions, then prints the inputs
and predictions side-by-side for review.

```
X=[-1.07296862 -0.52817175], Predicted=-61.32459258381131
X=[-0.61175641 1.62434536], Predicted=-30.922508147981667
X=[-2.3015387 0.86540763], Predicted=-127.34448527071137
```

### **Single Regression Prediction** 

The same function can be used to make a prediction for a single data
instance as long as it is suitably wrapped in a surrounding list or
array.

For example:

```
# example of training a final regression model
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
# generate regression dataset
X, y = make_regression(n_samples=100, n_features=2, noise=0.1)
# fit final model
model = LinearRegression()
model.fit(X, y)
# define one new data instance
Xnew = [[-1.07296862, -0.52817175]]
# make a prediction
ynew = model.predict(Xnew)
# show the inputs and predicted outputs
print("X=%s, Predicted=%s" % (Xnew[0], ynew[0]))
```

##### Run Notebook
Click notebook `19.ipynb` in jupterLab UI and run jupyter notebook.

Running the example makes a single prediction and prints the data
instance and prediction for review.

```
X=[-1.07296862, -0.52817175], Predicted=-77.17947088762787
```

**Further Reading** 
-------------------

This section provides more resources on the topic if you are looking to
go deeper.

-   [How to Train a Final Machine Learning
    Model](https://machinelearningmastery.com/train-final-machine-learning-model/)

-   [Save and Load Machine Learning Models in Python with
    scikit-learn](https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/)

-   [scikit-learn API
    Reference](http://scikit-learn.org/stable/modules/classes.html)

### **Summary** 

In this tutorial, you discovered how you can make classification and
regression predictions with a finalized machine learning model in the
scikit-learn Python library.

Specifically, you learned:

-   How to finalize a model in order to make it ready for making
    predictions.

-   How to make class and probability predictions in scikit-learn.

-   How to make regression predictions in scikit-learn.
