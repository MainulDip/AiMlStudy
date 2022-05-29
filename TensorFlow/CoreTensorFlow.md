### Tensorflow Core Learning Algorithms
Commonly used these 4 algorithms

    Linear Regression
    Classification
    Clustering
    Hidden Markov Models

## Linear Regression:
Linear regression follows a very simple concept. If data points are related linearly, we can generate a line of best fit for these points and use it to predict future values.

```py
import matplotlib.pyplot as plt
import numpy as np

x = [1, 2, 2.5, 3, 4]
y = [1, 4, 7, 9, 15]
plt.plot(x, y, 'ro')
plt.axis([0, 6, 0, 20])
```
We can see that this data has a linear coorespondence. When the x value increases, so does the y. Because of this relation we can create a line of best fit for this dataset. In this example our line will only use one input variable, as we are working with two dimensions. In larger datasets with more features our line will have more features and inputs.

"Line of best fit refers to a line through a scatter plot of data points that best expresses the relationship between those points." (https://www.investopedia.com/terms/l/line-of-best-fit.asp)

Here's a refresher on the equation of a line in 2D.

y=mx+b

Here's an example of a line of best fit for this graph.
```py
plt.plot(x, y, 'ro')
plt.axis([0, 6, 0, 20])
plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
plt.show()
```

### Loading Data and environment setup

```py
!pip install -q sklearn
%tensorflow_version 2.x

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np # optimized version of array, help to calculate complex multidimentional array
import pandas as pd # data analytics tool, help manupulate data
import matplotlib.pyplot as plt # data visualizer
from IPython.display import clear_output
from six.moves import urllib

import tensorflow.compat.v2.feature_column as fc

import tensorflow as tf

# Load dataset.
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') # training data
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') # testing data
# print(dftrain.head()) # head() shows the top 5 entries from the dataset
y_train = dftrain.pop('survived') # cutting survived data and putting it on its own variable
y_eval = dfeval.pop('survived')
print(dftrain.loc[0], y_eval.loc[0])
```

### Training vs Testing Data:
we loaded two different datasets (csv) above. This is because when we train models, we need two sets of data: training and testing.

The training data is what we feed to the model so that it can develop and learn. It is usually a much larger size than the testing data.

The testing data is what we use to evaulate the model and see how well it is performing. We must use a seperate set of data that the model has not been trained on to evaluate it.

### Feature Columns: (Feature is the input information. Lable is the predection)

In the csvs we have two different kinds of information: Categorical and Numeric

Our categorical data is anything that is not numeric! For example, the sex column does not use numbers, it uses the words "male" and "female".

Before we continue and create/train a model we must convet our categorical data into numeric data. We can do this by encoding each category with an integer (ex. male = 1, female = 2).

Fortunately for us TensorFlow has some tools to help!

```py
# Creating Feature Column on Tensor Flow's column format
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                       'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
  vocabulary = dftrain[feature_name].unique()  # gets a list of all unique values from given feature column
  # EX: dftrain['sex'].unique() will output array(['male', 'female'], dtype=object)
  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))
  # this will convert the columns into numeric data into tensorflow's numeric format and store that into feature_columns list
for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

print(feature_columns) 
```

### Traning Process:
For this specific model data is going to be streamed into it in small batches of 32. This means we will not feed the entire dataset to our model at once, but simply small batches of entries. We will feed these batches to our model multiple times according to the number of epochs.

An epoch is simply one stream of our entire dataset. The number of epochs we define is the amount of times our model will see the entire dataset. We use multiple epochs in hope that after seeing the same data multiple times the model will better determine how to estimate it.

Ex. if we have 10 ephocs, our model will see the same dataset 10 times.

Since we need to feed our data in batches and multiple times, we need to create something called an input function. The input function simply defines how our dataset will be converted into batches at each epoch.


### Input Function

The TensorFlow model we are going to use requires that the data we pass it comes in as a tf.data.Dataset object. This means we must create a input function that can convert our current pandas dataframe into that object.

Straight from the TensorFlow documentation (https://www.tensorflow.org/tutorials/estimator/linear)
```py
# Input Function
def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
  def input_function():  # inner function, this will be returned
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))  # create tf.data.Dataset object with data and its label, dectionary representatio of the dataset
    if shuffle:
      ds = ds.shuffle(1000)  # randomize order of data
    ds = ds.batch(batch_size).repeat(num_epochs)  # split dataset into batches of 32 and repeat process for number of epochs
    return ds  # return a batch of the dataset
  return input_function  # return a function object for use

train_input_fn = make_input_fn(dftrain, y_train)  # here we will call the input_function that was returned to us to get a dataset object we can feed to the model
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)
```

### Creating the Model
we are going to use a linear estimator to utilize the linear regression algorithm. 

```py
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
# Creates a linear estimtor by passing the feature columns that created earlier
```

### Training the Model

passing the input functions that we created earlier.
```py
linear_est.train(train_input_fn)  # train
result = linear_est.evaluate(eval_input_fn)  # get model metrics/stats by testing on tetsing data

clear_output()  # clears consoke output
print(result['accuracy'])  # the result variable is simply a dict of stats about our model
```

### Prediction and Probabilities Accuracy Testing:
```py
result = list(linear_est.predict(eval_input_fn))
clear_output()
print(result)
print('dfeval',dfeval.loc[1])
print('y-eval',y_eval.loc[1]) # y_eval is the popped out survived eval data for prediction testing
print(result[1]['probabilities'])
```

### Predictin:
We can use the .predict() method to get survival probabilities from the model. This method will return a list of dicts that store a predicition for each of the entries in our testing data set. Below we've used some pandas magic to plot a nice graph of the predictions.
```py
pred_dicts = list(linear_est.predict(eval_input_fn))
probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])
clear_output()
probs.plot(kind='hist', bins=20, title='predicted survival probabilities')
```

## Classification

Where regression was used to predict a numeric value, classification is used to seperate data points into classes of different labels. In this example we will use a TensorFlow estimator to classify flowers.

https://www.tensorflow.org/tutorials/estimator/premade