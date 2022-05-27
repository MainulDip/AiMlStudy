### Tensorflow Core Learning Algorithms
Commonly used these 4 algorithms

    Linear Regression
    Classification
    Clustering
    Hidden Markov Models

### Linear Regression:
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