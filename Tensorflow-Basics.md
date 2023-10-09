### Overview:
Tensorflow Workflow on Google Colab or Local Jupyter Notebook in a conda environment.

### Importing
```python
# Import Tensorflow
import tensorflow as tf
import tensorflow_hub as hub
print("TF version:", tf.__version__)
print("TF Hub version:", hub.__version__)

# Check For GPU availability (o use GPU change Runtime to GPU)
print("GPU", "available (Yess!!!!!!!!)" if tf.config.list_physical_devices("GPU") else "not Available")
```

### Brief Of Tensorflow:
* Tensor: Tensors are matrix (multidimensional array) of numbers, those run inside a GPU. In AIML all data are converted into numbers (if not).
* TensorFlow : Tensors with Machine Learning Workflow.

### 1. Convert Data into Tensor:
With all machine learning models, data has to be in numerical format.
- First access data and check the labels (using pandas)
```py
import pandas as pd
labels_csv = pd.read_csv("Path-To-CSV")
print(labels_csv.describe())
print(labels_csv.head())

# how many images per breed
image_count = labels_csv["breed"].value_counts()
image_count.plot.bar(figsize=(20,10))
```


### Colab Shortcuts:
- Find all the shortcuts using ctrl+MH