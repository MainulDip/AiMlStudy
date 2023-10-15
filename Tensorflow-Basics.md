### Overview:
Tensorflow Workflow on Google Colab or Local Jupyter Notebook in a conda environment.

### Concepts:

* Neural Networks: Neural networks are a set of algorithms, Inspired by the human brain, these networks are designed to mimic the way biological neurons signal to each other to recognize patterns. The patterns they recognize are numerical, contained in vectors. Consist of 3 layers (Input + Hidden + Output)

* Deep Learning: Its a `stacked neural networks`. Consist of More than 3 layers with more than one Hidden layers.

* Layers: The layers are made of nodes. A node is just a place where computation happens. A node is an artificial neuron of ANN. each node/neuron in a layer is connected to some or all of the nodes/neurons in the next layer.

* Weights and Biases: Weights and biases (commonly referred to as w and b) are the learnable parameters of Neural Networks. Neurons are the basic units of a neural network. In an ANN (Artificial Neural Network), each neuron in a layer is connected to some or all of the neurons in the next layer. When the inputs are transmitted between neurons, the weights are applied to the inputs along with the bias.
Y = Summation-of ( weight + input ) + b

`Weights` control the signal (or the strength of the connection) between two neurons. In other words, a weight decides how much influence the input will have on the output.

`Biases`, which are constant, are an additional input into the next layer that will always have the value of 1. Bias units are not influenced by the previous layer (they do not have any incoming connections) but they do have outgoing connections with their own weights. The bias unit guarantees that even when all the inputs are zeros there will still be an activation in the neuron.

Note: https://wiki.pathmind.com/neural-network

### Keras vs TensorFlow Core APIs :
Keras is the high-level API of the TensorFlow platform. Keras covers every step of the machine learning workflow, from data processing to hyperparameter tuning to deployment. Every TensorFlow user should use the Keras APIs by default Unless Very Low Level Access.

* Tensorflow Core APIs: It's the low-level APIs. For building more complex workflow like building tools on top of TensorFlow or developing custom high-performance platform, the Low Level APIs provide better control. Unless that Keras can do all the regular workflow tasks.

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