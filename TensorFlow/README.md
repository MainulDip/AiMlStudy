## Overview:
TensorFlow Study Using Python (primarly) and mini personalized docs to jump start anytime.

Applicability:
_
    Image Classification
    Data Clustering
    Regression
    Reinforcement Learning
    Natural Language Processing

### Tensors
"A tensor is a generalization of vectors and matrices to potentially higher dimensions. Internally, TensorFlow represents tensors as n-dimensional arrays of base datatypes." (https://www.tensorflow.org/guide/tensor)
>Each tensor has a data type and a shape.

Data Types Include: float32, int32, string and others.
Shape: Represents the dimension of data.

### Creating Tensors:
```py
string = tf.Variable("this is a string", tf.string) 
number = tf.Variable(324, tf.int16)
floating = tf.Variable(3.567, tf.float64)
```
tensor variable is Scalar type ( just one value, not like vector value can be one or multidimentional

### Rank/Degree and shape of Tensors
Number of dimensions involved in the tensor. What we created above is a *tensor of rank 0*, also known as a scalar. 

The shape of a tensor is simply the number of elements that exist in each dimension. TensorFlow will try to determine the shape of a tensor but sometimes it may be unknown.

```py
rank1_tensor = tf.Variable(["Test"], tf.string) # because it's one list only
rank2_tensor = tf.Variable([["test", "ok"], ["test", "yes"], ["tests", "yess"]], tf.string) # numpy=2 because list of list
rank_test = tf.Variable(["Test", "Foo", "Too"], tf.string) 
tf.rank(rank_test) # <tf.Tensor: shape=(), dtype=int32, numpy=1>
tf.rank(rank2_tensor) # <tf.Tensor: shape=(), dtype=int32, numpy=2>
#rank2_tensor.shape # TensorShape([3, 2])
```

### Changing Shape ro Dimention

```py
tensor1 = tf.ones([1,2,3])  # tf.ones() creates a shape [1,2,3] tensor full of ones (6 elements of 1)
tensor2 = tf.reshape(tensor1, [2,3,1])  # reshape existing data to shape [2,3,1] => 2 list, each with 3 list containing 1 element
tensor3 = tf.reshape(tensor2, [3,-1])  # -1 tells the tensor to calculate the size of the dimension in that place automatically
# this will reshape the tensor to [3,2]
# only single input can be -1 for automatic dimention calculation
                                                                            

print(tensor1)
print(tensor2)
print(tensor3)
```