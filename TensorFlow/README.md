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

### Creating Tensors
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

### Tensor Slicing:
slice operator can be used on tensors to select specific axes or elements.

When we slice or select elements from a tensor, we can use comma seperated values inside the set of square brackets. Each subsequent value refrences a different dimension of the tensor.

Ex: tensor[dim1, dim2, dim3]

```py
# Creating a 2D tensor
matrix = [[1,2,3,4,5],
          [6,7,8,9,10],
          [11,12,13,14,15],
          [16,17,18,19,20]]

tensor = tf.Variable(matrix, dtype=tf.int32) 
print(tf.rank(tensor)) # tf.Tensor(2, shape=(), dtype=int32)
print(tensor.shape) # (4, 5)

three = tensor[0,2]  # selects the 3rd element from the 1st row
print(three)  # -> 3 # tf.Tensor(3, shape=(), dtype=int32)

row1 = tensor[0]  # selects the first row
print(row1) # tf.Tensor([1 2 3 4 5], shape=(5,), dtype=int32)

column1 = tensor[:, 0]  # selects the first column
print(column1) # tf.Tensor([ 1  6 11 16], shape=(4,), dtype=int32)

row_2_and_4 = tensor[1::2]  # selects second and fourth row
print(row_2_and_4) # tf.Tensor([[ 6  7  8  9 10] [16 17 18 19 20]], shape=(2, 5), dtype=int32)

column_1_in_row_2_and_3 = tensor[1:3, 0]
print(column_1_in_row_2_and_3) # tf.Tensor([ 6 11], shape=(2,), dtype=int32)
```

### Types of Tensors

Before we go to far, I will mention that there are diffent types of tensors. These are the most used and we will talk more in depth about each as they are used.

    Variable
    Constant
    Placeholder
    SparseTensor

With the execption of Variable all these tensors are immuttable, meaning their value may not change during execution.

For now, it is enough to understand that we use the Variable tensor when we want to potentially change the value of our tensor.
Sources

Most of the information is taken direclty from the TensorFlow website which can be found below.

https://www.tensorflow.org/guide/tensor
