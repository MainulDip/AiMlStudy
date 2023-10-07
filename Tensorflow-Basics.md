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