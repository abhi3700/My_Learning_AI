# Understanding TensorFlow
* Difference b/w normal mathematical operation v/s TensorFlow [1]
```py
import tensorflow as tf

# Usual way to do math operation in a program
a = 2 + 2
print(a) # => 4

# TensorFlow's way
a = tf.add(2, 2)
print(a) # => Tensor("Add:0", shape=(), dtype=int32)
```


## References
1. https://blog.metaflow.fr/tensorflow-a-primer-4b3fa0978be3
