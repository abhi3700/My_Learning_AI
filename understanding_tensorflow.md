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
* TensorFlow mathematics: operation, graph [1]
  - when you `“add”` something in TF, you are designing an `“add”` operation, not actually adding anything.
  - All those operations are organised as a `Graph`, your Graph holds operations and tensors, not values.
* Traditional algorithm v/s ML-based algorithm
	- In traditional algorithm writing the rules takes time | But here, data training & computing takes time.



## References
1. https://blog.metaflow.fr/tensorflow-a-primer-4b3fa0978be3
