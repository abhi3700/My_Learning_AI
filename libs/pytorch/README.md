# Pytorch

## About
* PyTorch is a Python-based scientific computing package serving two broad purposes:
	- A replacement for NumPy to use the power of GPUs and other accelerators.
	- An automatic differentiation library that is useful to implement neural networks.

## Installation
### Using pip on Windows with CUDA 11.1 [Source](https://pytorch.org/get-started/locally/)
> Install CUDA, if your machine has a CUDA-enabled GPU.

```console
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

## Platforms
### Online: Google Colab (RAM: 12GB, Storage: 107.77 GB)
> Connected to the "Python 3 Google Compute Backend Engine"

* Just install using pip in notebook
* In order to connect using GPU, TPU, "Edit >> Notebook Settings >> Hardware Accelerator"

### Local:

## Getting Started
* To check if pytorch is working properly
```console
import torch
x = torch.rand(5, 3)
print(x)
```
* to check if your GPU driver and CUDA is enabled and accessible by PyTorch
```console
import torch
torch.cuda.is_available()
```


## Coding

### Lecture-1
#### a. Code folder structure for every examples (Vision, NLP) goes like this:
```console
data/
experiments/
model/
    net.py    				# specifies the neural network architecture, the loss function and evaluation metrics
    data_loader.py 		# specifies how the data should be fed to the network
train.py  						#	contains the main training loop
evaluate.py 					# contains the main loop for evaluating the model
search_hyperparams.py
synthesize_results.py
evaluate.py
utils.py 							# utility functions for handling hyperparams/logging/storing model
```

> Once you get the high-level idea, depending on your task and dataset, you might want to modify
> - `model/net.py` to change the model, i.e. how you transform your input into your prediction as well as your loss, etc.
> - `model/data_loader.py` to change the way you feed data to the model.
> - `train.py` and evaluate.py to make changes specific to your problem, if required
> Once you get something working for your dataset, feel free to edit any part of the code to suit your own needs.

#### b. Tensors and Variables
* Read [this](https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html#sphx-glr-beginner-blitz-tensor-tutorial-py)

## Repositories
* Simple examples to introduce PyTorch - https://github.com/jcjohnson/pytorch-examples

## References
* Pytorch Blogs - https://pytorch.org/blog/
* Pytorch blog by Standford - https://cs230.stanford.edu/blog/pytorch/