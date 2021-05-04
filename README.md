# My_Learning_AI
AI includes ML, DL, etc. techniques for prediction algorithms

## Installation
1. ### [Anaconda](https://www.anaconda.com/distribution/#download-section)
> NOTE: Don't forget to tick `Add to PATH` option during the installation.
2. ### [Cuda toolkit](https://developer.nvidia.com/cuda-10.0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal): download via local copy. 
> NOTE: The environment variables are set during the installation.

* __CUPTI:__ Installed via local copy (downloaded aboves)
	+ After this, add to PATH:
	Ensure this path is present in the environment variable - PATH:
	```md
	C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\extras\CUPTI\libx64
	```
* __cuDNN:__ Download "__cuDNN Library for Windows 10__" from [here](https://developer.nvidia.com/rdp/cudnn-download)
	+ Copy the following files into the CUDA Toolkit directory. <br/>
		a. Copy `<installpath>\cuda\bin\cudnn64_7.dll` to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\bin`.<br/>
		b. Copy `<installpath>\cuda\ include\cudnn.h` to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\include`.<br/>
		c. Copy `<installpath>\cuda\lib\x64\cudnn.lib` to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\lib\x64`.<br/>
	+ Ensure the following values (in Environment Variables at the bottom of the window) are set: [SOURCE](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#installwindows)
	```md
	Variable Name: CUDA_PATH 
	Variable Value: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0		
	```		
	+ Add to PATH:
Ensure these 2 paths are present in the environment variable - PATH:
```md
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\bin
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\libnvvp
```
* Verify CUDA installation
```console
C:\Users\abhijit>nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2018 NVIDIA Corporation
Built on Sat_Aug_25_21:08:04_Central_Daylight_Time_2018
Cuda compilation tools, release 10.0, V10.0.130
```

3. ### Libraries 
	* #### TensorFlow
		- CPU: `pip install tensorflow`
		- GPU: `pip install tensorflow-gpu`

## Github Repositories
* TensorFlow Examples - https://github.com/aymericdamien/TensorFlow-Examples
* TensorFlow 2.0 examples - https://github.com/dragen1860/TensorFlow-2.x-Tutorials
* Hands-on Machine Learning with Scikit-Learn and TensorFlow - [Book](https://github.com/abhi3700/My_Learning_AI/blob/master/books/Hands%20On%20Machine%20Learning%20with%20Scikit%20Learn%20and%20TensorFlow.pdf), [Repository](https://github.com/ageron/handson-ml)
* Scikit Learn for Python in C++ (Single Headers and No dependencies) - https://github.com/VISWESWARAN1998/sklearn
* Tensorflow examples written in C++ - https://github.com/ksachdeva/tensorflow-cc-examples
* 500 AI ML DL Computer vision NLP Projects with code - https://github.com/ashishpatel26/500-AI-Machine-learning-Deep-learning-Computer-vision-NLP-Projects-with-code

## Books
1. [Hands-on Machine Learning with Scikit-Learn and TensorFlow 1st edition](https://github.com/abhi3700/My_Learning_AI/blob/master/books/Hands%20On%20Machine%20Learning%20with%20Scikit%20Learn%20and%20TensorFlow.pdf), [Github](https://github.com/ageron/handson-ml)
1. [Hands-on Machine Learning with Scikit-Learn and TensorFlow 2nd edition](https://github.com/abhi3700/My_Learning_AI/blob/master/books/Hands_On_Machine_Learning_with_Scikit_Learn_Keras_2nd_edition.pdf), [Github](https://github.com/ageron/handson-ml2)
1. [Machine Learning with TensorFlow](https://github.com/abhi3700/My_Learning_AI/blob/master/books/Machine%20Learning%20with%20TensorFlow.pdf)
1. [Mathematics for Machine Learning](https://github.com/abhi3700/My_Learning_AI/blob/master/books/Mathematics%20For%20Machine%20Learning.pdf)
1. [Getting Started with TensorFlow](https://github.com/abhi3700/My_Learning_AI/blob/master/books/Getting%20Started%20with%20TensorFlow.pdf)

## Videos
### Understanding
* Neural Networks (explained by Blue1Brown) - https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
* Live Coding (Neural Networks using C++) - https://www.youtube.com/playlist?list=PL2-7U6BzddIYBOl98DDsmpXiTcj1ojgJG
* C++ Machine Learning - https://www.youtube.com/playlist?list=PL79n_WS-sPHKklEvOLiM1K94oJBsGnz71
* Deep Learning for Coders (2020) - https://www.youtube.com/watch?v=_QUEXsHfsA0&list=RDCMUCX7Y2qWriXpqocG97SFW2OQ
* Neural Networks by Blue1Brown - https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3p

### Talks
* [ ] [ML BuzzWords demystified - Oleksandra Sopova & Natalia](https://youtu.be/4pGhvcVz1Xg)
* [ ] [Machine Learning: The Bare Math Behind Libraries - Piotr Czajka and Łukasz Gebel](https://youtu.be/yoP2uNYFGSw)
* [ ] [ML and the IoT: Living on the Edge - Brandon Satrom](https://youtu.be/5SYjR2D4p0c)

## Papers
* Browse state-of-the-art - https://paperswithcode.com/sota
* Best AI papers 2020 - https://github.com/louisfb01/Best_AI_paper_2020
* [PyTorch implementation of our ICRA'19 paper on "Self-supervised Sparse-to-Dense: Self-supervised Depth Completion from LiDAR and Monocular Camera"](https://github.com/fangchangma/self-supervised-depth-completion)


## References
* TensorFlow Tutorial - https://www.guru99.com/tensorflow-tutorial.html
* TensorFlow Tutorial for beginners - https://www.datacamp.com/community/tutorials/tensorflow-tutorial
* 10 updates about TensorFlow 2.0 - https://www.datacamp.com/community/tutorials/ten-important-updates-tensorflow
* Infographic – A Complete Guide on Getting Started with Deep Learning in Python - https://www.analyticsvidhya.com/blog/2018/08/infographic-complete-deep-learning-path/
* A Complete Guide on Getting Started with Deep Learning in Python - https://www.analyticsvidhya.com/blog/2016/08/deep-learning-path/
* [Linear Regression Formula Explained](https://hackerstreak.com/linear-regression-formula/)
* Scikit-learn Tutorial: Machine Learning in Python – Dataquest - https://www.dataquest.io/blog/sci-kit-learn-tutorial/
* Python Machine Learning: Scikit-Learn Tutorial - DataCamp - https://www.datacamp.com/community/tutorials/machine-learning-python
* Scikit-Learn Cheat Sheet: Python Machine Learning - DataCamp - [Scikit learn Cheatsheet (in PDF)]("./docs/Scikit_Learn_Cheat_Sheet_Python.pdf"), https://www.datacamp.com/community/blog/scikit-learn-cheat-sheet

## Courses
* [Foundation of Machine Learning](https://bloomberg.github.io/foml/#home) (Learn Maths with Python)
* [Machine Learning by Andrew Ng on Coursera](https://www.coursera.org/learn/machine-learning)
* [Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning on Coursera](https://www.coursera.org/learn/introduction-tensorflow)
* [Introduction to Deep Learning with PyTorch](https://classroom.udacity.com/courses/ud188)
* [Machine Learning Interview Preparation](https://classroom.udacity.com/courses/ud1001/)
* [ML Crash course by Google](https://developers.google.com/machine-learning/crash-course/)

## Blogs
* [Project Spock at Tubi: Understanding Content Using Deep Learning for NLP](https://medium.com/pytorch/project-spock-at-tubi-understanding-content-using-deep-learning-for-nlp-2b102ca25ed9)