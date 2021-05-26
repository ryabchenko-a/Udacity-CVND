# Image Captioning Project

![Original Paper](https://user-images.githubusercontent.com/61123874/119670284-0dde0700-be39-11eb-8c42-51db7083176d.png)

In this project I've built a system that is able to automaticaly generate description of image content (caption).  
1. First, it uses CNN as an encoder to find content features.  
2. Then, it embeds the features and passess it to RNN of LSTM cells where each cell's output is an input to the next cell (which runs until the <end> word is predicted or word limit is reached).
3. Last, it uses gready search or beam search to generate the captions.

## Detailed description of the project

The project consists of 4 ipynb notebooks.

### [0_Dataset.ipynb](https://github.com/ryabchenko-a/Udacity-CVND/blob/main/Facial%20Keypoints%20Detector/1.%20Load%20and%20Visualize%20Data.ipynb)
![Coco Dataset Examples](https://github.com/ryabchenko-a/Udacity-CVND/raw/7b87e9dee9ee1edcb9712cede0a96a86b14461b4/Image%20Captioning/images/coco-examples.jpg)

This notebook explores the dataset I work with, that is 2014 [MS COCO Dataset](https://cocodataset.org/), show example images and annotations.  

### [1_Preliminaries.ipynb](https://github.com/ryabchenko-a/Udacity-CVND/blob/main/Image%20Captioning/1_Preliminaries.ipynb)

This notebook shows some preliminaries, that is: data pre-processing, how [data_loader.py](https://github.com/ryabchenko-a/Udacity-CVND/blob/main/Image%20Captioning/data_loader.py) works
as well as how to set up the RNN (LSTM) part of the model in [model.py](https://github.com/ryabchenko-a/Udacity-CVND/blob/main/Image%20Captioning/model.py).

### [2_Training.ipynb](https://github.com/ryabchenko-a/Udacity-CVND/blob/main/Image%20Captioning/2_Training.ipynb)

The third notebook shows the training pipeline.

### [3_Inference.ipynb](https://github.com/ryabchenko-a/Udacity-CVND/blob/main/Image%20Captioning/3_Inference.ipynb)
![Caption Example](https://user-images.githubusercontent.com/61123874/119670268-0b7bad00-be39-11eb-88eb-3f0c9a7a9650.png)

The last notebook presents the sampling method (implemented in [model.py](https://github.com/ryabchenko-a/Udacity-CVND/blob/main/Image%20Captioning/model.py)),
that is gready search or beam search, and shows the results.
