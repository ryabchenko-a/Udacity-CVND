# Facial Keypoints Detection Project

In this project I've built a system that is able to recognise faces on the image and then find facial keypoints of those faces.  
1. First, it uses Haar Cascades to find the images.  
2. Then, it crops the padded area of the face and pass it to CNN to predict the coordinates of the facial key points.
3. After key points are found, they are plotted on the cropped areas for visualisation.

The project also presents a funny example of how such system may be used, particularly to add moustache/hat on the image in the right place.  

## Detailed description of the project

The project consists of 4 ipynb notebooks.

### [1. Load and Visualise Data.ipynb](https://github.com/ryabchenko-a/Udacity-CVND/blob/main/Facial%20Keypoints%20Detector/1.%20Load%20and%20Visualize%20Data.ipynb)
<img src="https://github.com/ryabchenko-a/Udacity-CVND/raw/330dc180d2519140ef03ab58a4a12bea33f6e168/Facial%20Keypoints%20Detector/images/key_pts_example.png" alt="Data Examples" height="256"/> <img src="https://github.com/ryabchenko-a/Udacity-CVND/raw/330dc180d2519140ef03ab58a4a12bea33f6e168/Facial%20Keypoints%20Detector/images/landmarks_numbered.jpg" alt="Facial Keypoints Numeration" height="256"/>

This notebook explores the dataset I work with, that is [YouTube Faces Dataset](https://www.cs.tau.ac.il/~wolf/ytfaces/). 
Facial Keypoints numeration are presented and data transform is proposed.
To get more sense of data loader, check out [data_load.py](https://github.com/ryabchenko-a/Udacity-CVND/blob/main/Facial%20Keypoints%20Detector/data_load.py).  

### [2. Define the Network Architecture.ipynb](https://github.com/ryabchenko-a/Udacity-CVND/blob/main/Facial%20Keypoints%20Detector/2.%20Define%20the%20Network%20Architecture.ipynb)
<img src="https://user-images.githubusercontent.com/61123874/119662994-41696300-be32-11eb-9ac7-ead9ef9728d2.png" alt="NaimishNet Architecture" height="512"/>

The second notebook deals with everything that is related to the CNN part of the model, that is:  
1. Final version of data transform.  
2. CNN architecture (which is written separately in [models.py](https://github.com/ryabchenko-a/Udacity-CVND/blob/main/Facial%20Keypoints%20Detector/models.py) file).  
3. Training pipeline.  
4. Results examples and first layer feature visualisation and interpretation.  

In total, 2 different CNN models were trained:  
[NaimishNet](https://arxiv.org/pdf/1710.00977.pdf) was trained semi-manually since early stop criteria was too strict at the beginning and continuing training provided better results. Architecture was slightly changed since original paper uses different data transform (for example, 96x96 images instead of 224x224).  
resnet18 from [torchvision models](https://pytorch.org/vision/stable/models.html) was also considered. The model was trained in its entirety since it improved learning speed and results when compared to training only the last dense layer of the model.  
The latter model provided much better results.

### [3. Facial Keypoint Detection, Complete Pipeline.ipynb](https://github.com/ryabchenko-a/Udacity-CVND/blob/main/Facial%20Keypoints%20Detector/3.%20Facial%20Keypoint%20Detection%2C%20Complete%20Pipeline.ipynb)
<img src="https://user-images.githubusercontent.com/61123874/119662989-40383600-be32-11eb-9347-95f6cf212da9.png" alt="Haar Cascades in action" height="256"/> <img src="https://user-images.githubusercontent.com/61123874/119662983-3e6e7280-be32-11eb-9bf6-a90b34b9ab8b.png" alt="Predicted Keypoints detected" height="256"/>

The third notebook shows the complete pipeline:  
1. Pretrained Haar Cascades is used to predict the areas that contain faces. 
2. Predicted areas are padded (so that the entire face is in the area) and cropped.  
3. The cropped images are transformed and passed to the CNN which predicts the keypoints.  
4. The keypoints are plotted on the faces.  

### [4. Fun with Keypoints.ipynb](https://github.com/ryabchenko-a/Udacity-CVND/blob/main/Facial%20Keypoints%20Detector/4.%20Fun%20with%20Keypoints.ipynb) (Optional) 
<img src="https://user-images.githubusercontent.com/61123874/119663477-aa50db00-be32-11eb-8584-058781447481.png" alt="Moustaches" height="256"/>

The last, optional notebook, contains the pipeline for adding sunglasses/moustaches/hat on the pictures using our Facial Keypoints Detection system.
