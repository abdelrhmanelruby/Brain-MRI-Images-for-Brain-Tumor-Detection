# Brain-MRI-Images-for-Brain-Tumor-Detection

# Brief overview

This repository provides source code for a deep convolutional neural network architecture designed for brain tumor classification with ( Brain-MRI-Images , alzheimer-mri ) datasets. The architecture is fully convolutional network (FCN) built upon the well-known several models like ( ResNet50V2 , VGG16 , Efficientnetb5 ). 

# Sample images from Brain-MRI-Images dataset
![image](https://user-images.githubusercontent.com/114644816/232864644-023201d3-21ec-420a-bc1b-4efdd5d548b6.png)


# Sample images from alzheimer-mri-Images dataset
![image](https://user-images.githubusercontent.com/114644816/232865091-ebcd3a52-c52b-4f2e-a8c3-a195cc0aadcb.png)


# Requirements

To run the code, you first need to install the following prerequisites:

    Python 3.5 or above
    numpy
    pandas
    keras
    os
    matplotlib 
    tensorflow 
    PIL 

# About the datasts:
The dataset (https://www.kaggle.com/datasets/fernando2rad/brain-tumor-mri-images-44c) contains 4479 files and 44 classes

The dataset (https://www.kaggle.com/datasets/fernando2rad/brain-tumor-mri-images-17-classes) contains 4448 files and 17 classes

The dataset (https://www.kaggle.com/datasets/fernando2rad/brain-tumor-mri-images-17-classes) contains 253 files and 2 classes:

The dataset (https://www.kaggle.com/datasets/abhranta/brain-tumor-detection-mri) contains 3060 files and 2 classes:

The dataset (https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri) contains 3264 files and 4 classes:

# Training the model :

# brain-tumor-mri 44 Classes
![image](https://user-images.githubusercontent.com/114644816/232872010-07978b5f-974c-4958-8b08-2782d493444e.png)
96.5% accuracy

# brain-tumor-mri 2 Classes
![image](https://user-images.githubusercontent.com/114644816/232872191-a41a0413-af87-4008-82cb-3518218656c3.png)
99.1% accuracy

# brain-tumor-mri 15 Classes
![image](https://user-images.githubusercontent.com/114644816/232872301-2522c53e-f680-430e-9154-5b4bacfc0dc0.png)
99.8% accuracy

# brain-tumor-mri 17 Classes
![image](https://user-images.githubusercontent.com/114644816/232872738-fab00c16-2489-44bc-bc1e-3fe8cf4ee4ec.png)
98.1% accuracy

# alzheimer-mri 4 classes
![image](https://user-images.githubusercontent.com/114644816/232873136-1c6b60e2-8401-4279-8eb3-66d863aece51.png)
96.5% accuracy

# Final Notes

What's in the files?

    The code in the IPython Jupyter notebooks.
    The weights for models. The best model is named as 'imagenet'.
    The models are stored as model files. 



   

# Thank you!
