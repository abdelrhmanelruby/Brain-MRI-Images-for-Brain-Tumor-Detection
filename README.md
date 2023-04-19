# Brain-MRI-Tumor-and-Alzheimer-Classification
Please visit the following link to access the demo version: [Brain-MRI-Tumor-and-Alzheimer-Classification](https://huggingface.co/spaces/Longliveruby/Brain-MRI-Tumor-and-Alzheimer-Classification)

## Brief overview
The goal of this project is to create a deep convolutional neural network model that can classify brain tumors and Alzheimer's disease using MRI scan images.

The architecture is a fully convolutional network (FCN) built upon several well-known models like ResNet50V2, VGG16, and EfficientNetb5.

## Data
For the main model, we used [Brain Tumor MRI Images 44 Classes](https://www.kaggle.com/datasets/fernando2rad/brain-tumor-mri-images-44c) a collection of T1, contrast-enhanced T1, and T2 magnetic resonance images separated by brain tumor type. Contains a total of 4479 images and 44 classes.

We used this dataset to train our main CNN model and then tested it on different datasets. We used the same model and weights as the main model, with the only difference being the output layer. 

### Testing datasets 
- [Brain Tumor MRI Images 44 Classes](https://www.kaggle.com/datasets/fernando2rad/brain-tumor-mri-images-44c) using only tumor types 4479 images and 15 classes
- [Brain Tumor MRI Images 17 Classes](https://www.kaggle.com/datasets/fernando2rad/brain-tumor-mri-images-17-classes) contains 4448 images and 17 classes
- [Brain Tumor Classification (MRI)](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri) contains 3264 images and 4 classes
- [Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)contains 253 images and 2 classes
- [Brain_Tumor_Detection_MRI](https://www.kaggle.com/datasets/abhranta/brain-tumor-detection-mri) contains 3060 images and 2 classes
- [Alzheimer MRI Preprocessed Dataset](https://www.kaggle.com/datasets/sachinkumar413/alzheimer-mri-dataset) contains 6400 images and 2 classes

#### Sample images from Brain Tumor MRI Images 44 Classes dataset
![sample_44](https://user-images.githubusercontent.com/107134115/232938604-3b07397e-6168-4906-a898-9c405a0c347a.png)
![distribution_44](https://user-images.githubusercontent.com/107134115/232927885-a38a2138-d3e5-48a1-8d6f-7489ff35ab45.png)


# Sample images from Alzheimer MRI Preprocessed Dataset
![sample_alzheimer](https://user-images.githubusercontent.com/107134115/232930087-96ffa33f-b609-400e-ba17-f87248d157f2.png)
![distribution_alzheimer](https://user-images.githubusercontent.com/107134115/232930063-2f0e0027-c4bf-4a2c-a7a4-290ca1f0d973.png)

## Data augmentation
```python
rescale=1./255,
rotation_range=20,
width_shift_range=0.1,
height_shift_range=0.1,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True,
fill_mode='nearest'
```
- **Horizontal flip:** basically, flips both rows and columns horizontally. 
- **Rotation_range:** is a value in degrees (0-180), a range within which to randomly rotate pictures.
- **Rescale:** is a value by which we will multiply the data before any other processing. Our original images consist in RGB coefficients in the 0-255, but such values would be too high for our models to process (given a typical learning rate), so we target values between 0 and 1 instead by scaling with a 1/255. factor.
- **Width_shift and Height_shift:** are ranges (as a fraction of total width or height) within which to randomly translate pictures vertically or horizontally.
- **Zoom_range:** is for randomly zooming inside pictures.
- **Shear_range:** is for randomly applying shearing transformations.
- **Fill_mode:** is the strategy used for filling in newly created pixels, which can appear after a rotation or a width/height shift.


# Requirements
To run the code, you first need to install the following prerequisites:
```python
Python 3.5 or above
tensorflow==2.9.1 
keras==2.9.0
streamlit==1.14.0
streamlit_option_menu==0.3.2
numpy
pandas
matplotlib 
```
    
# Result

## Brain Tumor MRI Images 44 Classes
![EfficientNetB5_44_classes](https://user-images.githubusercontent.com/107134115/232933100-41f1b858-73c8-4432-9952-6c5d575c088a.png)
**96.5% accuracy on the test set**
## Brain Tumor MRI Images 17 Classes
![EfficientNetB5_17calss](https://user-images.githubusercontent.com/107134115/232933258-19252edc-a718-4813-b8f6-318ee95b54e0.png)
**98.1% accuracy on the test set**
## Brain Tumor MRI Images 15 Classes
![EfficientNetB5_15calss](https://user-images.githubusercontent.com/107134115/232933343-ef0858b1-e87f-4958-8846-777811b494c9.png)
**99.8% accuracy on the test set**
## Brain_Tumor_Detection_MRI 2 Classes
![EfficientNetB5_2calss_lagre_dataset](https://user-images.githubusercontent.com/107134115/232933464-853a75b0-e931-4a70-9785-6ab44756aa26.png)
**99.1% accuracy on the test set**
## Alzheimer MRI Preprocessed Dataset 4 Classes
![alzheimer](https://user-images.githubusercontent.com/107134115/232934983-ed4adbdd-a5e5-4f7e-ba98-0636a1905f86.png)
**99.5% accuracy on the test set**

# Deployment
Please visit the following link to access the app's demo version: https://huggingface.co/spaces/Longliveruby/Brain-MRI-Tumor-and-Alzheimer-Classification
The website can be accessed and tested out there. Due to the limitations of file sizes and RAM limits, I decided to go with
[huggingface](https://huggingface.co/) because the free version is not severely limited.  

You can test the app on localhost by cloning the repository data, cd into the folder and run the following commands:
```python
cd Streamlit
streamlit run main.py
```
Installing dependencies:
```python
pip install -r requirements.txt
```
## Contributors
- AbdElRahman Elruby [Linkedin](https://www.linkedin.com/in/abdelrhmanelruby/) | [Github](https://github.com/abdelrhmanelruby)
- Marwa Shaaban AbdElhakeem [Linkedin](https://www.linkedin.com/in/marwa-shaaban-abd-elhakim/) | [Github](https://github.com/Marwa-Shaaban)
- Yara Yasser Farouk
- Salma Mahmoud Fahim



# Thank you!
