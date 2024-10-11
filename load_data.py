from dotenv import load_dotenv
import os
import cv2
import random
import glob
import PIL
import shutil
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from skimage import data
from skimage.util import montage
import skimage.transform as skTrans
from skimage.transform import rotate
from skimage.transform import resize
from PIL import Image, ImageOps
import nibabel as nib
import keras
# import keras.backend as K
import tensorflow.keras.backend as K
from keras.callbacks import CSVLogger
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard

load_dotenv()


class Datasource:
    global TRAIN_DATASET_PATH, slice_tumor, test_image_flair, test_image_t1, test_image_t1ce, test_image_t2, test_image_seg, cmap, norm 
    cmap = matplotlib.colors.ListedColormap(['#440054', '#3b528b', '#18b880', '#e6d74f'])
    norm = matplotlib.colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    TRAIN_DATASET_PATH = os.getenv('DATASET_BASE_PATH')
    slice_tumor = 95
    
    def __init__(self):
        self.scaler = MinMaxScaler()
        

    # def download_dataset(self, dataset, download_path):
    #     """Download a Kaggle dataset to a specified path.

    #     Args:
    #         dataset (str): The Kaggle dataset identifier (e.g., 'awsaf49/brats20-dataset-training-validation').
    #         download_path (str): The directory path to download the dataset to.
    #     """
    #     if not os.path.exists(download_path):
    #         os.makedirs(download_path)
        
    #     api = KaggleApi()
    #     api.authenticate()
    #     api.dataset_download_files(dataset, path=download_path, unzip=True)
    #     print(f"Dataset {dataset} downloaded to {download_path}")
        
    def rename_file(self):
        old_name = TRAIN_DATASET_PATH + "BraTS20_Training_355/W39_1998.09.19_Segm.nii"
        new_name = TRAIN_DATASET_PATH + "BraTS20_Training_355/BraTS20_Training_355_seg.nii"

        # renaming the file
        try:
            os.rename(old_name, new_name)
            print("File has been re-named successfully!")
        except:
            print("File is already renamed!")
            
    def load_nii_as_narray(self):
        """
            load .nii file as a numpy array,
            Rescaling pixel values is essential
        """
        
        test_image_flair = nib.load(TRAIN_DATASET_PATH +  "/BraTS20_Training_355/BraTS20_Training_355_flair.nii").get_fdata()
        print("Shape: ", test_image_flair.shape)
        print("Dtype: ", test_image_flair.dtype)
        print("Min: ", test_image_flair.min())
        print("Max: ", test_image_flair.max())
        
        # Scale the test_image_flair array and then reshape it back to its original dimensions.
        # This ensures the data is normalized/standardized for model input without altering its spatial structure.
        test_image_flair = self.scaler.fit_transform(test_image_flair.reshape(-1, test_image_flair.shape[-1])).reshape(test_image_flair.shape)
        print("Min: ", test_image_flair.min())
        print("Max: ", test_image_flair.max())
        
        # rescaling t1
        test_image_t1 = nib.load(TRAIN_DATASET_PATH +  '/BraTS20_Training_355/BraTS20_Training_355_t1.nii').get_fdata()
        test_image_t1 = self.scaler.fit_transform(test_image_t1.reshape(-1, test_image_t1.shape[-1])).reshape(test_image_t1.shape)

        # rescaling t1ce
        test_image_t1ce = nib.load(TRAIN_DATASET_PATH +  '/BraTS20_Training_355/BraTS20_Training_355_t1ce.nii').get_fdata()
        test_image_t1ce = self.scaler.fit_transform(test_image_t1ce.reshape(-1, test_image_t1ce.shape[-1])).reshape(test_image_t1ce.shape)

        # rescaling t2
        test_image_t2 = nib.load(TRAIN_DATASET_PATH +  '/BraTS20_Training_355/BraTS20_Training_355_t2.nii').get_fdata()
        test_image_t2 = self.scaler.fit_transform(test_image_t2.reshape(-1, test_image_t2.shape[-1])).reshape(test_image_t2.shape)

        # we will not rescale the mask
        self.test_image_seg = nib.load(TRAIN_DATASET_PATH +  '/BraTS20_Training_355/BraTS20_Training_355_seg.nii').get_fdata()
        print("Slice Number: " + str(slice_tumor))
        
        plt.figure(figsize=(12, 8))
        self.show_img_feature( test_image_t1, 'T1')
        self.show_img_feature( test_image_t1ce, 'T1ce')
        self.show_img_feature( test_image_t2, 'T2')
        self.show_img_feature( test_image_flair, 'FLAIR')
        self.show_img_feature( self.test_image_seg, 'Mask')
        
        
        # the modalities and segmentations have 3 dimensions; quick presentation of these 3 planes
        self.show_img_plane( test_image_t1ce)
        
    def expert_segmentation(self):
        # We have to check that all those arrays are not empty.
        # Isolation of class 0
        seg_0 = self.test_image_seg.copy()
        seg_0[seg_0 != 0] = np.nan

        # Isolation of class 1
        seg_1 = self.test_image_seg.copy()
        seg_1[seg_1 != 1] = np.nan

        # Isolation of class 2
        seg_2 = self.test_image_seg.copy()
        seg_2[seg_2 != 2] = np.nan

        # Isolation of class 4
        seg_4 = self.test_image_seg.copy()
        seg_4[seg_4 != 4] = np.nan

        # Define legend
        class_names = ['class 0', 'class 1', 'class 2', 'class 4']
        legend = [plt.Rectangle((0, 0), 1, 1, color=cmap(i), label=class_names[i]) for i in range(len(class_names))]

        fig, ax = plt.subplots(1, 5, figsize=(20, 20))

        ax[0].imshow(self.test_image_seg[:,:, slice_tumor], cmap=cmap, norm=norm)
        ax[0].set_title('Original Segmentation')
        ax[0].legend(handles=legend, loc='lower left')

        ax[1].imshow(seg_0[:,:, slice_tumor], cmap=cmap, norm=norm)
        ax[1].set_title('Not Tumor (class 0)')

        ax[2].imshow(seg_1[:,:, slice_tumor], cmap=cmap, norm=norm)
        ax[2].set_title('Non-Enhancing Tumor (class 1)')

        ax[3].imshow(seg_2[:,:, slice_tumor], cmap=cmap, norm=norm)
        ax[3].set_title('Edema (class 2)')

        ax[4].imshow(seg_4[:,:, slice_tumor], cmap=cmap, norm=norm)
        ax[4].set_title('Enhancing Tumor (class 4)')

        plt.show()  

    def pathListIntoIds(self, data_path=TRAIN_DATASET_PATH ):
        # lists of directories with studies
        dirList = [f.path for f in os.scandir(data_path) if f.is_dir()]
        x = []
        for i in range(0,len(dirList)):
            x.append(dirList[i][dirList[i].rfind('/')+1:])
            
        self.train_test_ids, self.val_ids = train_test_split(x,test_size=0.2)
        self.train_ids, self.test_ids = train_test_split(self.train_test_ids,test_size=0.15)
        
        return x 
    
    def plot_train_val_test_frequence(self):
        plt.bar(["Train","Valid","Test"],
        [len(self.train_ids), len(self.val_ids), len(self.test_ids)],
        align='center',
        color=[ 'green','red', 'blue'],
        label=["Training Sample", "Validation sample", "Testing sample"]
       )

        plt.legend()

        plt.ylabel('Number of Images')
        plt.title('Data Distribution')

        plt.show()   
    
    def show_img_feature(self, arr, title):
        plt.subplot(2, 3, 1)
        plt.imshow(arr[:,:,slice_tumor], cmap='gray')
        plt.title(title)
        
    def show_img_plane(self, arr):
        # Apply a 90° rotation with an automatic resizing, otherwise the display is less obvious to analyze
        slice = slice_tumor

        print("Slice number: " + str(slice))

        plt.figure(figsize=(12, 8))

        # Apply a 90° rotation with an automatic resizing, otherwise the display is less obvious to analyze
        # T1 - Transverse View
        plt.subplot(1, 3, 1)
        plt.imshow(arr[:,:,slice], cmap='gray')
        plt.title('T1 - Transverse View')

        # T1 - Frontal View
        plt.subplot(1, 3, 2)
        plt.imshow(rotate(arr[:,slice,:], 90, resize=True), cmap='gray')
        plt.title('T1 - Frontal View')

        # T1 - Sagittal View
        plt.subplot(1, 3, 3)
        plt.imshow(rotate(arr[slice,:,:], 90, resize=True), cmap='gray')
        plt.title('T1 - Sagittal View')
        plt.show()
        
    def display_slice_and_segmentation(self, flair, t1ce, segmentation):
        fig, axes = plt.subplots(1, 3, figsize=(10, 5))

        axes[0].imshow(flair, cmap='gray')
        axes[0].set_title('Flair')
        axes[0].axis('off')

        axes[1].imshow(t1ce, cmap='gray')
        axes[1].set_title('T1CE')
        axes[1].axis('off')

        axes[2].imshow(segmentation) # Displaying segmentation
        axes[2].set_title('Segmentation')
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()

        
        
        
        
        
            

if __name__ == "__main__":

    # load_dotenv()
    dataset = os.getenv('DATASET_NAME')
    download_path = os.getenv('DATASET_PATH')
    
    
    source = Datasource()
    # source.download_dataset(dataset, download_path)
    source.rename_file()
    source.load_nii_as_narray()
    source.expert_segmentation()

    """ 
    # __________________________________________Data Spliting_____________________________________#
    # Split the Dataset
    train_and_test_ids = source.pathListIntoIds()
    train_test_ids, val_ids = train_test_split(train_and_test_ids,test_size=0.2)
    train_ids, test_ids = train_test_split(train_test_ids,test_size=0.15)
    """
    
    
    
    
    
    
    
    
    
    
    
    
    
