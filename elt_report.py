import os
import pandas as pd
import numpy as np
import nibabel as nib
from PIL import Image

from dotenv import load_dotenv
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
# from alibi_detect.cd import KSDrift, ChiSquareDrift

from load_data import Datasource
from eda import DataGenerator

load_dotenv()
DATASET_BASE_PATH=os.getenv("DATASET_BASE_PATH")
PATH_FOR_DRIFT_REPORT=os.getenv("PATH_FOR_DRIFT_REPORT")
WINDOWS_SIZE=800

# Get a train data and test data windows 
source_for_drift = Datasource()
train_and_test_ids = source_for_drift.pathListIntoIds()

# Function to calculate the characteristics of an image
def compute_features(image_data):
    features = {
        'mean': np.mean(image_data),
        'std': np.std(image_data),
        'min': np.min(image_data),
        'max': np.max(image_data),
        'mean_axial': np.mean(image_data, axis=(0, 1)).mean(),  # Moyenne dans le plan axial
        'mean_coronal': np.mean(image_data, axis=(0, 2)).mean(),  # Moyenne dans le plan coronal
        'mean_sagittal': np.mean(image_data, axis=(1, 2)).mean()  # Moyenne dans le plan sagittal
    }
    return features

# Function to load .nii files and calculate features
def load_images(test_ids, train_dataset_path, windows=None):
    features_data = []
    
    # Check if windows is None
    if windows is None:
        # If windows is None, process all test_ids
        ids_to_process = test_ids
    else:
        # If windows is specified, process only up to windows
        ids_to_process = test_ids[:windows]


    for i in ids_to_process:
        case_path = os.path.join(train_dataset_path, i)

        # Load .nii files
        flair_path = os.path.join(case_path, f'{i}_flair.nii')
        flair = nib.load(flair_path).get_fdata()

        t1ce_path = os.path.join(case_path, f'{i}_t1ce.nii')
        t1ce = nib.load(t1ce_path).get_fdata()

        seg_path = os.path.join(case_path, f'{i}_seg.nii')
        seg = nib.load(seg_path).get_fdata()

        # Calculate the characteristics for each modality
        flair_features = compute_features(flair)
        t1ce_features = compute_features(t1ce)
        seg_features = compute_features(seg)
        
        features_data.append({
            'flair_mean': flair_features['mean'],
            'flair_std': flair_features['std'],
            'flair_min': flair_features['min'],
            'flair_max': flair_features['max'],
            'flair_mean_axial': flair_features['mean_axial'],
            'flair_mean_coronal': flair_features['mean_coronal'],
            'flair_mean_sagittal': flair_features['mean_sagittal'],
            't1ce_mean': t1ce_features['mean'],
            't1ce_std': t1ce_features['std'],
            't1ce_min': t1ce_features['min'],
            't1ce_max': t1ce_features['max'],
            't1ce_mean_axial': t1ce_features['mean_axial'],
            't1ce_mean_coronal': t1ce_features['mean_coronal'],
            't1ce_mean_sagittal': t1ce_features['mean_sagittal'],
            'seg_mean': seg_features['mean'],
            'seg_std': seg_features['std'],
            'seg_min': seg_features['min'],
            'seg_max': seg_features['max'],
            'seg_mean_axial': seg_features['mean_axial'],
            'seg_mean_coronal': seg_features['mean_coronal'],
            'seg_mean_sagittal': seg_features['mean_sagittal']
        })

        # Add features to a list
        # features_data.append({
        #     'flair': flair_features,
        #     't1ce': t1ce_features,
        #     'seg': seg_features
        # })

    # Create a DataFrame from the features
    # df = pd.json_normalize(features_data)
    df = pd.DataFrame(features_data)
    return df


def generate_drift_report():
    
    df_train_ref = load_images(source_for_drift.train_ids, DATASET_BASE_PATH)
    df_test_actual = load_images( source_for_drift.test_ids, DATASET_BASE_PATH, WINDOWS_SIZE)
        
    # Create an Evidently report
    report = Report(metrics=[
        DataDriftPreset(num_stattest="ks", stattest_threshold=0.05),
        DataQualityPreset()
    ])
    
    # Run the report on the datasets
    report.run(reference_data=df_train_ref, current_data=df_test_actual)
    # Save the report as HTML
    report.save_html(os.path.join(PATH_FOR_DRIFT_REPORT, 'drift_seg_report.html'))

    return report