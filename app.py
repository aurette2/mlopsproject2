import streamlit as st
import requests
import streamlit.components.v1 as components
from PIL import Image
import nibabel as nib
import os
import cv2
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Base URL of FastAPI backend
BASE_URL = "http://localhost:8000"  # Update with your backend URL
VOLUME_SLICES = 100
VOLUME_START_AT = 22 # first slice of volume that we will include
IMG_SIZE=128


# Streamlit App
st.set_page_config(page_title="Medical Image Segmentation", layout="wide")

# Initialize session state for authentication and login status
if 'access_token' not in st.session_state:
    st.session_state.access_token = None

if 'is_logged_in' not in st.session_state:
    st.session_state.is_logged_in = False

# Initialize a state to store prediction results
if "prediction" not in st.session_state:
    st.session_state.prediction = None
    
if "seg_prediction" not in st.session_state:
    st.session_state.seg_prediction = None
    
if "flair_data" not in st.session_state:
    st.session_state.flair_data = None
    st.session_state.flair_path = None
    
if "t1ce_data" not in st.session_state:
    st.session_state.t1ce_data = None
    st.session_state.t1ce_path = None

# Authentication function
def login(username, password):
    try:
        response = requests.post(f"{BASE_URL}/token", data={"username": username, "password": password})
        if response.status_code == 200:
            st.session_state.access_token = response.json().get('access_token')
            st.session_state.username = username
            st.session_state.is_logged_in = True
            st.success(f"Welcome, {username}!")
        else:
            st.error("Invalid username or password")
    except Exception as e:
        st.error(f"Error logging in: {e}")

# Logout function
def logout():
    st.session_state.access_token = None
    st.session_state.is_logged_in = False
    st.success("Logged out successfully!")

# Check if user is authenticated
def is_authenticated():
    return st.session_state.access_token is not None

# Function to show drift (placeholder)
def show_drift():
    headers = {"Authorization": f"Bearer {st.session_state.access_token}"}
    response = requests.get(f"{BASE_URL}/showdrift/", headers=headers)
    if response.status_code == 200:
       st.components.v1.html(response.text, height=1000, scrolling=True)
    else:
        st.error("Error in fetching drift status.")

# Function to send authenticated request, updated to support file uploads and query parameters
def authenticated_request(endpoint, method="GET", params=None, json=None, files=None):
    headers = {"Authorization": f"Bearer {st.session_state.access_token}"}
    if method == "GET":
        response = requests.get(f"{BASE_URL}{endpoint}", headers=headers, params=params)
    elif method == "POST":
        if files:
            response = requests.post(f"{BASE_URL}{endpoint}", headers=headers, files=files, params=params)
            print("Streamlit_response", response)
        else:
            response = requests.post(f"{BASE_URL}{endpoint}", headers=headers, json=json, params=params)
    return response

# Create a directory to store the images if it doesn't exist
def save_plot_as_image(fig, file_name):
    image_dir = "saved_predictions"
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    file_path = os.path.join(image_dir, file_name)
    fig.savefig(file_path)
    return file_path


def showPredictsFromFile(p, origImage, start_slice=60):
    core, edema, enhancing = p[:,:,:,1], p[:,:,:,2], p[:,:,:,3]
    fig, axarr = plt.subplots(1, 6, figsize=(18, 10))

    # Show the original image slice
    for i in range(6):
        axarr[i].imshow(cv2.resize(origImage[:,:,start_slice], (IMG_SIZE, IMG_SIZE)), cmap="gray", interpolation='none')

    # Titles and images for each subplot
    axarr[0].set_title('Original image flair')
    axarr[2].imshow(p[start_slice,:,:,1:4], cmap="Reds", alpha=0.3)
    axarr[2].set_title('All classes predicted')
    axarr[3].imshow(edema[start_slice,:,:], cmap="OrRd", alpha=0.3)
    axarr[3].set_title('Edema predicted')
    axarr[4].imshow(core[start_slice,:,:], cmap="OrRd", alpha=0.3)
    axarr[4].set_title('Core predicted')
    axarr[5].imshow(enhancing[start_slice,:,:], cmap="OrRd", alpha=0.3)
    axarr[5].set_title('Enhancing predicted')
    
    # Render the plot in Streamlit
    st.pyplot(fig)
    return save_plot_as_image(fig, f"prediction_{start_slice}.png")

def show_predicted_segmentations(p, slice_to_plot=60, cmap='gray', norm=None):

    all_classes = p[slice_to_plot, :, :, 1:4]
    background = p[slice_to_plot, :, :, 0]
    core = p[slice_to_plot, :, :, 1]
    edema = p[slice_to_plot, :, :, 2]
    enhancing = p[slice_to_plot, :, :, 3]

    fig_seg, axes = plt.subplots(1, 6, figsize=(30, 10))

    axes[1].imshow(all_classes, cmap=cmap, norm=norm)
    axes[1].set_title('Predicted Segmentation - All Classes')
    axes[2].imshow(background)
    axes[2].set_title('Predicted Segmentation - Not Tumor')
    axes[3].imshow(core)
    axes[3].set_title('Predicted Segmentation - Necrotic/Core')
    axes[4].imshow(edema)
    axes[4].set_title('Predicted Segmentation - Edema')
    axes[5].imshow(enhancing)
    axes[5].set_title('Predicted Segmentation - Enhancing')

    plt.subplots_adjust(wspace=0.8)
    st.pyplot(fig_seg)
    return save_plot_as_image(fig_seg, f"segmentation_{slice_to_plot}.png")

# ---- MAIN APP LOGIC ----

# Only show login page if the user isn't authenticated yet
if not is_authenticated() and not st.session_state.is_logged_in:
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        login(username, password)
        # Rerun the app to apply the state change
        st.rerun()
             

# If authenticated, show navigation and operations
if is_authenticated() and st.session_state.is_logged_in:
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Welcome", "Segmentation Prediction", "Model Evaluation", "Drift Detection", "Logout"])

    # Welcome Page
    if page == "Welcome":
        st.title(f"Welcome, {st.session_state.username}!")
        st.subheader("Available Operations")
        st.write("- **Segmentation Prediction**: Upload a medical image and predict segmentation.")
        st.write("- **Model Evaluation**: Evaluate the model on the test dataset.")
        st.write("- **Drift Detection**: Check for any data drift.")
        st.write("- **Logout**: End your session.")

    # Segmentation Prediction Page
    if page == "Segmentation Prediction":
        st.title("Segmentation Prediction")
        st.write("Upload a medical image case and make segmentation predictions.")

        # Upload flair and t1ce files
        uploaded_flair = st.file_uploader("Choose a FLAIR image (filename should end with _flair.nii)", type=["nii"])
        uploaded_t1ce = st.file_uploader("Choose a T1CE image (filename should end with _t1ce.nii)", type=["nii"])

        if uploaded_flair and uploaded_t1ce:
            st.write("FLAIR Image:", uploaded_flair.name)
            st.write("T1CE Image:", uploaded_t1ce.name)

            # Ensure the uploaded files are not empty
            if uploaded_flair.size == 0 or uploaded_t1ce.size == 0:
                st.error("One or more of the uploaded files is empty. Please upload valid files.")
                
            else:
                
                if st.button("View Predictions"):
                    
                    with st.spinner("Generating segmentation plot..."):
                        
                        # Save files temporarily
                        with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as flair_tempfile:
                            flair_tempfile.write(uploaded_flair.read())
                            flair_tempfile.flush()
                            flair_path = flair_tempfile.name

                        with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as t1ce_tempfile:
                            t1ce_tempfile.write(uploaded_t1ce.read())
                            t1ce_tempfile.flush()
                            t1ce_path = t1ce_tempfile.name
                            
                        # Load the saved FLAIR and T1CE files with nibabel
                        st.session_state.flair_path = flair_path
                        st.session_state.flair_data = nib.load(flair_path).get_fdata()
                        st.session_state.t1ce_path = t1ce_path
                        st.session_state.t1ce_data = nib.load(t1ce_path).get_fdata()
                        
                        # Prepare files for the request
                        files = {
                            "flair": (uploaded_flair.name, open(flair_path, 'rb'), 'application/octet-stream'),
                            "t1ce": (uploaded_t1ce.name, open(t1ce_path, 'rb'), 'application/octet-stream')
                        }

                        # Send the request to backend
                        response = authenticated_request("/predictbypath/", method="POST", files=files)

                        if response.status_code == 200:
                            st.success("Prediction successful")
                            st.session_state.prediction = np.array(response.json().get('prediction'))
                            
                            showPredictsFromFile(st.session_state.prediction, st.session_state.flair_data, start_slice=60)

                        else:
                            st.error(f"Failed to predict segmentation: {response.text}")

                if st.session_state.prediction is not None:
                    if st.button("Previous Prediction Image"):
                        saved_prediction_path = os.path.join("saved_predictions", f"prediction_{60}.png")
                        if os.path.exists(saved_prediction_path):
                            st.image(saved_prediction_path, caption="Previous Prediction")
                        else:
                            st.warning("No previous prediction image found.")
                            
                # Display Segmented Predictions
                st.subheader("View Predicted Segmentations")
                slice_to_plot = st.slider("Select Slice to Plot", min_value=10, max_value=100, value=60)
                
                if st.button("Show Predicted Segmentations"):
                    with st.spinner("Generating segmentation plot..."):
                        
                        files = {
                            "flair": (uploaded_flair.name, open(st.session_state.flair_path, 'rb'), 'application/octet-stream'),
                            "t1ce": (uploaded_t1ce.name, open(st.session_state.flair_path, 'rb'), 'application/octet-stream')
                        }
                        show_segmented_response = authenticated_request(
                            "/showPredictSegmented/", 
                            method="POST", 
                            files=files,
                            params={"slice_to_plot": slice_to_plot}
                        )

                        if show_segmented_response.status_code == 200:
                            st.success("Predicted segmentations displayed")
                            st.session_state.seg_prediction = np.array(show_segmented_response.json().get('prediction'))
                            
                            show_predicted_segmentations(st.session_state.seg_prediction, slice_to_plot)
                                                       
                        else:
                            st.error(f"Failed to show predicted segmentations: {show_segmented_response.text}")
                
                if st.session_state.seg_prediction is not None:
                    if st.button("Previous Segmentation Image"):
                        saved_segmentation_path = os.path.join("saved_predictions", f"segmentation_{60}.png")
                        if os.path.exists(saved_segmentation_path):
                            st.image(saved_segmentation_path, caption="Previous Segmentation")
                        else:
                            st.warning("No previous segmentation image found.")
                            
        else:
            st.warning("Please upload both _flair.nii and _t1ce.nii files.")

    # Model Evaluation Page
    if page == "Model Evaluation":
        st.title("Model Evaluation")
        st.write("Evaluate the segmentation model on the test dataset.")
        
        if st.button("Evaluate Model"):
            response = authenticated_request("/evaluate/", method="POST")
            if response.status_code == 200:
                st.subheader("Model Evaluation Metrics")
                # Assuming response.text contains your JSON-like string
                response_json = response.json()  # Convert JSON string to a dictionary

                # Convert dictionary to DataFrame
                df = pd.DataFrame(list(response_json.items()), columns=["Metric", "Value"],index=None)

                # Display the DataFrame as a table
                st.table(df.style.hide(axis='index'))
            else:
                st.error("Failed to evaluate model")

    # Drift Detection Page
    if page == "Drift Detection":
        st.title("Drift Detection")
        st.write("Monitor the model for data drift.")

        if st.button("Check for Drift"):
            show_drift()

    # Logout page
    if page == "Logout":
        logout()