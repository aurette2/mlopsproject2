import streamlit as st
import requests
import streamlit.components.v1 as components
from PIL import Image
import os
import pandas as pd

# Base URL of FastAPI backend
BASE_URL = "http://localhost:8000"  # Update with your backend URL

# Streamlit App
st.set_page_config(page_title="Medical Image Segmentation", layout="wide")

# Initialize session state for authentication and login status
if 'access_token' not in st.session_state:
    st.session_state.access_token = None

if 'is_logged_in' not in st.session_state:
    st.session_state.is_logged_in = False

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
        else:
            response = requests.post(f"{BASE_URL}{endpoint}", headers=headers, json=json, params=params)
    return response

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
            # Display uploaded images
            st.write("FLAIR Image:", uploaded_flair.name)
            st.write("T1CE Image:", uploaded_t1ce.name)

            # Prediction
            if st.button("View Predictions"):
                with st.spinner("Generating segmentation plot..."):
                    files = {
                        "flair": (uploaded_flair.name, uploaded_flair, uploaded_flair.type),
                        "t1ce": (uploaded_t1ce.name, uploaded_t1ce, uploaded_t1ce.type)
                    }
                    response = authenticated_request("/predictbypath/", method="POST", files=files)
                    if response.status_code == 200:
                        st.success("Prediction successful")
                        # You can handle displaying the prediction results here
                    else:
                        st.error(f"Failed to predict segmentation: {response.text}")

            # Display Segmented Predictions
            st.subheader("View Predicted Segmentations")
            try:
                slice_to_plot = st.slider("Select Slice to Plot", min_value=0, max_value=100, value=60)
                if st.button("Show Predicted Segmentations"):
                    with st.spinner("Generating segmentation plot..."):
                        files = [
                            ("files", (uploaded_flair.name, uploaded_flair, uploaded_flair.type)),
                            ("files", (uploaded_t1ce.name, uploaded_t1ce, uploaded_t1ce.type))
                        ]
                        show_segmented_response = authenticated_request("/showPredictSegmented/", method="POST", files=files, params={"slice_to_plot": slice_to_plot})
                        if show_segmented_response.status_code == 200:
                            st.success("Predicted segmentations displayed")
                            # You may want to show images or segmented results here
                        else:
                            st.error(f"Failed to show predicted segmentations: {show_segmented_response.text}")
            except ValueError:
                st.error("Please provide a valid list of samples")
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