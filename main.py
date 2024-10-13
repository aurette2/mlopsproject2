import os
from typing import List
import streamlit as st
import numpy as np

from elt_report import generate_drift_report

from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from datetime import timedelta
from pydantic import BaseModel
import os
from auth import (
    authenticate_user,
    create_access_token,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    decode_token,
    oauth2_scheme
)
# from model import predictByPath, showPredictsById, show_predicted_segmentations, evaluate
from model import Unet
from load_data import Datasource
from eda import DataGenerator
from config import MODELS_DIR, DRIFT_BASE_PATH
from elt_report import generate_drift_report
app = FastAPI()

source = Datasource()
train_and_test_ids = source.pathListIntoIds()
test_generator = DataGenerator(source.test_ids)

# Initialize the Unet model (set appropriate parameters)
unet_model = Unet(img_size=128, num_classes=4)
unet_model.compile_and_load_weights( os.path.join(MODELS_DIR,'my_model.keras') )

@app.post("/")
async def hello():
    # Placeholder logic for drift detection
    return {"message": "Welcome"}

# Token endpoint to login and obtain a JWT token
@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, role=user["role"], expires_delta=access_token_expires
    )
    return {"access_token": access_token, "role":user["role"],  "token_type": "bearer"}

# # Secured endpoint to retrieve user information
@app.get("/users/me/")
async def read_users_me(token: str = Depends(oauth2_scheme)):
    current_user = decode_token(token)
    return current_user

#Endpoint to get case
@app.get("/case")
async def get_case(num: int = 0):
    return source.test_ids[num][-3:]

#Endpoint to get samples_list
@app.get("/samples_list")
async def get_samples_list():
    return source.test_ids


# Endpoint to evaluate the model on test data
@app.post("/evaluate/")
async def evaluate_model_api(token: str = Depends(oauth2_scheme)):
    payload = decode_token(token)
    role = payload.get("role")
    
    if role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions")
    
    try:
        # Evaluate the model
        decode_token(token)
        metrics_dict = unet_model.evaluate(test_generator)
        print(metrics_dict)
        return metrics_dict
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to show drift (placeholder)
@app.get("/showdrift/")
async def show_drift(token: str = Depends(oauth2_scheme)):
    payload = decode_token(token)
    role = payload.get("role")
    
    if role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions")
    
    try:
        # report_html_path = "custom_report.html"
        report_html_path = DRIFT_BASE_PATH + "drift_seg_report.html"
        print(report_html_path)
    # Check if the file exists
        if os.path.exists(report_html_path):
            # Read the HTML file and return as a response
            with open(report_html_path, "r") as f:
                html_content = f.read()
            return HTMLResponse(content=html_content)
        else:
            # Generate It from data / This should take time
            generate_drift_report()
            with open(report_html_path, "r") as f:
                html_content = f.read()
            return HTMLResponse(content=html_content)
        
    except Exception as e:
        return {"error": str(e)}


@app.post("/predictbypath/")
async def predict(flair: UploadFile = File(...), t1ce: UploadFile = File(...)):
    try:
        # Save and process FLAIR and T1CE files separately
        flair_file_path = f"{flair.filename}"
        t1ce_file_path = f"{t1ce.filename}"

        with open(flair_file_path, "wb") as f:
            f.write(await flair.read())
        
        with open(t1ce_file_path, "wb") as f:
            f.write(await t1ce.read())

        # Call the prediction method with the paths to both files
        prediction = unet_model.predictFromFiles(flair_file_path, t1ce_file_path)

        # Return the prediction as a list (to handle numpy arrays)
        return {"prediction": prediction.tolist()}
    
    except Exception as e:
        # If any error occurs, raise an HTTP exception with the error details
        raise HTTPException(status_code=500, detail=str(e))
    
   
@app.post("/showPredictSegmented/")
async def show_predicted_segmentations_api(flair: UploadFile = File(...), t1ce: UploadFile = File(...)):
    try:
        flair_file_path = f"{flair.filename}"
        t1ce_file_path = f"{t1ce.filename}"

        # Save files
        with open(flair_file_path, "wb") as f:
            f.write(await flair.read())

        with open(t1ce_file_path, "wb") as f:
            f.write(await t1ce.read())

        # Ensure both flair and t1ce files were uploaded
        if not flair_file_path or not t1ce_file_path:
            raise HTTPException(status_code=400, detail="Both _flair.nii and _t1ce.nii files must be provided.")
        
        # Call the prediction method with the paths to both files
        prediction = unet_model.predict_segmentation(flair_file_path, t1ce_file_path)

        prediction_list = np.array(prediction).tolist()

        # return JSONResponse(content={"prediction": prediction_list})
        return {"prediction": prediction_list}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))   
    
    