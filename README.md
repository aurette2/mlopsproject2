# Brain Tumor Detection (MLOPS Project)
An AI-powered image segmentation system designed to automatically delineate tumors in brain MRI images. This system will provide accurate, consistent, and fast segmentations, improving diagnostic workflows and aiding in treatment planning.


# Project structure 

- `backend/app/`:Contains the `model.py` file in which we have defined the UNET-2D architecture, `metris.py` contains the implementation of the Unet model evaluation metrics and the segmentation task, `main.py` contains our APIs served by fastapi, `load_data.py` contains the operations for loading, processing and virsualizing images from our dataset (How to process MRI), `eda.py` we want to avoid directly loading all 3D images because they can overload system memory and cause shape mismatch errors, therefore, we use a Data Generator for image preprocessing.

- `dataops`: This folder contains all the data manipulated in the system (input data, output, temporal) and tracked using dvc.
- `frontend`: This folder contains the file `app.py` in which there is the implementation of all the user interfaces. 
- `modelops`: This folder contains the trained `Unet-2D` model named .
- `Dockerfile`: This file contains all the commands we requires to call on the command line to assemble an image
- `main.py`: Contains the workflow needed to perform the of our model (command line prediction)  with specific . 
- `requirements.txt`: Contains the specific packages to install for the project to function properly
- `airflow`: contains the elements necessary for the functioning of apache airflow. Its `/dag/` subfolder contains the `drift_dag.py` file which has the logic for generating drift reports to be orchestrated into a DAG.


# Setup

### Clone

```bash
  git clone https://github.com/aurette2/mlopsproject2
```
```
cd mlopsproject2
```
### Install requirements
- Create an activate a virtual enviroment
```
    pip install virtualenv
    virtualenv venv
    source venv/bin/activate
```

```
pip install -r requirements.txt
```

### Run it locally 
launch the backend (fastapi)
```bash
   fastapi run main.py
```

launch the streamlit ()

```bash
   streamlit run app.y
```

### Run it locally in docker image

```
    cd mlopsproject2
```

```
    docker build -t fastapi-app:latest .
```

```
    docker run -d -p 8000 -p 8501 fastapi-project:latest 
```

# Tech Stack

- **Frontend:** Streamlit
- **Backend:** Fastapi, 
- **Data Management:** DVC, Git
- **CI/CD Pipeline:** Github action 
- **Deployment:** Azure, 
- **Orchestration:**  Apache airflow,
- **Monitoring:** Evidently AI 


# Appendix
Aditionals guide and Assets that can help to better understand this project.
- [Google Drive link of projects artefacts](https://drive.google.com/drive/folders/1rI5kliXcn50TZAS0ZsWDgl56ZBZz_Nv2?usp=sharing)


## License

[MIT](https://choosealicense.com/licenses/mit/)

------------------------------------------------------------------------------------------------------------

⭐️ If you find this repository helpful, we’d be thrilled if you could give it a star! 
