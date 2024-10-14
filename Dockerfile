# Use an official Python image as a base
FROM python:3.12

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY . .

# Install necessary system dependencies, including libGL for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
    
# Install the required Python packages
RUN pip install --upgrade -r requirements.txt

# Pull the DVC dataset
# RUN dvc pull

# Expose the ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000
EXPOSE 8501

# Set environment variables
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Generate a SECRET_KEY and export it
RUN SECRET_KEY=$(python -c "import secrets; print(secrets.token_hex(32))") && \
    export SECRET_KEY && \
    echo "Generated SECRET_KEY: $SECRET_KEY" && \
    echo "SECRET_KEY=$SECRET_KEY" > /tmp/secret_key.env

# Use the shell form of ENV to source the SECRET_KEY from the generated file
RUN . /tmp/secret_key.env && echo $SECRET_KEY && \
    echo "SECRET_KEY=$SECRET_KEY" >> /etc/environment

# Set the SECRET_KEY as an environment variable
ENV SECRET_KEY=$SECRET_KEY
ENV MODELS_DIR="./brain_data/"
ENV DATASET_BASE_PATH="./brain_data/BraTS2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"

# Optional: Clean up the temporary file
RUN rm -f /tmp/secret_key.env

# CMD /bin/bash -c "fastapi run controller.py --host 0.0.0.0 --port 8000 & sleep 5 && streamlit run test.py --server.address 0.0.0.0 --server.port 8501"
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 8000 & sleep 5 && streamlit run app.py --server.address 0.0.0.0 --server.port 8501"]