#!/bin/bash

# Generate a dynamic SECRET_KEY using Python
SECRET_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")

# Export the environment variables
export SECRET_KEY=$SECRET_KEY

export MODELS_DIR="/Users/charme/Documents/Gitub_Repositories/MLOps_project_group3/project2/modelops"
export DATASET_BASE_PATH="/Users/charme/Documents/Gitub_Repositories/MLOps_project_group3/project2/dataops/BraTS2020/BraTS2020_TrainingData"

# Print the generated secret key (for testing)
echo "Generated SECRET_KEY: $SECRET_KEY"

fastapi run /app/main.py --host 0.0.0.0 --port 8000 & sleep 5 && streamlit run /app/app.py --server.port 8501 --server.address 0.0.0.0